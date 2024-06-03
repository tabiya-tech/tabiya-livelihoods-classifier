from typing import List, Optional, Tuple
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
import transformers
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
#from google.cloud import translate_v2 as translate
#import ftlangdetect
from util.utilfunctions import CPU_Unpickler
from util.transformersCRF import AutoModelCrfForNer
import pickle
import os

class EntityLinker:
	"""
	Creates a pipeline of an entity recognition transformer and a sentence transformer for embedding text.

	Initialization Parameters
	----------
	entity_model : str, default='tabiya/roberta-base-job-ner'
		Path to a pre-trained `AutoModelForTokenClassification` model or an `AutoModelCrfForNer` model. 
		This model is used for entity recognition within the input text.

	similarity_model : str, default='all-MiniLM-L6-v2'
		Path or name of a sentence transformer model used for embedding text. 
		The sentence transformer is used to compute embeddings for the extracted entities and the reference sets.
		The model 'all-mpnet-base-v2' is available but not in cache, so it should be used with the parameter `from_cache=False` at least the first time.

	crf : bool, default=False
		A flag to indicate whether to use an `AutoModelCrfForNer` model instead of a standard `AutoModelForTokenClassification`. 
		`CRF` (Conditional Random Field) models are used when the task requires sequential predictions with dependencies between the outputs.

	hf_token : str, default=None
		HuggingFace token used for accessing private models. 
		This is necessary as the models are not publicly available and require authentication.

	evaluation_mode : bool, default=False
		If set to `True`, the linker will return the cosine similarity scores between the embeddings. 
		This mode is useful for evaluating the quality of the linkages.

	k : int, default=32
		Specifies the number of items to retrieve from the reference sets. 
		This parameter limits the number of top matches to consider when linking entities.

	from_cache : bool, default=True
		If set to `True`, the precomputed embeddings are loaded from cache to save time. 
		If set to `False`, the embeddings are computed on-the-fly, which requires GPU access for efficiency and can be time-consuming.

	output_format : str, default='name'
		Specifies the format of the output for occupations, either 'name' for occupation names or 'esco_code' for ESCO codes.

	Calling Parameters
	----------
	text : str
		An arbitrary job vacancy-related string.

	linking : bool, default=True
		Specify whether the model performs the entity linking to the taxonomy.
	"""

	def __init__(
			self,
			entity_model: str = 'tabiya/roberta-base-job-ner',
			similarity_model: str = 'all-MiniLM-L6-v2',
			crf: Optional[bool] = False,
			hf_token: str = None,
			evaluation_mode: bool = False,
			k: int = 32,
			from_cache: bool = True,
			output_format: str = 'name'
	):
		# Initialize the model paths and settings
		self.entity_model = entity_model
		self.similarity_model_type = similarity_model
		self.similarity_model = SentenceTransformer(similarity_model)
		self.crf = crf
		self.evaluation_mode = evaluation_mode
		self.k = k
		self.from_cache = from_cache
		self.output_format = output_format

		# Set the device to GPU if available, otherwise CPU
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# Load the appropriate entity recognition model based on the crf flag
		if self.crf:
			self.entity_model = AutoModelCrfForNer.from_pretrained(entity_model)
		else:
			self.entity_model = AutoModelForTokenClassification.from_pretrained(entity_model, token=hf_token)

		# Move the entity model to the appropriate device
		self.entity_model.to(self.device)

		# Initialize the tokenizer for the entity model
		self.tokenizer = AutoTokenizer.from_pretrained(entity_model, token=hf_token)

		# Load reference sets for occupations, skills, and qualifications
		self.df_occ = pd.read_csv('inference/files/occupations_augmented.csv')
		self.df_skill = pd.read_csv('inference/files/skills.csv')
		self.df_qual = pd.read_csv('inference/files/qualifications.csv')

		# Load precomputed embeddings for the reference sets
		self.occupation_emb, self.skill_emb, self.qualification_emb = self._load_tensors()


	def __call__(self, text: str, linking: bool = True) -> List[dict]:
		"""
		Perform job-related entity recognition and optionally link entities to a taxonomy.

		Parameters
		----------
		text : str
			An arbitrary job vacancy-related string that the model processes to extract and link entities.
			
		linking : bool, default=True
			Specifies whether the model should perform the entity linking to the taxonomy. 
			If `False`, it might only extract entities without linking them to a predefined taxonomy.

		Returns
		-------
		List[dict]
			A list of dictionaries with the extracted entities and the top-k most similar entities from the reference sets. 
			Each dictionary contains the following keys:
			- `type`: The category of the identified entity (e.g., 'Occupation', 'Qualifications', 'Skill', 'Experience').
			- `tokens`: The specific part of the input text that was identified as an entity of the right category.
			- `retrieved`: A list of related names or ESCO codes retrieved from the reference sets. 
				These items represent the most similar entities or concepts based on the embeddings and similarity calculations.
				Appear if linking=True.
			- `scores`: A list of cosine similarity scores between the extracted entity and the retrieved items.
				Appear if linking=True and evaluation_mode=True.
		"""

		# Replace newlines in the text with spaces
		text = text.replace('\n', ' ')

		# TODO: Implement the Google Translate features to enable multilingual entity linking.
		# language = UtilFunctions.detect_language(text)
		# if language != 'en':
		#     text = UtilFunctions.translate(text)

		# Sentence tokenize with nltk to handle lengthy inputs.
		text_list = sent_tokenize(text)
		output = []

		# Process each sentence in the text
		for item in text_list:
			# Run the model on each sentence and extend the output list with the results
			output.extend(self._run_model(item, linking)) if self._run_model(item, linking) else None

		return output


	def _run_model(self, sentence: str, link: bool) -> List[dict]:
		"""
		Perform entity extraction and optionally link entities to the ESCO taxonomie.

		Parameters
		----------
		sentence : str
			A sentence from which to extract and possibly link entities.
			
		link : bool
			Specifies whether the model should perform entity linking to the knowledge base. 
			If `False`, it only extracts entities without linking them.

		Returns
		-------
		List[dict]
			A list of dictionaries with the extracted entities. Each dictionary contains the following keys:
			- `type`: The category of the identified entity (e.g., 'Occupation', 'Skill', 'Qualification').
			- `tokens`: The specific part of the input sentence that was identified as an entity of the right category.
			- `retrieved`: If `link` is `True`, a list of related names or ESCO codes retrieved from the reference sets. 
				These items represent the most similar entities or concepts based on the embeddings and similarity calculations.
			- `scores`: (Optional) If `evaluation_mode` is `True`, the cosine similarity scores for the retrieved items.
		"""

		# Extract entities from the text
		formatted_entities = self._ner_pipeline(sentence)

		# Check whether or not linking should be performed
		if link:
			for entry in formatted_entities:
				if entry['type'] in {"Occupation", "Skill", "Qualification"}:
					# Encode the extracted entity tokens into embeddings
					emb = self.similarity_model.encode(entry['tokens'])
					emb = torch.from_numpy(emb).to(self.device)
					# Retrieve the top-k suggestions based on the extracted entity
					if self.evaluation_mode:
						entry['retrieved'], entry['scores'] = self._top_k(emb, entry['type'])
					else:
						entry['retrieved'] = self._top_k(emb, entry['type'])

		return formatted_entities


	def _ner_pipeline(self, text: str) -> List[dict]:
		"""
		Entity extraction pipeline. Runs the text through the BERT-based encoders, performs post-processing for tagging cleanup,
		and returns a list of dictionaries with all relevant information.

		Parameters
		----------
		text : str
			The input text to process for entity extraction.

		Returns
		-------
		List[dict]
			A list of dictionaries with the extracted entities. Each dictionary contains the following keys:
			- `tokens`: The specific part of the input text identified as an entity.
			- `type`: The category of the identified entity (e.g., 'Occupation', 'Skill', 'Qualification').
		"""

		# Tokenize inputs
		inputs = self.tokenizer(text, return_tensors='pt', truncation=True).to(self.device)

		# Check whether a CRF entity extraction model is used and produce the logits and prediction entity numerical categories
		if self.crf:
			with torch.no_grad():
				logits = self.entity_model(**inputs)
			predictions = logits[1][0]
		else:
			with torch.no_grad():
				logits = self.entity_model(**inputs).logits
			predictions = torch.argmax(logits, dim=2)

		# Produce the BIO tags
		predicted_token_class = [self.entity_model.config.id2label[t.item()] for t in predictions[0]]

		# Post-processing: Hand-crafted rules that fix common tagging errors and undesirable outputs
		predicted_token_class = self.fix_bio_tags(predicted_token_class)

		# Filters out special tags from transformer outputs
		input_ids, predicted_token_class = self.remove_special_tokens_and_tags(inputs['input_ids'][0], predicted_token_class, self.tokenizer)

		# Format the output
		result = self.extract_entities(input_ids, predicted_token_class)

		# Decode the extracted entities into word n-grams
		for entry in result:
			sentence = self.tokenizer.decode(entry['tokens'])
			# Fix common decoding error in DeBERTa and RoBERTa that produces a blank space at the start of some tokens
			if sentence.startswith(' '):
				sentence = sentence[1:]
			entry['tokens'] = sentence

		return result


	def _top_k(self, embedding: torch.Tensor, entity_type: str) -> list:
		"""
		Entity similarity pipeline. Retrieves the top-k most similar entities using cosine similarity from the corresponding reference vector database.

		Parameters
		----------
		embedding : torch.Tensor
			The embedding of the entity to find similar entities for.

		entity_type : str
			The type of entity to compare (e.g., 'Occupation', 'Qualification', 'Skill').

		Returns
		-------
		list
			A list of the top-k most similar entities from the reference set. If `evaluation_mode` is `True`, also returns the cosine similarity scores.
		"""

		if entity_type == "Occupation":
			local_df = self.df_occ['occupation'] if self.output_format == 'name' else self.df_occ['esco_code']
			local_emb = self.occupation_emb
		elif entity_type == "Qualification":
			local_df = self.df_qual['qualification']
			local_emb = self.qualification_emb
		else:
			local_df = self.df_skill['skills']
			local_emb = self.skill_emb

		# Calculate cosine similarity between the input embedding and the reference embeddings
		cos_scores = util.cos_sim(embedding, local_emb)[0]

		# Find the top-k highest cosine similarity scores
		top_k_scores = torch.topk(cos_scores, k=self.k)
		top_k_list = top_k_scores.indices.tolist()

		# Retrieve the top-k most similar entities from the reference DataFrame
		top_k = list(local_df.iloc[top_k_list])

		# If evaluation_mode is enabled, return the top-k entities along with their similarity scores
		if self.evaluation_mode:
			return top_k, top_k_scores.values.tolist()

		# For better formatted outputs in occupations, remove duplicate suggestion codes
		return self.remove_duplicates_ordered(top_k)


	def _load_tensors(self) -> Tuple[List[torch.Tensor]]:
		"""
		Load the embeddings for occupations, skills, and qualifications. If the embeddings are not cached 
		(`from_cache=False`), this function creates a folder inside the files directory with the name of 
		the Sentence Transformer to store the embeddings.

		Returns
		-------
		Tuple[List[torch.Tensor]]
			A tuple containing three lists of tensors: the embeddings for occupations, skills, and qualifications.
		"""

		# Determine the path for storing or loading the embeddings
		path = 'inference/files/' + str(self.similarity_model_type).split("/")[-1]

		if self.from_cache:
			# Load cached embeddings from precomputed files
			occupation_emb = self.create_tensors(path + '/occupations.pkl', self.device)
			skill_emb = self.create_tensors(path + '/skills.pkl', self.device)
			qualification_emb = self.create_tensors(path + '/qualifications.pkl', self.device)
		else:
			# Create a new directory to store embeddings
			os.mkdir(path)
			# Compute and store embeddings
			occupation_emb = self._corpus_embedding(corpus=[occ for occ in self.df_occ['occupation']], entity_type='Occupations', path=path)
			skill_emb = self._corpus_embedding(corpus=[skill for skill in self.df_skill['skills']], entity_type='Skills', path=path)
			qualification_emb = self._corpus_embedding(corpus=[qual for qual in self.df_qual['qualification']], entity_type='Qualifications', path=path)

		return occupation_emb, skill_emb, qualification_emb


	def _corpus_embedding(self, corpus: List[str], entity_type: str, path: str) -> List[torch.Tensor]:
		"""
		Compute and store the embeddings for a given corpus if `from_cache` is `False`.

		Parameters
		----------
		corpus : List[str]
			The list of strings to compute embeddings for.

		entity_type : str
			The type of entities in the corpus (e.g., 'Occupations', 'Skills', 'Qualifications').

		path : str
			The directory path where the embeddings will be stored.

		Returns
		-------
		List[torch.Tensor]
			A list of tensors representing the embeddings for the corpus.
		"""

		# Compute the embeddings for the corpus
		corpus_embeddings = self.similarity_model.encode(corpus, convert_to_tensor=True)
		
		# Define the path for storing the embeddings
		embeddings_path = path + '/' + entity_type.lower() + '.pkl'
		
		# Store the computed embeddings in a pickle file
		with open(embeddings_path, 'wb') as f:
			pickle.dump(corpus_embeddings, f)
		
		# Return the embeddings loaded as tensors
		return self.create_tensors(embeddings_path, self.device)


	@staticmethod
	def extract_entities(tokens : list, tags : list) -> List[dict]:
		"""
		Function that formats the tokens and tags to a JSON-like output.
		"""
		result = []
		#Loop through the dictionary of tags, while tracking the current entity 
		current_entity = None
		for token, tag in zip(tokens, tags):
				#Get label tag and tag type if tag is not O.
				tag_type, tag_label = tag.split('-') if '-' in tag else ('O', tag)
				if tag_type != 'O':
						#Check if tracking an entity and the type matches the tag label. TODO: Handle the cases where I- tags follows B- tags of the same type. 
						if current_entity and current_entity['type'] == tag_label:
								current_entity['tokens'].append(token)
						else:
								if current_entity:
										result.append(current_entity)
								current_entity = {'type': tag_label, 'tokens': [token]}
				else:
						if current_entity:
								result.append(current_entity)
								current_entity = None
		if current_entity:
				result.append(current_entity)
		#Post Processing. Remove empty entries in results
		condition_function = lambda x: len(x['tokens']) != 0
		filtered_list = [item for item in result if condition_function(item)]

		return filtered_list
	
	@staticmethod
	def create_tensors(file : str, device : str) -> List[torch.Tensor]:
		"""
		Function that checks type of device to load the torch tensors
		"""
		with open(file, 'rb') as f:
			if device.type=='cpu':
				embeddings = CPU_Unpickler(f).load()
			else:
				embeddings = pickle.load(f)
		# Ensure embeddings is a tensor
		if isinstance(embeddings, list):
			arrayEmbeddings = np.array(embeddings)
			embeddings = torch.tensor(arrayEmbeddings)
		
		# Move tensor to the specified device
		embeddings = embeddings.to(device)
		return embeddings


	@staticmethod
	def remove_duplicates_ordered(input_list : list) -> list:
		"""
		Function thet removes duplicates from list retaining the order
		"""
		unique_list = []
		for item in input_list:
				if item not in unique_list:
						unique_list.append(item)
		return unique_list
	

	@staticmethod
	def fix_bio_tags(tags:list)-> list:
		"""
		Function that is used for post processing and impelmentig hand crafted rules. First, it checks if there is a tagging sequence of B, O, I, and replaces O with I.
		Then, checks if a sequence ends with O, I and replaces I with O.
		"""
		fixed_tags = list(tags)
		for i in range(len(tags) - 2):
				if tags[i].startswith('B-') and tags[i + 1] == 'O' and tags[i + 2].startswith('I-'):
						fixed_tags[i + 1] = tags[i + 2]
				if tags[i] == 'O' and tags[i + 1].startswith('I-') and tags[i + 2] == 'O':
						fixed_tags[i + 1] = 'O'
		if tags[-2] == 'O' and tags[-1].startswith('I-'):
						fixed_tags[i + 1] = 'O'
		return fixed_tags
	
	@staticmethod
	def remove_special_tokens_and_tags(input_ids:List[int], bio_tags:List[str], tokenizer) -> Tuple[List[int], List[str]]:
		"""
		Function that filters out special tags from transformer outputs. 
		"""
		special_tokens_ids = tokenizer.all_special_ids

		# Filter out special token IDs and corresponding tags
		filtered_ids = []
		filtered_tags = []
		for id_, tag in zip(input_ids, bio_tags):
				if id_ not in special_tokens_ids:
						filtered_ids.append(id_)
						filtered_tags.append(tag)

		return filtered_ids, filtered_tags