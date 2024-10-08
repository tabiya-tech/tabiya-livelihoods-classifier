from typing import List, Optional, Tuple
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
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
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

class Entity:
    def __init__(self, **entries):
        self.__dict__.update(entries)

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

	evaluation_mode : bool, default=False
		If set to `True`, the linker will return the cosine similarity scores between the embeddings. 
		This mode is useful for evaluating the quality of the linkages.

	k : int, default=32
		Specifies the number of items to retrieve from the reference sets. 
		This parameter limits the number of top matches to consider when linking entities.

	from_cache : bool, default=True
		If set to `True`, the precomputed embeddings are loaded from cache to save time. 
		If set to `False`, the embeddings are computed on-the-fly, which requires GPU access for efficiency and can be time-consuming.

	output_format : str, default='occupation'
		Specifies the format of the output for occupations, either `occupation`, `preffered_label`, `esco_code`, `uuid` or `all` to get all the columns. 
		The `uuid` is also available for the skills.

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
			evaluation_mode: bool = False,
			k: int = 32,
			from_cache: bool = True,
			output_format: str = 'occupation'
	):
		# Initialize the EntityRecognition model
		self.entity_recognition_model = EntityRecognition(
			entity_model=entity_model,
			crf=crf
		)

		# Initialize the SimilarityJobSearch model
		self.similarity_search_model = SimilarityJobSearch(
			similarity_model=similarity_model,
			evaluation_mode=evaluation_mode,
			k=k,
			from_cache=from_cache,
			output_format=output_format
		)


	def settings(
			self,
			evaluation_mode: bool = None,
			k: int = None,
			output_format: str = None
	):
		"""
		Change the settings of the EntityLinker object.

		Parameters
		----------
		evaluation_mode : bool, optional
			If set to `True`, the linker will return the cosine similarity scores between the embeddings.
			This mode is useful for evaluating the quality of the linkages.

		k : int, optional
			Specifies the number of items to retrieve from the reference sets.
			This parameter limits the number of top matches to consider when linking entities.

		output_format : str, optional
			Specifies the field in the output for occupations, either `occupation`, `preffered_label`, `esco_code`, `uuid` or `all` to get all the fields.
			The `uuid` is also available for the skills.
		"""
		self.similarity_search_model.settings(evaluation_mode, k, output_format)


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
				If `output_format` is set to `all`, the retrieved items will be in the form of a list of Entity objects 
				with as attributes the columns names of the database.
			- `scores`: A list of cosine similarity scores between the extracted entity and the retrieved items.
				Appear if linking=True and evaluation_mode=True.
		"""

		formatted_entities = self.entity_recognition_model(text)
		if linking:
			linked_entities = self.similarity_search_model(formatted_entities)
			return linked_entities
		return formatted_entities
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


class EntityRecognition:
	"""
	Performs entity recognition on job-related text.
	
	Initialization Parameters
	----------
	entity_model : str, default='tabiya/roberta-base-job-ner'
		Path to a pre-trained `AutoModelForTokenClassification` model or an `AutoModelCrfForNer` model.
		This model is used for entity recognition within the input text.
		
	crf : bool, default=False
		A flag to indicate whether to use an `AutoModelCrfForNer` model instead of a standard `AutoModelForTokenClassification`.
		`CRF` (Conditional Random Field) models are used when the task requires sequential predictions with dependencies between the outputs.

	Calling Parameters
	----------
	text : str
		An arbitrary job vacancy-related string that the model processes to extract entities.

	Returns
	-------
	List[dict]
		A list of dictionaries with the extracted entities. 
		Each dictionary contains the following keys:
		- `type`: The category of the identified entity (e.g., 'Occupation', 'Qualifications', 'Skill', 'Experience').
		- `tokens`: The specific part of the input text that was identified as an entity of the right category		
	"""

	def __init__(
			self,
			entity_model: str = 'tabiya/roberta-base-job-ner',
			crf: Optional[bool] = False
	):
		# Initialize the model paths and settings
		self.entity_model = entity_model
		self.crf = crf

		# Set the device to GPU if available, otherwise CPU
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# Load the appropriate entity recognition model based on the crf flag
		if self.crf:
			self.entity_model = AutoModelCrfForNer.from_pretrained(entity_model)
		else:
			self.entity_model = AutoModelForTokenClassification.from_pretrained(entity_model, token=os.getenv('HF_TOKEN'))

		# Move the entity model to the appropriate device
		self.entity_model.to(self.device)

		# Initialize the tokenizer for the entity model
		self.tokenizer = AutoTokenizer.from_pretrained(entity_model, token=os.getenv('HF_TOKEN'))


	def __call__(self, text: str) -> List[dict]:
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
			output.extend(self._ner_pipeline(item)) if self._ner_pipeline(item) else None

		return output


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


class SimilarityJobSearch:
	"""
	Performs similarity search on a precomputed set of vectors using a sentence transformer model on job related entity.

	Initialization Parameters
	----------
	similarity_model : str, default='all-MiniLM-L6-v2'
		Path or name of a sentence transformer model used for embedding text. 
		The sentence transformer is used to compute embeddings for the extracted entities and the reference sets.
		The model 'all-mpnet-base-v2' is available but not in cache, so it should be used with the parameter `from_cache=False` at least the first time.

	evaluation_mode : bool, default=False
		If set to `True`, the linker will return the cosine similarity scores between the embeddings.
		This mode is useful for evaluating the quality of the linkages.

	k : int, default=32
		Specifies the number of items to retrieve from the reference sets.
		This parameter limits the number of top matches to consider when linking entities.

	from_cache : bool, default=True
		If set to `True`, the precomputed embeddings are loaded from cache to save time. 
		If set to `False`, the embeddings are computed on-the-fly, which requires GPU access for efficiency and can be time-consuming.

	output_format : str, default='occupation'
		Specifies the field in the output for occupations, either `occupation`, `preffered_label`, `esco_code`, `uuid` or `all` to get all the fields.
		The `uuid` is also available for the skills.

	Calling Parameters
	----------
	dict
		A list of dictionaries with an extracted entities and its type. 
		Each dictionary contains the following keys:
		- `type`: The category of the identified entity (e.g., 'Occupation', 'Qualifications', 'Skill', 'Experience').
		- `tokens`: An input text that was identified as an entity of the right category.

	Returns
	-------
	dict
		A list of dictionaries with the extracted entities and the top-k most similar entities from the reference sets. 
		Each dictionary contains the following keys:
		- `type`: The category of the identified entity (e.g., 'Occupation', 'Qualifications', 'Skill', 'Experience').
		- `tokens`: An input text that was identified as an entity of the right category.
		- `retrieved`: A list of related names or ESCO codes retrieved from the reference sets. 
			These items represent the most similar entities or concepts based on the embeddings and similarity calculations.
			If `output_format` is set to `all`, the retrieved items will be in the form of a list of Entity objects 
			with as attributes the columns names of the database.
		- `scores`: A list of cosine similarity scores between the extracted entity and the retrieved items.
			The scores are only returned if `evaluation_mode` is `True`.
	"""

	def __init__(
			self,
			similarity_model: str = 'all-MiniLM-L6-v2',
			evaluation_mode: bool = False,
			k: int = 32,
			from_cache: bool = True,
			output_format: str = 'occupation'
	):
		# Initialize the model path and settings
		self.similarity_model_type = similarity_model
		self.similarity_model = SentenceTransformer(similarity_model)
		self.from_cache = from_cache
		self.path_to_files = os.path.abspath(os.path.join(os.path.dirname(__file__), 'files'))

		# Initialize the model paths and settings
		self.similarity_model_type = similarity_model
		self.similarity_model = SentenceTransformer(similarity_model)
		self.evaluation_mode = evaluation_mode
		self.k = k
		self.from_cache = from_cache
		self.output_format = output_format
		self.path_to_files = os.path.abspath(os.path.join(os.path.dirname(__file__), 'files'))

		# Set the device to GPU if available, otherwise CPU
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# Load reference sets for occupations, skills, and qualifications
		self.df_occ = pd.read_csv(os.path.join(self.path_to_files, 'occupations_augmented.csv'))
		self.df_skill = pd.read_csv(os.path.join(self.path_to_files, 'skills.csv'))
		self.df_qual = pd.read_csv(os.path.join(self.path_to_files, 'qualifications.csv'))

		# Fix the number of rows to check to get the top-k most similar entities
		self.relative_k = self.df_occ['esco_code'].value_counts().head(k-1).sum() + 1  

		# Load precomputed embeddings for the reference sets
		self.occupation_emb, self.skill_emb, self.qualification_emb = self._load_tensors()


	def settings(
			self,
			evaluation_mode: bool = None,
			k: int = None,
			output_format: str = None
	):
		"""
		Change the settings of the SimilarityVectorSearch object.

		Parameters
		----------
		evaluation_mode : bool, optional
			If set to `True`, the linker will return the cosine similarity scores between the embeddings.
			This mode is useful for evaluating the quality of the linkages.

		k : int, optional
			Specifies the number of items to retrieve from the reference sets.
			This parameter limits the number of top matches to consider when linking entities.

		output_format : str, optional
			Specifies the field in the output for occupations, either `occupation`, `preffered_label`, `esco_code`, `uuid` or `all` to get all the fields.
			The `uuid` is also available for the skills.
		"""
		if evaluation_mode is not None:
			self.evaluation_mode = evaluation_mode
		if k is not None:
			self.k = k
			self.relative_k = self.df_occ['esco_code'].value_counts().head(k-1).sum() + 1
		if output_format is not None:
			self.output_format = output_format


	def __call__(self, entity: List[dict]) -> List[dict]:
		output = []
		for item in entity:
			output.append(self.search(item['tokens'], item['type']))
		return output


	def search(self, text: str, entity_type: str) -> dict:
		"""
		Search for the most similar entities in the reference set based on the extracted entity.

		Parameters
		----------
		text : str
			The extracted entity to search for similar entities in the reference set.

		entity_type : str
			The type of entity to compare (e.g., 'Occupation', 'Qualification', 'Skill').

		Returns
		-------
		dict
			A dictionary with the extracted entity and the top-k most similar entities from the reference set.
		"""
		# Encode the extracted entity tokens into embeddings
		emb = self.similarity_model.encode(text)
		emb = torch.from_numpy(emb).to(self.device)
		# Retrieve the top-k suggestions based on the extracted entity
		match entity_type, self.output_format:
			case "Occupation", "all":
				local_df = self.df_occ
				local_emb = self.occupation_emb
			case "Qualification", "all":
				local_df = self.df_qual
				local_emb = self.qualification_emb
			case "Skill", "all":
				local_df = self.df_skill
				local_emb = self.skill_emb
			case "Occupation", _:
				local_df = self.df_occ[self.output_format]
				local_emb = self.occupation_emb
			case "Qualification", _:
				local_df = self.df_qual['qualification']
				local_emb = self.qualification_emb
			case "Skill", _:
				local_df = self.df_skill['skills'] if self.output_format != 'uuid' else self.df_skill['uuid']
				local_emb = self.skill_emb
			case _, _:
				return {"type": entity_type, "tokens": text}
			
		# Calculate cosine similarity between the input embedding and the reference embeddings
		cos_scores = util.cos_sim(emb, local_emb)[0]

		# Find the top-k highest cosine similarity scores
		# In certain cases, the relative_k is used as there are duplicates in the reference set
		if self.output_format == 'occupation' or entity_type != "Occupation" or self.evaluation_mode:
			top_k_scores = torch.topk(cos_scores, k=self.k)
			top_k_list = top_k_scores.indices.tolist()
		else:
			top_k_scores = torch.topk(cos_scores, k=self.relative_k)
			top_k_list = top_k_scores.indices.tolist()

		if self.output_format == 'all':
			top_k_df = local_df.iloc[top_k_list]
			# Convert each row of the DataFrame to an Entity object
			top_k = [Entity(**row) for _, row in top_k_df.iterrows()]
		else:
			# Retrieve the top-k most similar entities from the reference DataFrame
			top_k = list(local_df.iloc[top_k_list])

		# If evaluation_mode is enabled, return the top-k entities along with their similarity scores
		if self.evaluation_mode:
			return {"type": entity_type, "tokens": text, "retrieved": top_k, "scores": top_k_scores.values.tolist()}
		
		if self.output_format == 'all':
			if entity_type == "Occupation":
				# For better formatted outputs in occupations, remove duplicate suggestion codes
				print(len(top_k))	
				return {"type": entity_type, "tokens": text, "retrieved": self.remove_duplicates_ordered_entities(top_k, self.k)}
			else:
				return {"type": entity_type, "tokens": text, "retrieved": top_k}
		
		return {"type": entity_type, "tokens": text, "retrieved": self.remove_duplicates_ordered(top_k, self.k)}


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
		path = os.path.join(self.path_to_files, self.similarity_model_type)

		if self.from_cache:
			# Load cached embeddings from precomputed files
			occupation_emb = self.create_tensors(os.path.join(path, 'occupations.pkl'), self.device)
			skill_emb = self.create_tensors(os.path.join(path, 'skills.pkl'), self.device)
			qualification_emb = self.create_tensors(os.path.join(path, 'qualifications.pkl'), self.device)
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
		embeddings_path = os.path.join(path, f'{entity_type.lower()}.pkl')
		
		# Store the computed embeddings in a pickle file
		with open(embeddings_path, 'wb') as f:
			pickle.dump(corpus_embeddings, f)
		
		# Return the embeddings loaded as tensors
		return self.create_tensors(embeddings_path, self.device)


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
	def remove_duplicates_ordered(input_list : list, max_length=-1) -> list:
		"""
		Remove duplicates from a list retaining the order.

		Parameters
		----------
		input_list : list
			List of elements where duplicates are removed.
		max_length : int, optional
			Maximum length of the output list. If -1, no limit is applied.

		Returns
		-------
		list
			List of elements with duplicates removed, retaining the order.
		"""
		seen = set()
		seen_add = seen.add
		if max_length == -1:
			return [x for x in input_list if not (x in seen or seen_add(x))]
		else:
			return [x for x in input_list if not (x in seen or seen_add(x))][:max_length]


	@staticmethod
	def remove_duplicates_ordered_entities(input_list: List[Entity], max_length=-1) -> List[Entity]:
		"""
		Remove duplicates from a list of entities based on the 'esco_code' attribute, retaining the order.
		
		Parameters
		----------
		input_list : List[Entity]
			List of entities where each entity has an 'esco_code' attribute.
		max_length : int, optional
			Maximum length of the output list. If -1, no limit is applied.
		
		Returns
		-------
		list
			List of entities with duplicates removed, retaining the order.
		"""
		seen = set()
		seen_add = seen.add
		result = []
		for entity in input_list:
			if hasattr(entity, 'esco_code'):
				esco_code = entity.esco_code
				if esco_code not in seen:
					seen_add(esco_code)
					result.append(entity)
					if 0 <= max_length == len(result):
						break
		return result