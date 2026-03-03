from typing import List, Optional, Tuple
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from nltk.tokenize import sent_tokenize
from util.transformersCRF import AutoModelCrfForNer
from util.ner_utils import fix_bio_tags, extract_entities, remove_special_tokens
from util.embedding_utils import load_tensor, compute_and_cache
import os
from dotenv import load_dotenv

load_dotenv()

class Entity:
    def __init__(self, **entries):
        self.__dict__.update(entries)

class EntityLinker:
	"""NER + entity linking pipeline that extracts entities from job text and maps them to ESCO taxonomy."""

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
		self.entity_model = entity_model
		self.similarity_model_type = similarity_model
		self.similarity_model = SentenceTransformer(similarity_model)
		self.crf = crf
		self.evaluation_mode = evaluation_mode
		self.k = k
		self.from_cache = from_cache
		self.output_format = output_format
		self.path_to_files = os.path.abspath(os.path.join(os.path.dirname(__file__), 'files'))

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		if self.crf:
			self.entity_model = AutoModelCrfForNer.from_pretrained(entity_model)
		else:
			self.entity_model = AutoModelForTokenClassification.from_pretrained(entity_model, token=os.getenv('HF_TOKEN'))

		self.entity_model.to(self.device)
		self.tokenizer = AutoTokenizer.from_pretrained(entity_model, token=os.getenv('HF_TOKEN'))

		self.df_occ = pd.read_csv(os.path.join(self.path_to_files, 'occupations_augmented.csv'))
		self.df_skill = pd.read_csv(os.path.join(self.path_to_files, 'skills.csv'))
		self.df_qual = pd.read_csv(os.path.join(self.path_to_files, 'qualifications.csv'))

		self.occupation_emb, self.skill_emb, self.qualification_emb = self._load_tensors()


	def __call__(self, text: str, linking: bool = True) -> List[dict]:
		"""Extract entities from text and optionally link them to ESCO taxonomy."""
		text = text.replace('\n', ' ')

		# TODO: Implement the Google Translate features to enable multilingual entity linking.
		# language = UtilFunctions.detect_language(text)
		# if language != 'en':
		#     text = UtilFunctions.translate(text)

		text_list = sent_tokenize(text)
		output = []

		for item in text_list:
			entities = self._run_model(item, linking)
			if entities:
				output.extend(entities)

		return output


	def _run_model(self, sentence: str, link: bool) -> List[dict]:
		"""Extract entities from a sentence and optionally link them to ESCO."""
		formatted_entities = self._ner_pipeline(sentence)

		if link:
			for entry in formatted_entities:
				if entry['type'] in {"Occupation", "Skill", "Qualification"}:
					emb = self.similarity_model.encode(entry['tokens'])
					emb = torch.from_numpy(emb).to(self.device)
					if self.evaluation_mode:
						entry['retrieved'], entry['scores'] = self._top_k(emb, entry['type'])
					else:
						entry['retrieved'] = self._top_k(emb, entry['type'])

		return formatted_entities


	def _ner_pipeline(self, text: str) -> List[dict]:
		"""Run BIO tagging, post-processing, and entity span extraction on a single sentence."""
		inputs = self.tokenizer(text, return_tensors='pt', truncation=True).to(self.device)

		if self.crf:
			with torch.no_grad():
				logits = self.entity_model(**inputs)
			predictions = logits[1][0]
		else:
			with torch.no_grad():
				logits = self.entity_model(**inputs).logits
			predictions = torch.argmax(logits, dim=2)

		predicted_token_class = [self.entity_model.config.id2label[t.item()] for t in predictions[0]]

		predicted_token_class = fix_bio_tags(predicted_token_class)

		input_ids, predicted_token_class = remove_special_tokens(
			inputs['input_ids'][0], predicted_token_class, set(self.tokenizer.all_special_ids)
		)

		result = extract_entities(input_ids, predicted_token_class)

		for entry in result:
			sentence = self.tokenizer.decode(entry['tokens'])
			# Fix common decoding error in DeBERTa and RoBERTa that produces a blank space at the start of some tokens
			if sentence.startswith(' '):
				sentence = sentence[1:]
			entry['tokens'] = sentence

		return result


	def _top_k(self, embedding: torch.Tensor, entity_type: str) -> list:
		"""Retrieve top-k ESCO matches by cosine similarity."""
		if self.output_format == 'all':
			if entity_type == "Occupation":
				local_df = self.df_occ
				local_emb = self.occupation_emb
			elif entity_type == "Qualification":
				local_df = self.df_qual
				local_emb = self.qualification_emb
			else:
				local_df = self.df_skill
				local_emb = self.skill_emb
		else:
			if entity_type == "Occupation":
				local_df = self.df_occ[self.output_format]
				local_emb = self.occupation_emb
			elif entity_type == "Qualification":
				local_df = self.df_qual['qualification']
				local_emb = self.qualification_emb
			else:
				local_df = self.df_skill['skills'] if self.output_format != 'uuid' else self.df_skill['uuid']
				local_emb = self.skill_emb

		cos_scores = util.cos_sim(embedding, local_emb)[0]
		top_k_scores = torch.topk(cos_scores, k=self.k)
		top_k_list = top_k_scores.indices.tolist()

		if self.output_format == 'all':
			top_k_df = local_df.iloc[top_k_list]
			top_k_entities = [Entity(**row) for _, row in top_k_df.iterrows()]
			
			if self.evaluation_mode:
				return top_k_entities, top_k_scores.values.tolist()
			return top_k_entities
		else:
			top_k = list(local_df.iloc[top_k_list])

			if self.evaluation_mode:
				return top_k, top_k_scores.values.tolist()

			# Dedup occupation suggestion codes for cleaner output
			return self.remove_duplicates_ordered(top_k)


	def _load_tensors(self) -> Tuple[List[torch.Tensor]]:
		"""Load or compute ESCO reference embeddings."""
		path = os.path.join(self.path_to_files, self.similarity_model_type)

		if self.from_cache:
			occupation_emb = load_tensor(os.path.join(path, 'occupations.pkl'), self.device)
			skill_emb = load_tensor(os.path.join(path, 'skills.pkl'), self.device)
			qualification_emb = load_tensor(os.path.join(path, 'qualifications.pkl'), self.device)
		else:
			os.mkdir(path)
			occupation_emb = compute_and_cache(
				self.similarity_model, list(self.df_occ['occupation']), 'occupations', path, self.device
			)
			skill_emb = compute_and_cache(
				self.similarity_model, list(self.df_skill['skills']), 'skills', path, self.device
			)
			qualification_emb = compute_and_cache(
				self.similarity_model, list(self.df_qual['qualification']), 'qualifications', path, self.device
			)

		return occupation_emb, skill_emb, qualification_emb


	@staticmethod
	def remove_duplicates_ordered(input_list: list) -> list:
		"""Remove duplicates from a list while retaining order."""
		unique_list = []
		for item in input_list:
			if item not in unique_list:
				unique_list.append(item)
		return unique_list
