from typing import List, Optional
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
#from google.cloud import translate_v2 as translate
#import ftlangdetect
from util.utilfunctions import UtilFunctions
from util.transformersCRF import BertCrfForNer

class EntityLinker:
  """
  Creates a pipeline of an entity recognition transformer and a sentence transformer for embedding text
  Initialization arguments:
    entity_model: Path to a pre-trained entity recognition model or a BertForCrf model
    similarity_model: Path or string of a sentence transformer
    tokenizer_type: default 'wordpiece'. Choose 'bpe' when using an appropriate model
    crf: default False. Choose True when using a BertForCrf model

  Calling arguments:
    text: An arbitrary job vacancy-related string. It is preprocessed using the sentence tokenizer of nltk
    linking: default True. Specify whether the model peforms the entity linking to the taxonomy

  """
  def __init__(
      self,
      entity_model : str ,
      similarity_model : str ,
      tokenizer_type : Optional[str] = 'wordpiece',
      crf : Optional[bool] = False,
      hf_token : str = None
      ):
    self.entity_model = entity_model
    self.similarity_model = SentenceTransformer(similarity_model)
    self.tokenizer_type = tokenizer_type
    self.crf = crf
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if self.crf:
      self.entity_model = BertCrfForNer.from_pretrained(entity_model)
    else:
      self.entity_model = AutoModelForTokenClassification.from_pretrained(entity_model, token=hf_token)
    self.entity_model.to(self.device)
    self.tokenizer = AutoTokenizer.from_pretrained(entity_model, token=hf_token)
    #self.classifier = pipeline('ner', model=entity_model, tokenizer=AutoTokenizer.from_pretrained(entity_model))


    self.occupation_emb = UtilFunctions.create_tensors('inference/files/augmented_occupation_embeddings.pkl', self.device)
    self.skill_emb = UtilFunctions.create_tensors('inference/files/skill_embeddings.pkl', self.device)
    self.qualification_emb = UtilFunctions.create_tensors('inference/files/qualificaton_embeddings.pkl', self.device)

    self.df_occ = pd.read_csv('inference/files/occupations_augmented.csv')
    self.df_skill = pd.read_csv('inference/files/skills.csv')
    self.df_qual = pd.read_csv('inference/files/qualifications.csv')

    label_list = [ 'O',
               'B-Skill',
               'B-Qualification',
               'I-Domain',
               'I-Experience',
               'I-Qualification',
               'B-Occupation',
               'B-Domain',
               'I-Occupation',
               'I-Skill',
               'B-Experience',
               'X',
               ]

    self.id2label = {idx:tag for idx, tag in enumerate(label_list)}

  def __call__(self, text : str, linking : bool  = True) -> List[dict]:
      text = text.replace('\n', '')
      #language = UtilFunctions.detect_language(text)
      #if language != 'en':
      #    text = UtilFunctions.translate(text)
      text_list = sent_tokenize(text)
      output = []
      for item in tqdm(text_list):
          output.extend(self._run_model(item, linking)) if self._run_model(item, linking) else None
      return output

  def _run_model(self, sentence : str, link : bool) -> List[dict]:
      #extracted_entities = self.classifier(sentence)
      formatted_entities = self._ner_pipeline(sentence)
      if link:
        for entry in formatted_entities:
            if entry['type'] == "Occupation" or entry['type'] == "Skill" or entry['type'] == "Qualification":
                emb = self.similarity_model.encode(entry['tokens'])
                emb = torch.from_numpy(emb).to(self.device)
                entry['retrieved'] = self._top_5(emb, entry['type'])

      return formatted_entities



  def _ner_pipeline(self, text : str) -> List[dict]:
      inputs = self.tokenizer(text, return_tensors='pt', return_special_tokens_mask=self.crf).to(self.device)
      if self.crf:
          tensor = inputs['special_tokens_mask'][0]
          tensor[[0,-1]] = 0
          inputs['special_tokens_mask'][0] = tensor
          with torch.no_grad():
              logits = self.entity_model(**inputs)
          predictions = logits[1][0]
          predicted_token_class = [self.id2label[t.item()] for t in predictions[0]]
      else:
          with torch.no_grad():
              logits = self.entity_model(**inputs).logits
          predictions = torch.argmax(logits, dim=2)
          predicted_token_class = [self.id2label[t.item()] for t in predictions[0]]

      result = UtilFunctions.extract_entities(self.tokenizer.batch_decode(inputs['input_ids'][0]), predicted_token_class)

      for entry in result:
          sentence = UtilFunctions.decode(entry['tokens'], self.tokenizer_type)
          entry['tokens'] = sentence
      return result


  def _top_5(self, embedding : torch.Tensor, entity_type : str) -> list:
      if entity_type == "Occupation":
          local_df = self.df_occ['esco_code']
          local_emb = self.occupation_emb
      elif entity_type == "Qualification":
          local_df = self.df_qual['EQF_level']
          local_emb = self.qualification_emb
      else:
          local_df = self.df_skill['skills']
          local_emb = self.skill_emb

      # We use cosine-similarity and torch.topk to find the highest 5 scores
      cos_scores = util.cos_sim(embedding, local_emb)[0]
      top_k = torch.topk(cos_scores, k=5)
      top_5_list = top_k.indices.tolist()
      top_5 = list(local_df.iloc[top_5_list])
      return UtilFunctions.remove_duplicates_ordered(top_5)