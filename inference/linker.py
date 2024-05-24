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
  Creates a pipeline of an entity recognition transformer and a sentence transformer for embedding text
  Initialization arguments:
    entity_model: Path to a pre-trained AutoModelForTokenClassification model or an AutoModelCrfForNer model
    similarity_model: Path or string of a sentence transformer
    crf: default False. Choose True when using a AutoModelCrfForNer model
    hf_token: default None. HuggingFace token for using private models
    evaluation_mode: default False. If True the linker will return the cosine scores
    k: default 32. The number of retrieved items from the reference sets
    from_cache: default True. Choose False when there are no precomputed embeddings

  Calling arguments:
    text: An arbitrary job vacancy-related string
    linking: default True. Specify whether the model peforms the entity linking to the taxonomy

  """
  def __init__(
      self,
      entity_model : str = 'tabiya/roberta-base-job-ner',
      similarity_model : str = 'all-MiniLM-L6-v2',
      crf : Optional[bool] = False,
      hf_token : str = None,
      evaluation_mode: bool = False,
      k:int = 32,
      from_cache:bool=True
      ):
    self.entity_model = entity_model
    self.similarity_model_type = similarity_model
    self.similarity_model = SentenceTransformer(similarity_model)
    self.crf = crf
    self.evaluation_mode = evaluation_mode
    self.k = k
    self.from_cache = from_cache
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if self.crf:
      self.entity_model = AutoModelCrfForNer.from_pretrained(entity_model)
    else:
      self.entity_model = AutoModelForTokenClassification.from_pretrained(entity_model, token=hf_token)
    self.entity_model.to(self.device)
    self.tokenizer = AutoTokenizer.from_pretrained(entity_model, token=hf_token)

    #Load Reference Sets - ESCO and EQF lookups
    self.df_occ = pd.read_csv('inference/files/occupations_augmented.csv')
    self.df_skill = pd.read_csv('inference/files/skills.csv')
    self.df_qual = pd.read_csv('inference/files/qualifications.csv')

    #Load embeddings
    self.occupation_emb, self.skill_emb, self.qualification_emb = self._load_tensors()




  def __call__(self, text : str, linking : bool  = True) -> List[dict]:
      """
      Calling the model needs an arbitrary job related text
      """

      text = text.replace('\n', ' ')

      #TODO: Implement the google translate features to enable multilingual entity linking.
      #language = UtilFunctions.detect_language(text)
      #if language != 'en':
      #    text = UtilFunctions.translate(text)

      #Sentence tolenize with nltk to handle lenghty inputs.
      text_list = sent_tokenize(text)
      output = []
      for item in text_list:
          output.extend(self._run_model(item, linking)) if self._run_model(item, linking) else None
      return output

  def _run_model(self, sentence : str, link : bool) -> List[dict]:
      """
      Function that performs entity extraction and entity similarity with the corresponding knowledge base if link=True. 
      As an input needs the sentence to permorm the linking. It returns a dictionary with the necessary information. 
      """

      #Extract entitiies from text
      formatted_entities = self._ner_pipeline(sentence)
      #Check wether or not linking should be performed.
      if link:
        for entry in formatted_entities:
            if entry['type'] == "Occupation" or entry['type'] == "Skill" or entry['type'] == "Qualification":
                emb = self.similarity_model.encode(entry['tokens'])
                emb = torch.from_numpy(emb).to(self.device)
                # Returns the top-k suggestions based on the extracted entity. If evaliation mode=True, return the cosine similatity scores as well.
                if self.evaluation_mode:
                    entry['retrieved'], entry['scores'] = self._top_k(emb, entry['type'])
                else:
                    entry['retrieved'] = self._top_k(emb, entry['type'])

      return formatted_entities



  def _ner_pipeline(self, text : str) -> List[dict]:
      """
      Entity extraction pipeline. Runs the text through the BERT-based encoders, performs some post processing for tagging cleanup 
      and returns a dictionary with all relevant information.
      """
      #Tokenize inputs
      inputs = self.tokenizer(text, return_tensors='pt', truncation=True).to(self.device)
      #Check whether a CRF entity ectraction model is used and produce the logits and prediction entity numerical categories.
      if self.crf:
          with torch.no_grad():
              logits = self.entity_model(**inputs)
          predictions = logits[1][0]
      else:
          with torch.no_grad():
              logits = self.entity_model(**inputs).logits
          predictions = torch.argmax(logits, dim=2)
      
      #Produce the BIO tags
      predicted_token_class = [self.entity_model.config.id2label[t.item()] for t in predictions[0]]
      #Post processing. Hand crafted rules that fix common tagging errors and undesirable outputs
      predicted_token_class = self.fix_bio_tags(predicted_token_class)
      input_ids, predicted_token_class = self.remove_special_tokens_and_tags(inputs['input_ids'][0], predicted_token_class, self.tokenizer)

      #Format the output
      result = self.extract_entities(input_ids, predicted_token_class)

      #Decode the extracted entities into word n-gramms
      for entry in result:
          sentence = self.tokenizer.decode(entry['tokens'])
          #Fix common decoding error in DeBERTa and RoBERTa, that procuces blank space at the start of some tokens
          if sentence.startswith(' '):
            sentence = sentence[1:]
          entry['tokens'] = sentence
      return result


  def _top_k(self, embedding : torch.Tensor, entity_type : str) -> list:
      """
      Entity similarity pipeline. Checks the type of entity and retrieves the top-k most similar using cosine similarity from the corresponding referance vector database.
      """

      if entity_type == "Occupation":
          local_df = self.df_occ['occupation']
          local_emb = self.occupation_emb
      elif entity_type == "Qualification":
          local_df = self.df_qual['qualification']
          local_emb = self.qualification_emb
      else:
          local_df = self.df_skill['skills']
          local_emb = self.skill_emb

      # We use cosine-similarity and torch.topk to find the highest k cosine similarity scores
      cos_scores = util.cos_sim(embedding, local_emb)[0]
      top_k_scores = torch.topk(cos_scores, k=self.k)
      top_k_list = top_k_scores.indices.tolist()
      top_k = list(local_df.iloc[top_k_list])
      if self.evaluation_mode:
          return top_k, top_k_scores.values.tolist()
      #for better formatted outputs in occupations, we remove the duplicate suggestion codes. 
      return self.remove_duplicates_ordered(top_k)
  
  def _load_tensors(self) -> Tuple[List[torch.Tensor]]:
      """
      Function that loads the embeddings. If they are not cached (ie from_cache=False), this function creates 
      a folder inside the files directory with the name of the Sentence Transformer to store the embeddings
      """
      #TODO: Error handling
      path = 'inference/files/'+str(self.similarity_model_type).split("/")[-1]
      if self.from_cache==True:       
        occupation_emb = self.create_tensors(path+'/occupations.pkl', self.device)
        skill_emb = self.create_tensors(path+'/skills.pkl', self.device)
        qualification_emb = self.create_tensors(path+'/qualifications.pkl', self.device)
      else:   
        os.mkdir(path)
        occupation_emb = self._corpus_embedding(corpus = [occ for occ in self.df_occ['occupation']], entity_type='Occupations', path=path)
        skill_emb = self._corpus_embedding(corpus = [skill for skill in self.df_skill['skills']], entity_type='Skills', path=path)  
        qualification_emb = self._corpus_embedding(corpus = [qual for qual in self.df_qual['qualification']], entity_type='Qualifications', path=path)

      return occupation_emb, skill_emb, qualification_emb

  def _corpus_embedding(self, corpus : List[str] , entity_type : str, path : str) -> List[torch.Tensor]:
      """
      Function that computes and stores the embeddings, if from_cache=False
      """
      corpus_embeddings = self.similarity_model.encode(corpus,convert_to_tensor=True)
      embeddings_path = path+'/'+entity_type.lower()+'.pkl'
      with open(embeddings_path, 'wb') as f:
        pickle.dump(corpus_embeddings, f)
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