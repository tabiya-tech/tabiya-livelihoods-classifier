from typing import List, Optional
import pickle
#from google.cloud import translate_v2 as translate
import torch
import numpy as np
import json


class UtilFunctions:

  """
  A class containing methods for the Custom Pipeline

  extract_entities: Given two lists of tokens and their BIO tags, it returns a list of dictionaries with the extracted token spans and their respective tag.
  decode: Method for decoding wordpiece and byte pair encoding tokens to the original text.
  create_tensors: Load the vector databases.  

  """

  @staticmethod
  def extract_entities(tokens : list, tags : list) -> List[dict]:
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
    #Post Processing. Remove empty entries in results or entries that have only one token that starts with ##
    condition_function = lambda x: len(x['tokens']) != 0 and (len(x['tokens']) != 1 and not x['tokens'][0].startswith('##'))
    filtered_list = [item for item in result if condition_function(item)]

    return filtered_list


  @staticmethod
  def decode(list_of_tokens : list , tokenizer_type : str) -> str:
    """
    These decoding strategies are naive. They will be decapetated.
    Use cases: wordpiece for BERT
               bpe for DeBERTa
    """
    pretok_sent = ""
    if tokenizer_type == "wordpiece":
      for tok in list_of_tokens:
        if tok.startswith("##"):
          pretok_sent += tok[2:]
        else:
          pretok_sent += " " + tok
      pretok_sent = pretok_sent[1:]
    elif tokenizer_type == "bpe":
      for tok in list_of_tokens:
        if tok.startswith("Ġ"):
          pretok_sent += " " + tok[1:]
        else:
          pretok_sent += tok
    return pretok_sent


  @staticmethod
  def create_tensors(file : str, device : str) -> torch.Tensor:
    with open(file, 'rb') as f:
      embeddings = pickle.load(f)
    embeddings = torch.from_numpy(np.array(embeddings)).to(device)
    return embeddings

  @staticmethod
  def remove_duplicates_ordered(input_list : list) -> list:
    unique_list = []
    for item in input_list:
        if item not in unique_list:
            unique_list.append(item)
    return unique_list

  #@staticmethod
  #def detect_language(text : str) -> str:
  #  result = ftlangdetect.detect(text=text, low_memory=False)
  #  return result['lang']


  #@staticmethod
  #def translate(text : str) -> str:
  #  translate_client = translate.Client()
  #  result = translate_client.translate(text, target_language='en')
  #  sentence = result['translatedText']
  #  return sentence

class Config(object):
  """
  Configuration class for the training hyperparameters
  """
  def __init__(self, config_file):
      # Initialization from a json configuration file.
      self._readConfigFile(config_file)
  def _readConfigFile(self, file):
      try:
          with open(file, 'r') as cfg:
              data = json.load(cfg)
          for key, value in data.items():
              setattr(self, key, value)
      except:
          print('Configuration file read error')
          raise