import torch
from torch import Tensor
import logging
from tqdm import trange
from sentence_transformers.util import cos_sim, dot_score
from sentence_transformers.evaluation import SentenceEvaluator
import os
import numpy as np
from typing import List, Dict, Set, Tuple
import pandas as pd
from inference.linker import EntityLinker



class InformationRetrievalEvaluator(SentenceEvaluator):
    """
    This class evaluates an Information Retrieval (IR) setting.

    Given a set of queries and a large corpus set. It will retrieve for each query the top-k most similar document. It measures
    Mean Reciprocal Rank (MRR), Recall@k, and Normalized Discounted Cumulative Gain (NDCG)
    """

    def __init__(
        self,
        queries: Dict[str, str],  # qid => query
        corpus: Dict[str, str],  # cid => doc
        relevant_docs: Dict[str, Set[str]],  # qid => Set[cid]
        corpus_chunk_size: int = 50000,
        mrr_at_k: List[int] = [1, 4, 16, 32],
        ndcg_at_k: List[int] = [1, 4, 16, 32],
        accuracy_at_k: List[int] = [1, 4, 16, 32],
        precision_recall_at_k: List[int] = [1, 4, 16, 32],
        map_at_k: List[int] = [1, 4, 5, 16, 32],
        show_progress_bar: bool = False,
        batch_size: int = 32,
        name: str = "",
    ):
        self.queries_ids = []
        for qid in queries:
            if qid in relevant_docs and len(relevant_docs[qid]) > 0:
                self.queries_ids.append(qid)

        self.queries = [queries[qid] for qid in self.queries_ids]

        self.corpus_ids = list(corpus.keys())
        self.corpus = [corpus[cid] for cid in self.corpus_ids]
        self.relevant_docs = relevant_docs
        self.corpus_chunk_size = corpus_chunk_size
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k

        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




    def compute_metrics(self, queries_result_list: List[object]):
        # Init score computation values
        num_hits_at_k = {k: 0 for k in self.accuracy_at_k}
        precisions_at_k = {k: [] for k in self.precision_recall_at_k}
        recall_at_k = {k: [] for k in self.precision_recall_at_k}
        MRR = {k: 0 for k in self.mrr_at_k}
        ndcg = {k: [] for k in self.ndcg_at_k}
        AveP_at_k = {k: [] for k in self.map_at_k}

        # Compute scores on results
        for query_itr in range(len(queries_result_list)):
            query_id = self.queries_ids[query_itr]

            # Sort scores
            #top_hits = sorted(queries_result_list[query_itr], key=lambda x: x["score"], reverse=True)
            top_hits = queries_result_list[query_itr]
            query_relevant_docs = self.relevant_docs[query_id]

            # Accuracy@k - We count the result correct, if at least one relevant doc is across the top-k documents
            for k_val in self.accuracy_at_k:
                for hit in top_hits[0:k_val]:
                    if hit["corpus_id"] in query_relevant_docs:
                        num_hits_at_k[k_val] += 1
                        break

            # Precision and Recall@k
            for k_val in self.precision_recall_at_k:
                num_correct = 0
                for hit in top_hits[0:k_val]:
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1

                precisions_at_k[k_val].append(num_correct / k_val)
                recall_at_k[k_val].append(num_correct / len(query_relevant_docs))

            # MRR@k
            for k_val in self.mrr_at_k:
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        MRR[k_val] += 1.0 / (rank + 1)
                        break

            # NDCG@k
            for k_val in self.ndcg_at_k:
                predicted_relevance = [
                    1 if top_hit["corpus_id"] in query_relevant_docs else 0 for top_hit in top_hits[0:k_val]
                ]
                true_relevances = [1] * len(query_relevant_docs)

                ndcg_value = self.compute_dcg_at_k(predicted_relevance, k_val) / self.compute_dcg_at_k(
                    true_relevances, k_val
                )
                ndcg[k_val].append(ndcg_value)

            # MAP@k
            for k_val in self.map_at_k:
                num_correct = 0
                sum_precisions = 0

                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1
                        sum_precisions += num_correct / (rank + 1)

                avg_precision = sum_precisions / min(k_val, len(query_relevant_docs))
                AveP_at_k[k_val].append(avg_precision)

        # Compute averages
        for k in num_hits_at_k:
            num_hits_at_k[k] /= len(self.queries)

        for k in precisions_at_k:
            precisions_at_k[k] = np.mean(precisions_at_k[k])

        for k in recall_at_k:
            recall_at_k[k] = np.mean(recall_at_k[k])

        for k in ndcg:
            ndcg[k] = np.mean(ndcg[k])

        for k in MRR:
            MRR[k] /= len(self.queries)

        for k in AveP_at_k:
            AveP_at_k[k] = np.mean(AveP_at_k[k])

        return {
            "accuracy@k": num_hits_at_k,
            "precision@k": precisions_at_k,
            "recall@k": recall_at_k,
            "ndcg@k": ndcg,
            "mrr@k": MRR,
            "map@k": AveP_at_k,
        }

    @staticmethod
    def compute_dcg_at_k(relevances, k):
        dcg = 0
        for i in range(min(len(relevances), k)):
            dcg += relevances[i] / np.log2(i + 2)  # +2 as we start our idx at 0
        return dcg



class NotValidEntity(Exception):
    """
    Custom Class for indicating an exception when an Entity Type is not Occupation, Skill or Qualification
    """
    pass


class Evaluator(EntityLinker):
  def __init__(self, entity_type, **kwargs):
    """
    Evaluator class that inherits the Entity Linker. It computes the queries, corpus, inverted corpus and relevant docs for the InformationRetrievalEvaluator, performs 
    entity linking and computes the Information Retrieval Metrics. 
    Args: entity_type. Occupation, Skill or Qualification
    """
    super().__init__(**kwargs)
    self.dir = os.getcwd() + '/inference/files/'
    self.entity_type = entity_type
    #Set the entity bounds for the minimum cosine similarity to be returned from the Entity Linker
    self.entity_bounds = {'Occupation' : 0, 'Skill': 0.7, 'Qualification': 0.8}
    
    if not isinstance(self.entity_type, str):
      raise NotValidEntity("Entity type must be a string")
        
    if self.entity_type not in self.entity_bounds.keys():
      raise NotValidEntity(f"Invalid entity type. Please set entity_type= {' or '.join(self.entity_bounds.keys())}")

    self.dictionary = self._load_dataset()
    self.indexes = self._build_indexes()
    queries_result_list = self._run_inference()
    results = InformationRetrievalEvaluator(corpus=self.indexes['corpus'], queries=self.indexes['queries'], relevant_docs=self.indexes['relevant_docs'])
    self.output = results.compute_metrics(queries_result_list)


  def _load_dataset(self) -> dict:
    """
    Function that loads the evaluation dataset depending on the entity type.
    For skills, evaluation is based on the extention of the Skill Span dataset provided by https://github.com/jensjorisdecorte/Skill-Extraction-benchmark
    For occupations and qualifications we use custom datasets.
    """
    dictionary = {"sentence": [], "spans":[], "labels": []}

    if self.entity_type=="Skill":

      #Load necessary files in pandas dataframes
      dftech_val = pd.read_csv(self.dir + 'eval/tech_validation_annotations.csv')
      dftech_test = pd.read_csv(self.dir + 'eval/tech_test_annotations.csv')
      dfhouse_val = pd.read_csv(self.dir + 'eval/house_validation_annotations.csv')
      dfhouse_test = pd.read_csv(self.dir + 'eval/house_test_annotations.csv')
      #Prepeocess
      df = pd.concat([dftech_val, dftech_test, dfhouse_val, dfhouse_test], ignore_index=True)
      df = df.drop_duplicates(ignore_index=True)
      #Replace label not present and underspecified with UNK label
      for index, item in enumerate(df['label']):
        if item == 'LABEL NOT PRESENT' or item == 'UNDERSPECIFIED':
          df['label'].iloc[index] = 'UNK'

      #Create the dictionaty with unique sentences, different skill spans and their coresponding labels. The spans may have multiple labels
      for sentence, group in df.groupby('sentence'):
        dictionary["sentence"].append(sentence)
        spans = []
        labels = []
        current_span = None
        for span, label in zip(group['span'], group['label']):
          if span == current_span:
            labels[-1].append(label)
          else:
            labels.append([label])
            spans.append(span)
          current_span = span
        dictionary["spans"].append(spans)
        dictionary["labels"].append(labels)

    elif self.entity_type=="Occupation":

      df = pd.read_csv(self.dir + 'eval/redacted_hahu_test_with_id.csv')
      for i in range(len(df)):
        dictionary['sentence'].append( df['title'][i]+ ' '+ df['description'][i])
        dictionary['spans'].append([df['title'][i]])
        dictionary['labels'].append( [df['esco_code'][i]])

    elif self.entity_type=="Qualification":

      #Load necessary files in pandas dataframes
      df = pd.read_csv(self.dir + 'eval/qualification_mapping.csv')

      #Convert EQF labels to floats
      float_labels =[]
      for i in range(len(df)):
        try:
          float_labels.append(float(df['label'][i][-1]))
        except:
          float_labels.append(float(0))
      df['label'] = float_labels

      #Create the dictionaty with unique sentences, different qualification spans and their coresponding labels.
      for sentence, group in df.groupby('text'):
        dictionary["sentence"].append(sentence)
        spans = []
        labels = []
        current_span = None
        for span, label in zip(group['subtext'], group['label']):
          if span == current_span:
            labels[-1].append(label)
          else:
            labels.append([label])
            spans.append(span)
          current_span = span
        dictionary["spans"].append(spans)
        dictionary["labels"].append(labels)
    return dictionary


  def _build_indexes(self) -> dict:
    """
    Function that builds corpus, queries, relevant docs and invertes coprus depending on the entity type/ evaluation set.
    """
    if self.entity_type=="Occupation":

      #Build Corpus/ Reference Sets & Inverted Coprus with ascending indexes.
      esco = pd.read_csv(self.dir + 'occupations_augmented.csv')
      esco = esco.drop_duplicates(subset='occupation', keep='first', ignore_index=True)
      corpus = {str(k):str(v) for k,v in enumerate(esco['occupation'])}
      corpus['UNK'] = 'UNK'
      corpus_inverted = {v:k for k,v in corpus.items()}

      queries = {}
      relevant_docs ={}
      index = 0
      #Treat as queries each entity span. In this case, the relevant spans are the title of the job description of the Hahu Jobs test set. 
      for i in range(len(self.dictionary['sentence'])):
        #Build queries
        qid= str(i)
        queries[qid] = self.dictionary['sentence'][i]
        #Build relevant docs
        relevant = set()
        for index, item in enumerate(esco['esco_code']):
          if item == self.dictionary['labels'][i][0]:
            relevant.add(corpus_inverted[esco['occupation'][index]])
        if relevant:
          relevant_docs[qid] = relevant
        else:
          relevant_docs[qid] = 'UNK' #there exist some codes that were removed from the latest verison of ESCO. Append the UNK label for those.

    elif self.entity_type=="Skill":

      #Build Corpus/ Reference Sets & Inverted Coprus with ascending indexes.
      esco = pd.read_csv(self.dir + 'skills.csv')
      corpus = {str(k):str(v) for k,v in enumerate(esco['skills'])}
      corpus['UNK'] = 'UNK'
      corpus_inverted = {v:k for k,v in corpus.items()}

      queries = {}
      relevant_docs ={}
      index = 0
      #Treat as queries each entity span. In this case, the relevant spans are the unique skill spans presented in the SkillSpan extention.
      for i in range(len(self.dictionary['sentence'])):
        for index2, span in enumerate(self.dictionary['spans'][i]):
          #Build queries
          qid= str(index)
          queries[qid] = span
          index+=1
          #Build relevant docs
          relevant = set()
          for label in self.dictionary['labels'][i][index2]:
            relevant.add(corpus_inverted[label])
          relevant_docs[qid] = relevant

    elif self.entity_type=="Qualification":

      #Build Corpus/ Reference Sets & Inverted Coprus with ascending indexes.
      esco = pd.read_csv(self.dir + 'qualifications.csv')
      esco = esco.sort_values(by='eqf_level', ascending=False)
      esco = esco.drop_duplicates(subset='qualification', keep='first', ignore_index=True)
      corpus = {str(k):str(v) for k,v in enumerate(esco['qualification'])}
      corpus['UNK'] = 'UNK'
      corpus_inverted = {v:k for k,v in corpus.items()}
      queries = {}
      relevant_docs ={}
      index = 0
      #Treat as queries each entity span. In this case, the relevant spans are the unique qualification spans presented in the Green Benchmark extention.
      for i in range(len(self.dictionary['sentence'])):
        for index2, span in enumerate(self.dictionary['spans'][i]):
          #Build queries
          qid= str(index)
          queries[qid] = span
          index+=1
          #Build relevant docs
          relevant = set()
          for label in self.dictionary['labels'][i][index2]:
            for index3, esco_label in enumerate(esco['eqf_level']):
              if label == esco_label:
                relevant.add(corpus_inverted[esco['qualification'][index3]])
              if label==0.0:
                relevant.add(corpus_inverted['UNK'])
          relevant_docs[qid] = relevant

    return {
        "corpus": corpus,
        "corpus_inverted": corpus_inverted,
        "queries" : queries,
        "relevant_docs" : relevant_docs
        }

  def _run_inference(self) -> List[List[dict]]:
    """
    Functions that runs the Entity Linker, finds the most similar extracted entity on the evaluation set based on the Jaccard similarity and returns 
    a dictionary of the recommendations and their respective cosine similarity score.
    """
    queries_result_list = []
    for number in range(len(self.dictionary["sentence"])):
      relevant_spans = self.dictionary["spans"][number]
      #Run Entity Linker
      result = self(self.dictionary["sentence"][number])
      for relevant_span in relevant_spans:
        retrieved, scores = self._most_similar(relevant_span, result, self.entity_bounds[self.entity_type])
        final_retrieved = []
        for index, item in enumerate(retrieved):
          #Reference the recommended items to the relevent corpus to get the coprus id.
          corpus_id = self.indexes["corpus_inverted"][item]
          final_retrieved.append({"corpus_id": corpus_id, "score": float(scores[index])})
        queries_result_list.append(final_retrieved)
    return queries_result_list

  def _most_similar(self, relevant_span : str, entity_list : List[dict], bound : float) -> Tuple[list,list]:
    """
    Function that computes the most similar extracted entity in the evaluation sentence.
    """
    max = 0
    index = 0
    for i, entity in enumerate(entity_list):
      if entity['type'] == self.entity_type:
        similarity = self.Jaccard_Similarity(relevant_span, entity['tokens'])
        if similarity > max:
          max = similarity
          index = i
    if max > 0 and entity_list[index]['scores'][0] > bound:
      return entity_list[index]['retrieved'], entity_list[index]['scores']
    else:
      return ['UNK'], [1]

  @staticmethod
  def Jaccard_Similarity(doc1:str, doc2:str) -> float:

      words_doc1 = set(doc1.lower().split())
      words_doc2 = set(doc2.lower().split())

      intersection = words_doc1.intersection(words_doc2)

      union = words_doc1.union(words_doc2)

      # Calculate Jaccard similarity score
      # using length of intersection set divided by length of union set
      return float(len(intersection)) / len(union)