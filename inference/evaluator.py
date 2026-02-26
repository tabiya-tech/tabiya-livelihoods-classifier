import os, sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

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
from linker import EntityLinker



class InformationRetrievalEvaluator(SentenceEvaluator):
    """Evaluates IR metrics (MRR, Recall@k, NDCG) for query-to-corpus retrieval."""

    def __init__(
        self,
        queries: Dict[str, str],
        corpus: Dict[str, str],
        relevant_docs: Dict[str, Set[str]],
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
        num_hits_at_k = {k: 0 for k in self.accuracy_at_k}
        precisions_at_k = {k: [] for k in self.precision_recall_at_k}
        recall_at_k = {k: [] for k in self.precision_recall_at_k}
        MRR = {k: 0 for k in self.mrr_at_k}
        ndcg = {k: [] for k in self.ndcg_at_k}
        AveP_at_k = {k: [] for k in self.map_at_k}

        for query_itr in range(len(queries_result_list)):
            query_id = self.queries_ids[query_itr]

            #top_hits = sorted(queries_result_list[query_itr], key=lambda x: x["score"], reverse=True)
            top_hits = queries_result_list[query_itr]
            query_relevant_docs = self.relevant_docs[query_id]

            for k_val in self.accuracy_at_k:
                for hit in top_hits[0:k_val]:
                    if hit["corpus_id"] in query_relevant_docs:
                        num_hits_at_k[k_val] += 1
                        break

            for k_val in self.precision_recall_at_k:
                num_correct = 0
                for hit in top_hits[0:k_val]:
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1

                precisions_at_k[k_val].append(num_correct / k_val)
                recall_at_k[k_val].append(num_correct / len(query_relevant_docs))

            for k_val in self.mrr_at_k:
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        MRR[k_val] += 1.0 / (rank + 1)
                        break

            for k_val in self.ndcg_at_k:
                predicted_relevance = [
                    1 if top_hit["corpus_id"] in query_relevant_docs else 0 for top_hit in top_hits[0:k_val]
                ]
                true_relevances = [1] * len(query_relevant_docs)

                ndcg_value = self.compute_dcg_at_k(predicted_relevance, k_val) / self.compute_dcg_at_k(
                    true_relevances, k_val
                )
                ndcg[k_val].append(ndcg_value)

            for k_val in self.map_at_k:
                num_correct = 0
                sum_precisions = 0

                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1
                        sum_precisions += num_correct / (rank + 1)

                avg_precision = sum_precisions / min(k_val, len(query_relevant_docs))
                AveP_at_k[k_val].append(avg_precision)

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
            dcg += relevances[i] / np.log2(i + 2)  # +2 because idx is 0-based
        return dcg



class NotValidEntity(Exception):
    pass


class Evaluator(EntityLinker):
  """Runs entity linking on evaluation datasets and computes IR metrics."""

  def __init__(self, entity_type, **kwargs):
    super().__init__(**kwargs)
    self.dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'files'))
    self.entity_type = entity_type
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
    """Load evaluation dataset for the configured entity type.

    Skills use the SkillSpan benchmark (https://github.com/jensjorisdecorte/Skill-Extraction-benchmark).
    Occupations and qualifications use custom datasets.
    """
    dictionary = {"sentence": [], "spans":[], "labels": []}

    if self.entity_type=="Skill":

      dftech_val = pd.read_csv(os.path.join(self.dir, 'eval', 'tech_validation_annotations.csv'))
      dftech_test = pd.read_csv(os.path.join(self.dir, 'eval', 'tech_test_annotations.csv'))
      dfhouse_val = pd.read_csv(os.path.join(self.dir, 'eval', 'house_validation_annotations.csv'))
      dfhouse_test = pd.read_csv(os.path.join(self.dir, 'eval', 'house_test_annotations.csv'))

      df = pd.concat([dftech_val, dftech_test, dfhouse_val, dfhouse_test], ignore_index=True)
      df = df.drop_duplicates(ignore_index=True)

      for index, item in enumerate(df['label']):
        if item == 'LABEL NOT PRESENT' or item == 'UNDERSPECIFIED':
          df['label'].iloc[index] = 'UNK'

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

      df = pd.read_csv(os.path.join(self.dir, 'eval', 'redacted_hahu_test_with_id.csv'))
      for i in range(len(df)):
        dictionary['sentence'].append( df['title'][i]+ ' '+ df['description'][i])
        dictionary['spans'].append([df['title'][i]])
        dictionary['labels'].append( [df['esco_code'][i]])

    elif self.entity_type=="Qualification":

      df = pd.read_csv(os.path.join(self.dir, 'eval', 'qualification_mapping.csv'))

      float_labels =[]
      for i in range(len(df)):
        try:
          float_labels.append(float(df['label'][i][-1]))
        except:
          float_labels.append(float(0))
      df['label'] = float_labels

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
    """Build corpus, queries, relevant docs and inverted corpus for evaluation."""
    if self.entity_type=="Occupation":

      esco = pd.read_csv(os.path.join(self.dir, 'occupations_augmented.csv'))
      esco = esco.drop_duplicates(subset='occupation', keep='first', ignore_index=True)
      corpus = {str(k):str(v) for k,v in enumerate(esco['occupation'])}
      corpus['UNK'] = 'UNK'
      corpus_inverted = {v:k for k,v in corpus.items()}

      queries = {}
      relevant_docs ={}
      index = 0

      for i in range(len(self.dictionary['sentence'])):
        qid= str(i)
        queries[qid] = self.dictionary['sentence'][i]

        relevant = set()
        for index, item in enumerate(esco['esco_code']):
          if item == self.dictionary['labels'][i][0]:
            relevant.add(corpus_inverted[esco['occupation'][index]])
        if relevant:
          relevant_docs[qid] = relevant
        else:
          # Some codes were removed from the latest ESCO version
          relevant_docs[qid] = 'UNK'

    elif self.entity_type=="Skill":

      esco = pd.read_csv(os.path.join(self.dir, 'skills.csv'))
      corpus = {str(k):str(v) for k,v in enumerate(esco['skills'])}
      corpus['UNK'] = 'UNK'
      corpus_inverted = {v:k for k,v in corpus.items()}

      queries = {}
      relevant_docs ={}
      index = 0

      for i in range(len(self.dictionary['sentence'])):
        for index2, span in enumerate(self.dictionary['spans'][i]):
          qid= str(index)
          queries[qid] = span
          index+=1

          relevant = set()
          for label in self.dictionary['labels'][i][index2]:
            relevant.add(corpus_inverted[label])
          relevant_docs[qid] = relevant

    elif self.entity_type=="Qualification":

      esco = pd.read_csv(os.path.join(self.dir, 'qualifications.csv'))
      esco = esco.sort_values(by='eqf_level', ascending=False)
      esco = esco.drop_duplicates(subset='qualification', keep='first', ignore_index=True)
      corpus = {str(k):str(v) for k,v in enumerate(esco['qualification'])}
      corpus['UNK'] = 'UNK'
      corpus_inverted = {v:k for k,v in corpus.items()}
      queries = {}
      relevant_docs ={}
      index = 0

      for i in range(len(self.dictionary['sentence'])):
        for index2, span in enumerate(self.dictionary['spans'][i]):
          qid= str(index)
          queries[qid] = span
          index+=1

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
    """Run entity linking on the evaluation set, match by Jaccard similarity, and collect results."""
    queries_result_list = []
    for number in range(len(self.dictionary["sentence"])):
      relevant_spans = self.dictionary["spans"][number]
      result = self(self.dictionary["sentence"][number])
      for relevant_span in relevant_spans:
        retrieved, scores = self._most_similar(relevant_span, result, self.entity_bounds[self.entity_type])
        final_retrieved = []
        for index, item in enumerate(retrieved):
          corpus_id = self.indexes["corpus_inverted"][item]
          final_retrieved.append({"corpus_id": corpus_id, "score": float(scores[index])})
        queries_result_list.append(final_retrieved)
    return queries_result_list

  def _most_similar(self, relevant_span : str, entity_list : List[dict], bound : float) -> Tuple[list,list]:
    """Find the extracted entity most similar to the relevant span (by Jaccard), above the similarity bound."""
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
      return float(len(intersection)) / len(union)
