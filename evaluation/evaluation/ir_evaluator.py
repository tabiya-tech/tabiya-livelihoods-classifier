"""Information Retrieval evaluator migrated from inference/evaluator.py."""

import numpy as np
import torch
from typing import Dict, List, Set


class InformationRetrievalEvaluator:
    """Evaluates IR settings: given queries and a corpus, computes MRR, Recall@k, NDCG."""

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
        self.queries_ids = [
            qid for qid in queries if qid in relevant_docs and len(relevant_docs[qid]) > 0
        ]
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

    def compute_metrics(self, queries_result_list: List[object]) -> dict:
        num_hits_at_k = {k: 0 for k in self.accuracy_at_k}
        precisions_at_k = {k: [] for k in self.precision_recall_at_k}
        recall_at_k = {k: [] for k in self.precision_recall_at_k}
        MRR = {k: 0 for k in self.mrr_at_k}
        ndcg = {k: [] for k in self.ndcg_at_k}
        AveP_at_k = {k: [] for k in self.map_at_k}

        for query_itr in range(len(queries_result_list)):
            query_id = self.queries_ids[query_itr]
            top_hits = queries_result_list[query_itr]
            query_relevant_docs = self.relevant_docs[query_id]

            for k_val in self.accuracy_at_k:
                for hit in top_hits[0:k_val]:
                    if hit["corpus_id"] in query_relevant_docs:
                        num_hits_at_k[k_val] += 1
                        break

            for k_val in self.precision_recall_at_k:
                num_correct = sum(
                    1 for hit in top_hits[0:k_val] if hit["corpus_id"] in query_relevant_docs
                )
                precisions_at_k[k_val].append(num_correct / k_val)
                recall_at_k[k_val].append(num_correct / len(query_relevant_docs))

            for k_val in self.mrr_at_k:
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        MRR[k_val] += 1.0 / (rank + 1)
                        break

            for k_val in self.ndcg_at_k:
                predicted_relevance = [
                    1 if top_hit["corpus_id"] in query_relevant_docs else 0
                    for top_hit in top_hits[0:k_val]
                ]
                true_relevances = [1] * len(query_relevant_docs)
                ndcg_value = self._compute_dcg_at_k(predicted_relevance, k_val) / max(
                    self._compute_dcg_at_k(true_relevances, k_val), 1e-10
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
            precisions_at_k[k] = float(np.mean(precisions_at_k[k]))
        for k in recall_at_k:
            recall_at_k[k] = float(np.mean(recall_at_k[k]))
        for k in ndcg:
            ndcg[k] = float(np.mean(ndcg[k]))
        for k in MRR:
            MRR[k] /= len(self.queries)
        for k in AveP_at_k:
            AveP_at_k[k] = float(np.mean(AveP_at_k[k]))

        return {
            "accuracy@k": num_hits_at_k,
            "precision@k": precisions_at_k,
            "recall@k": recall_at_k,
            "ndcg@k": ndcg,
            "mrr@k": MRR,
            "map@k": AveP_at_k,
        }

    @staticmethod
    def _compute_dcg_at_k(relevances: List[int], k: int) -> float:
        return sum(
            relevances[i] / np.log2(i + 2) for i in range(min(len(relevances), k))
        )
