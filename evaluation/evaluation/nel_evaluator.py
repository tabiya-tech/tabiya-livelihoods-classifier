"""NEL evaluation: runs the NEL linker against annotated datasets and computes IR metrics.

Migrated and refactored from inference/evaluator.py.
"""

import os
from typing import Dict, List, Set, Tuple

import pandas as pd

from evaluation.ir_evaluator import InformationRetrievalEvaluator


class NotValidEntity(Exception):
    pass


class NELEvaluator:
    """Evaluates the NEL linker against annotated datasets.

    Usage::

        evaluator = NELEvaluator(entity_type="Occupation", files_dir="path/to/inference/files")
        print(evaluator.output)
    """

    ENTITY_BOUNDS = {"Occupation": 0.0, "Skill": 0.7, "Qualification": 0.8}

    def __init__(self, entity_type: str, files_dir: str):
        if entity_type not in self.ENTITY_BOUNDS:
            raise NotValidEntity(
                f"Invalid entity_type. Choose from: {', '.join(self.ENTITY_BOUNDS)}"
            )
        self.entity_type = entity_type
        self.files_dir = files_dir
        self.eval_dir = os.path.join(files_dir, "eval")

        # Lazy import to avoid loading heavy deps unless needed
        from backend.nel.nel.linker import NELLinker  # type: ignore

        self.linker = NELLinker(files_path=files_dir)
        self.dictionary = self._load_dataset()
        self.indexes = self._build_indexes()
        queries_result_list = self._run_inference()
        ir = InformationRetrievalEvaluator(
            corpus=self.indexes["corpus"],
            queries=self.indexes["queries"],
            relevant_docs=self.indexes["relevant_docs"],
        )
        self.output = ir.compute_metrics(queries_result_list)

    def _load_dataset(self) -> dict:
        dictionary: dict = {"sentence": [], "spans": [], "labels": []}

        if self.entity_type == "Skill":
            dfs = [
                pd.read_csv(os.path.join(self.eval_dir, f))
                for f in [
                    "tech_validation_annotations.csv",
                    "tech_test_annotations.csv",
                    "house_validation_annotations.csv",
                    "house_test_annotations.csv",
                ]
            ]
            df = pd.concat(dfs, ignore_index=True).drop_duplicates(ignore_index=True)
            df["label"] = df["label"].replace(
                ["LABEL NOT PRESENT", "UNDERSPECIFIED"], "UNK"
            )
            for sentence, group in df.groupby("sentence"):
                spans, labels, current_span = [], [], None
                for span, label in zip(group["span"], group["label"]):
                    if span == current_span:
                        labels[-1].append(label)
                    else:
                        labels.append([label])
                        spans.append(span)
                    current_span = span
                dictionary["sentence"].append(sentence)
                dictionary["spans"].append(spans)
                dictionary["labels"].append(labels)

        elif self.entity_type == "Occupation":
            df = pd.read_csv(os.path.join(self.eval_dir, "redacted_hahu_test_with_id.csv"))
            for i in range(len(df)):
                dictionary["sentence"].append(df["title"][i] + " " + df["description"][i])
                dictionary["spans"].append([df["title"][i]])
                dictionary["labels"].append([df["esco_code"][i]])

        elif self.entity_type == "Qualification":
            df = pd.read_csv(os.path.join(self.eval_dir, "qualification_mapping.csv"))
            float_labels = []
            for label in df["label"]:
                try:
                    float_labels.append(float(str(label)[-1]))
                except Exception:
                    float_labels.append(0.0)
            df["label"] = float_labels
            for sentence, group in df.groupby("text"):
                spans, labels, current_span = [], [], None
                for span, label in zip(group["subtext"], group["label"]):
                    if span == current_span:
                        labels[-1].append(label)
                    else:
                        labels.append([label])
                        spans.append(span)
                    current_span = span
                dictionary["sentence"].append(sentence)
                dictionary["spans"].append(spans)
                dictionary["labels"].append(labels)

        return dictionary

    def _build_indexes(self) -> dict:
        if self.entity_type == "Occupation":
            esco = pd.read_csv(os.path.join(self.files_dir, "occupations_augmented.csv"))
            esco = esco.drop_duplicates(subset="occupation", keep="first", ignore_index=True)
            corpus = {str(k): str(v) for k, v in enumerate(esco["occupation"])}
            corpus["UNK"] = "UNK"
            corpus_inverted = {v: k for k, v in corpus.items()}
            queries, relevant_docs = {}, {}
            for i in range(len(self.dictionary["sentence"])):
                qid = str(i)
                queries[qid] = self.dictionary["sentence"][i]
                relevant = {
                    corpus_inverted[esco["occupation"][idx]]
                    for idx, code in enumerate(esco["esco_code"])
                    if code == self.dictionary["labels"][i][0]
                }
                relevant_docs[qid] = relevant or {"UNK"}

        elif self.entity_type == "Skill":
            esco = pd.read_csv(os.path.join(self.files_dir, "skills.csv"))
            corpus = {str(k): str(v) for k, v in enumerate(esco["skills"])}
            corpus["UNK"] = "UNK"
            corpus_inverted = {v: k for k, v in corpus.items()}
            queries, relevant_docs = {}, {}
            idx = 0
            for i in range(len(self.dictionary["sentence"])):
                for j, span in enumerate(self.dictionary["spans"][i]):
                    qid = str(idx)
                    queries[qid] = span
                    relevant_docs[qid] = {corpus_inverted.get(l, "UNK") for l in self.dictionary["labels"][i][j]}
                    idx += 1

        elif self.entity_type == "Qualification":
            esco = pd.read_csv(os.path.join(self.files_dir, "qualifications.csv"))
            esco = esco.sort_values("eqf_level", ascending=False).drop_duplicates(
                subset="qualification", keep="first", ignore_index=True
            )
            corpus = {str(k): str(v) for k, v in enumerate(esco["qualification"])}
            corpus["UNK"] = "UNK"
            corpus_inverted = {v: k for k, v in corpus.items()}
            queries, relevant_docs = {}, {}
            idx = 0
            for i in range(len(self.dictionary["sentence"])):
                for j, span in enumerate(self.dictionary["spans"][i]):
                    qid = str(idx)
                    queries[qid] = span
                    relevant = set()
                    for label in self.dictionary["labels"][i][j]:
                        for k2, eqf in enumerate(esco["eqf_level"]):
                            if label == eqf or label == 0.0:
                                key = corpus_inverted.get(esco["qualification"][k2], "UNK")
                                relevant.add(key)
                    relevant_docs[qid] = relevant or {"UNK"}
                    idx += 1

        return {
            "corpus": corpus,
            "corpus_inverted": corpus_inverted,
            "queries": queries,
            "relevant_docs": relevant_docs,
        }

    def _run_inference(self) -> List[List[dict]]:
        queries_result_list = []
        bound = self.ENTITY_BOUNDS[self.entity_type]
        for i in range(len(self.dictionary["sentence"])):
            result = self.linker.link(
                [{"text": span, "entity_type": self.entity_type.lower()} for span in self.dictionary["spans"][i]],
                top_k=32,
            )
            for j, item in enumerate(result):
                final_retrieved = [
                    {
                        "corpus_id": self.indexes["corpus_inverted"].get(m["label"], "UNK"),
                        "score": m["similarity_score"],
                    }
                    for m in item["matches"]
                    if m["similarity_score"] >= bound
                ] or [{"corpus_id": "UNK", "score": 1.0}]
                queries_result_list.append(final_retrieved)
        return queries_result_list
