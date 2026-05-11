from typing import List, Optional, Tuple

import os
import pickle

import pandas as pd
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

from inference.linker import EntityLinker

load_dotenv()


class NELLinker:
    """Links entity text to ESCO taxonomy entries using embedding similarity."""

    def __init__(
        self,
        similarity_model: str = "all-MiniLM-L6-v2",
        k: int = 32,
        from_cache: bool = True,
    ):
        self.similarity_model_name = similarity_model
        self.similarity_model = SentenceTransformer(similarity_model)
        self.k = k
        self.from_cache = from_cache
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path_to_files = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "files")
        )

        self.df_occ = pd.read_csv(os.path.join(self.path_to_files, "occupations_augmented.csv"))
        self.df_skill = pd.read_csv(os.path.join(self.path_to_files, "skills.csv"))
        self.df_qual = pd.read_csv(os.path.join(self.path_to_files, "qualifications.csv"))

        self.occupation_emb, self.skill_emb, self.qualification_emb = self._load_tensors()

    def link(
        self,
        entities: List[dict],
        top_k: Optional[int] = None,
        min_similarity: float = 0.0,
    ) -> List[dict]:
        """Link entities to ESCO taxonomy entries.

        Each entity dict needs ``text`` and ``entity_type`` keys.
        Linkable types are batch-encoded per type (one ``encode`` call per type group).
        """
        k = top_k or self.k
        n = len(entities)
        results: List[Optional[dict]] = [None] * n

        occ_texts: List[str] = []
        occ_idx: List[int] = []
        skill_texts: List[str] = []
        skill_idx: List[int] = []
        qual_texts: List[str] = []
        qual_idx: List[int] = []

        for i, entity in enumerate(entities):
            text = entity["text"]
            et = entity["entity_type"].lower()

            if et not in ("occupation", "skill", "qualification"):
                results[i] = {"input_text": text, "entity_type": et, "matches": []}
            elif et == "occupation":
                occ_texts.append(text)
                occ_idx.append(i)
            elif et == "qualification":
                qual_texts.append(text)
                qual_idx.append(i)
            else:
                skill_texts.append(text)
                skill_idx.append(i)

        self._link_batch_group(entities, occ_texts, occ_idx, "occupation", k, min_similarity, results)
        self._link_batch_group(entities, skill_texts, skill_idx, "skill", k, min_similarity, results)
        self._link_batch_group(entities, qual_texts, qual_idx, "qualification", k, min_similarity, results)

        assert None not in results
        return results  # type: ignore[return-value]

    def _link_batch_group(
        self,
        entities: List[dict],
        texts: List[str],
        indices: List[int],
        entity_type: str,
        k: int,
        min_similarity: float,
        results: List[Optional[dict]],
    ) -> None:
        if not texts:
            return
        embs = self.similarity_model.encode(texts, convert_to_tensor=True).to(self.device)
        batch_matches = self._top_k_batch(embs, entity_type, k, min_similarity)
        for j, orig_i in enumerate(indices):
            results[orig_i] = {
                "input_text": entities[orig_i]["text"],
                "entity_type": entities[orig_i]["entity_type"].lower(),
                "matches": batch_matches[j],
            }

    def _top_k(
        self,
        embedding: torch.Tensor,
        entity_type: str,
        k: int,
        min_similarity: float,
    ) -> List[dict]:
        """Retrieve top-k ESCO matches for a single entity embedding."""
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        return self._top_k_batch(embedding, entity_type, k, min_similarity)[0]

    def _top_k_batch(
        self,
        embeddings: torch.Tensor,
        entity_type: str,
        k: int,
        min_similarity: float,
    ) -> List[List[dict]]:
        """Top-k matches for each row of ``embeddings`` (shape B×D) against the taxonomy corpus."""
        if entity_type == "occupation":
            local_df = self.df_occ
            local_emb = self.occupation_emb
        elif entity_type == "qualification":
            local_df = self.df_qual
            local_emb = self.qualification_emb
        else:
            local_df = self.df_skill
            local_emb = self.skill_emb

        cos_scores = util.cos_sim(embeddings, local_emb)
        k_eff = min(k, cos_scores.shape[1])
        out: List[List[dict]] = []

        for row in range(cos_scores.shape[0]):
            top_k_results = torch.topk(cos_scores[row], k=k_eff)
            matches: List[dict] = []
            for idx, score in zip(
                top_k_results.indices.tolist(),
                top_k_results.values.tolist(),
            ):
                if score < min_similarity:
                    continue
                row_df = local_df.iloc[idx]
                matches.append(self._match_from_row(row_df, entity_type, float(score)))
            out.append(matches)

        return out

    def _match_from_row(self, row: pd.Series, entity_type: str, score: float) -> dict:
        match = {"similarity_score": round(score, 4), "taxonomy": "esco"}

        if entity_type == "occupation":
            match["label"] = row.get("occupation", row.get("preffered_label", ""))
            if "esco_code" in row:
                match["code"] = str(row["esco_code"])
            if "uuid" in row:
                match["uri"] = f"http://data.europa.eu/esco/occupation/{row['uuid']}"
        elif entity_type == "skill":
            match["label"] = row.get("skills", "")
            if "uuid" in row:
                match["uri"] = f"http://data.europa.eu/esco/skill/{row['uuid']}"
        elif entity_type == "qualification":
            match["label"] = row.get("qualification", "")
            if "eqf_level" in row:
                match["eqf_level"] = str(row["eqf_level"])

        return match

    def _load_tensors(self) -> Tuple:
        """Load precomputed or compute fresh embeddings for all reference sets.

        uses EntityLinker.create_tensors() for the cached-load path.
        """
        path = os.path.join(self.path_to_files, self.similarity_model_name)

        if self.from_cache:
            occupation_emb = EntityLinker.create_tensors(
                os.path.join(path, "occupations.pkl"), self.device
            )
            skill_emb = EntityLinker.create_tensors(
                os.path.join(path, "skills.pkl"), self.device
            )
            qualification_emb = EntityLinker.create_tensors(
                os.path.join(path, "qualifications.pkl"), self.device
            )
        else:
            os.makedirs(path, exist_ok=True)
            occupation_emb = self._compute_and_cache(
                list(self.df_occ["occupation"]), "occupations", path
            )
            skill_emb = self._compute_and_cache(
                list(self.df_skill["skills"]), "skills", path
            )
            qualification_emb = self._compute_and_cache(
                list(self.df_qual["qualification"]), "qualifications", path
            )

        return occupation_emb, skill_emb, qualification_emb

    def _compute_and_cache(self, corpus: List[str], name: str, path: str) -> torch.Tensor:
        """Encode corpus, cache to disk, and return the tensor."""
        embeddings = self.similarity_model.encode(corpus, convert_to_tensor=True)
        filepath = os.path.join(path, f"{name}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(embeddings, f)
        return EntityLinker.create_tensors(filepath, self.device)
