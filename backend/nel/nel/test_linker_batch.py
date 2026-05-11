"""Tests for batched NEL encoding (point E)."""

from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch

from inference.nel import NELLinker


def _stub_linker_for_encode_tracking(embedding_dim: int = 384):
    """Minimal linker instance without loading real CSVs or SentenceTransformer."""
    linker = NELLinker.__new__(NELLinker)
    linker.device = torch.device("cpu")
    linker.k = 32

    linker.df_occ = pd.DataFrame(
        [
            {"occupation": "Chef", "uuid": "u1", "esco_code": "3434.1"},
            {"occupation": "Cook", "uuid": "u2", "esco_code": "3434.2"},
        ]
    )
    linker.occupation_emb = torch.nn.functional.normalize(torch.randn(2, embedding_dim), dim=1)

    linker.df_skill = pd.DataFrame([{"skills": "Python", "uuid": "s1"}])
    linker.skill_emb = torch.nn.functional.normalize(torch.randn(1, embedding_dim), dim=1)

    linker.df_qual = pd.DataFrame([{"qualification": "Bachelor", "eqf_level": "6"}])
    linker.qualification_emb = torch.nn.functional.normalize(torch.randn(1, embedding_dim), dim=1)

    def fake_encode(texts, convert_to_tensor=True, **_kwargs):
        n = len(texts)
        t = torch.nn.functional.normalize(torch.randn(n, embedding_dim), dim=1)
        return t if convert_to_tensor else t.numpy()

    linker.similarity_model = MagicMock()
    linker.similarity_model.encode = MagicMock(side_effect=fake_encode)
    return linker


class TestNELLinkerBatchEncode:
    def test_one_encode_call_per_entity_type_group(self):
        linker = _stub_linker_for_encode_tracking()
        entities = [
            {"text": "Head Chef", "entity_type": "occupation"},
            {"text": "Sous Chef", "entity_type": "occupation"},
            {"text": "Python", "entity_type": "skill"},
        ]

        out = linker.link(entities, top_k=2, min_similarity=0.0)

        assert len(out) == 3
        assert linker.similarity_model.encode.call_count == 2
        calls = linker.similarity_model.encode.call_args_list
        assert calls[0][0][0] == ["Head Chef", "Sous Chef"]
        assert calls[1][0][0] == ["Python"]

        assert out[0]["entity_type"] == "occupation"
        assert out[1]["entity_type"] == "occupation"
        assert out[2]["entity_type"] == "skill"
        assert all("matches" in o for o in out)

    def test_non_linkable_skips_encode(self):
        linker = _stub_linker_for_encode_tracking()
        entities = [
            {"text": "5 years", "entity_type": "experience"},
            {"text": "Chef", "entity_type": "occupation"},
        ]

        out = linker.link(entities, top_k=1, min_similarity=0.0)

        assert len(out) == 2
        linker.similarity_model.encode.assert_called_once()
        assert linker.similarity_model.encode.call_args[0][0] == ["Chef"]
        assert out[0]["matches"] == []
        assert out[1]["entity_type"] == "occupation"

    def test_empty_entities_returns_empty_list(self):
        linker = _stub_linker_for_encode_tracking()
        assert linker.link([], top_k=5) == []
        linker.similarity_model.encode.assert_not_called()
