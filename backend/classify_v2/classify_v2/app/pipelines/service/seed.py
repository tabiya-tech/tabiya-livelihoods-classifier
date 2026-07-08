"""Default Tabiya pipeline seed helper.

Composes the four-stage `text_input → NER → NEL → results` chain that
mirrors the pre-plugin classify behaviour. A fresh user gets one of these
inserted on their first `GET /v2/pipelines` or `POST /v2/classify`
(design §10 — lazy seeding on the two endpoints an API-only caller can hit).

The seed intentionally leaves NEL's `nel_model_id` and `taxonomy_model_id`
as configurable via env-var-backed defaults; if either is unset the
seeded pipeline still validates because the config_schema declares the
fields required — the service will populate them from the user's active
NEL model config when seeding for a real user.
"""

from __future__ import annotations

from typing import Any

from classify_v2.app.pipelines.repository import StageDocument

DEFAULT_TABIYA_NAME = "Default Tabiya"


def build_default_tabiya_stages(
    *,
    nel_model_id: str,
    taxonomy_model_id: str,
    top_k: int = 5,
    min_similarity: float = 0.0,
) -> list[StageDocument]:
    """Build the four-stage default pipeline."""

    source_config: dict[str, Any] = {"text": ""}
    nel_config: dict[str, Any] = {
        "nel_model_id": nel_model_id,
        "taxonomy_model_id": taxonomy_model_id,
        "top_k": top_k,
        "min_similarity": min_similarity,
    }
    return [
        StageDocument(plugin_id="tabiya.source.text.v1", config=source_config),
        StageDocument(plugin_id="tabiya.ner.v1", config={}),
        StageDocument(plugin_id="tabiya.nel.v1", config=nel_config),
        StageDocument(plugin_id="tabiya.sink.results.v1", config={}),
    ]
