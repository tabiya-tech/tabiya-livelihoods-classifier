"""Spanish language config (Argentina / Empujar — taxonomy locale ``AR-es``).

Data pack: ``nel/nel/files/es/``, generated from the AR-es taxonomy export with
``backend/nel/scripts/build_nel_files.py``. Its ``uuid`` column is identical to the
English pack's — ``UUIDHISTORY[-1]`` is stable across taxonomy locales — so entities
linked in Spanish resolve to the same taxonomy entries downstream.

Override via env: ``NER_MODEL_ES``, ``LINKER_MODEL_ES``, ``TAXONOMY_MODEL_ID_ES``.
"""

from shared.languages._env import lang_env_str

# `ner` has no Spanish-specific checkpoint yet, so it falls back to the English one and
# says so (loudly at load, and in every response's metadata). Extraction quality on
# Spanish text is poor: this is the known gap tracked in `empujar/task.md` V2.
# Point NER_MODEL_ES at a Spanish or multilingual token-classification checkpoint to
# close it — no code change, and set `ner_label_map` below if its label vocabulary is
# not occupation / skill / qualification.
_NER_MODEL_FALLBACK = "tabiya/roberta-base-job-ner"
_ner_model = lang_env_str("NER_MODEL", "es", _NER_MODEL_FALLBACK)

LANGUAGE_CONFIG = {
    "language": "es",
    "name": "Spanish",
    "locales": (
        "es",
        "spa",
        "spanish",
        "espanol",
        "ar-es",  # taxonomy export locale (model_info.csv LOCALE)
        "es-ar",
        "es-419",
        "es-es",
        "es-mx",
        "argentina",  # TARGET_COUNTRY spellings, so one variable can drive the run
        "argentine",
    ),
    # ── ner ───────────────────────────────────────────────────────────────────
    "ner_model": _ner_model,
    "ner_crf": False,
    # False ⇒ the service warns that extraction is running on a model trained for
    # another language, and marks it in the response metadata.
    "ner_model_is_language_specific": _ner_model != _NER_MODEL_FALLBACK,
    # Map a checkpoint's own labels onto the pipeline's entity types. Empty = identity.
    # Example for a knowledge/skill checkpoint: {"knowledge": "skill", "skill": "skill"}
    "ner_label_map": {},
    "sentence_tokenizer_language": "spanish",
    # ── nel (v1: local CSV + sentence-transformer embeddings) ─────────────────
    # Multilingual sentence-transformer, same 384 dims as the English all-MiniLM-L6-v2.
    "nel_similarity_model": lang_env_str(
        "LINKER_MODEL", "es", "paraphrase-multilingual-MiniLM-L12-v2"
    ),
    "nel_files_subdir": "es",
    # ── nel_v2 (taxonomy API + Atlas vector search) ───────────────────────────
    # The AR-es model id in the taxonomy platform. Its embeddings must exist in the
    # taxonomy Atlas cluster or v2 linking returns nothing (runbook phase 6).
    "taxonomy_model_id": lang_env_str("TAXONOMY_MODEL_ID", "es", ""),
    "taxonomy_locale": "AR-es",
}
