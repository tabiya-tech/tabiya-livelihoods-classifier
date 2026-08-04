"""English language config — the pipeline's original behaviour, unchanged.

Every default here is what the services hardcoded before multi-language support, so an
existing deployment that sets none of the new variables behaves exactly as it did.
"""

from shared.languages._env import lang_env_str

LANGUAGE_CONFIG = {
    "language": "en",
    "name": "English",
    # Locale spellings that resolve to this language (see `normalise_language`).
    "locales": ("en", "eng", "english", "en-gb", "en-us", "en-ke", "en-zm", "en-za"),
    # ── ner ───────────────────────────────────────────────────────────────────
    # Gated on HuggingFace; needs HF_TOKEN.
    "ner_model": lang_env_str("NER_MODEL", "en", "tabiya/roberta-base-job-ner"),
    "ner_crf": False,
    # NLTK punkt sentence tokeniser language (`nltk.tokenize.sent_tokenize(..., language=)`).
    "sentence_tokenizer_language": "english",
    # ── nel (v1: local CSV + sentence-transformer embeddings) ─────────────────
    "nel_similarity_model": lang_env_str("LINKER_MODEL", "en", "all-MiniLM-L6-v2"),
    # Sub-directory of `nel/nel/files/` holding this language's data pack.
    "nel_files_subdir": "en",
    # ── nel_v2 (taxonomy API + Atlas vector search) ───────────────────────────
    # Taxonomy model in the taxonomy platform. Empty = fall back to the caller's
    # user config / DEFAULT_TAXONOMY_MODEL_ID, i.e. today's behaviour.
    "taxonomy_model_id": lang_env_str("TAXONOMY_MODEL_ID", "en", ""),
    # LOCALE column of the taxonomy export's model_info.csv; reported in responses.
    "taxonomy_locale": "en",
}
