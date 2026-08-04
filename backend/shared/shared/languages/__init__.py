"""Language registry — the one place to edit when adding a language.

Every service in this monorepo (``ner``, ``nel``, ``nel_v2``, ``classify``) is a single
image that serves **all** registered languages. Which language a given request uses is an
*input*, not a deploy-time property:

* HTTP callers send ``language`` on the request body (``/v1/ner``, ``/v1/nel``,
  ``/v1/classify``). This is what the ``job-pipeline`` GCP Workflow sets per country.
* When the field is omitted, ``default_language()`` (``TARGET_LANGUAGE``, else
  ``LANGUAGE``, else ``en``) decides — so existing callers keep working unchanged.

Each registered language has a ``shared/languages/{lang}_config.py`` exporting
``LANGUAGE_CONFIG``: model names, taxonomy locale, and the name of its data pack
directory. Same shape as ``scraper/config/{country}_config.py``.

Adding a language:

1. Add its code to :data:`LANGUAGE_REGISTRY`.
2. Add ``shared/languages/{code}_config.py`` exporting ``LANGUAGE_CONFIG``.
3. Generate its NEL data pack:
   ``python backend/nel/scripts/build_nel_files.py --taxonomy-dir … --language {code}``
"""

from __future__ import annotations

import importlib
import logging
import os
from functools import lru_cache
from typing import Any

log = logging.getLogger(__name__)

# ── Language registry ─────────────────────────────────────────────────────────
# The ONLY place you need to edit when adding a new language. Each entry must have a
# matching `shared/languages/{code}_config.py` exporting LANGUAGE_CONFIG.
LANGUAGE_REGISTRY: tuple[str, ...] = ("en", "es")

DEFAULT_LANGUAGE = "en"


def _normalise_key(value: str) -> str:
    return str(value).strip().lower().replace("_", "-")


@lru_cache(maxsize=None)
def get_language_config(language: str) -> dict[str, Any]:
    """``LANGUAGE_CONFIG`` for a registered language code.

    :raises ValueError: if the language is not in :data:`LANGUAGE_REGISTRY`.
    """
    code = _normalise_key(language)
    if code not in LANGUAGE_REGISTRY:
        raise ValueError(
            f"Unsupported language {language!r}. Registered: {', '.join(LANGUAGE_REGISTRY)}"
        )
    mod = importlib.import_module(f"shared.languages.{code}_config")
    # Config modules read env at import time. Reloading on a cache miss means
    # `get_language_config.cache_clear()` re-reads the environment, which is what tests
    # and any runtime that mutates env expect. Costs one module exec per language.
    mod = importlib.reload(mod)
    return dict(mod.LANGUAGE_CONFIG)


@lru_cache(maxsize=None)
def _locale_index() -> dict[str, str]:
    """Map every alias/locale declared by a language config to its language code."""
    index: dict[str, str] = {}
    for code in LANGUAGE_REGISTRY:
        cfg = get_language_config(code)
        index[code] = code
        for alias in cfg.get("locales", ()):
            index[_normalise_key(alias)] = code
    return index


def normalise_language(value: str | None, *, fallback: str | None = None) -> str:
    """Resolve any locale spelling to a registered language code.

    Accepts what the callers in this pipeline actually send: a bare code (``es``), a
    taxonomy locale (``AR-es`` — the ``LOCALE`` column of ``model_info.csv``), a POSIX
    locale (``es_AR.UTF-8``), or an English language name (``spanish``).

    Unknown values fall back to ``fallback`` (default :func:`default_language`) with a
    warning rather than raising: a bad ``language`` on one job should not fail a batch.
    """
    fb = fallback or default_language()
    if value is None:
        return fb
    key = _normalise_key(value)
    if not key:
        return fb
    index = _locale_index()
    if key in index:
        return index[key]
    # `AR-es`, `es-419`, `es_AR.UTF-8` → try the sub-tags, most specific first.
    for part in (key.split(".")[0], *reversed(key.split(".")[0].split("-"))):
        if part in index:
            return index[part]
    log.warning(
        "Unknown language %r; falling back to %r (registered: %s)",
        value,
        fb,
        ", ".join(LANGUAGE_REGISTRY),
    )
    return fb


def default_language() -> str:
    """``TARGET_LANGUAGE``, else ``LANGUAGE``, else ``en``. Read fresh (tests set env)."""
    raw = os.getenv("TARGET_LANGUAGE") or os.getenv("LANGUAGE") or DEFAULT_LANGUAGE
    key = _normalise_key(raw)
    index = _locale_index()
    if key in index:
        return index[key]
    log.warning("TARGET_LANGUAGE=%r is not registered; using %r", raw, DEFAULT_LANGUAGE)
    return DEFAULT_LANGUAGE


def enabled_languages() -> tuple[str, ...]:
    """Languages to warm up at startup — ``ENABLED_LANGUAGES`` (``all`` for every one).

    Defaults to just :func:`default_language`, because each extra language costs a
    model load and its embedding tensors in RAM. Languages outside this list still
    work; they are loaded lazily on the first request that asks for them.
    """
    raw = (os.getenv("ENABLED_LANGUAGES") or "").strip()
    if not raw:
        return (default_language(),)
    if _normalise_key(raw) in ("all", "*"):
        return LANGUAGE_REGISTRY
    codes: list[str] = []
    index = _locale_index()
    for part in raw.split(","):
        key = _normalise_key(part)
        if not key:
            continue
        code = index.get(key)
        if code is None:
            log.warning("ENABLED_LANGUAGES: ignoring unregistered language %r", part)
            continue
        if code not in codes:
            codes.append(code)
    return tuple(codes) or (default_language(),)


def language_setting(language: str, key: str, default: Any = None) -> Any:
    """One value out of a language's ``LANGUAGE_CONFIG``."""
    return get_language_config(language).get(key, default)


__all__ = [
    "DEFAULT_LANGUAGE",
    "LANGUAGE_REGISTRY",
    "default_language",
    "enabled_languages",
    "get_language_config",
    "language_setting",
    "normalise_language",
]
