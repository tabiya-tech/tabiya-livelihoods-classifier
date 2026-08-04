"""Env helpers for the language config modules.

Override convention, so one image can serve every language:

    <SETTING>_<LANG>   per-language override, e.g. NER_MODEL_ES, LINKER_MODEL_ES
    <SETTING>          applies to the default language only (pre-existing behaviour)
"""

from __future__ import annotations

import os


def env_str(key: str, default: str | None = None) -> str | None:
    """Trimmed env value, or ``default`` when unset or empty."""
    v = os.getenv(key)
    return v.strip() if v and v.strip() else default


def lang_env_str(setting: str, language: str, default: str | None = None) -> str | None:
    """``<SETTING>_<LANG>``, falling back to bare ``<SETTING>``, then ``default``.

    The bare variable is honoured for every language on purpose: a deployment that
    only ever runs one language (today's Kenya/Zambia jobs) keeps working with the
    variable names it already sets.
    """
    return env_str(f"{setting}_{language.upper().replace('-', '_')}", env_str(setting, default))
