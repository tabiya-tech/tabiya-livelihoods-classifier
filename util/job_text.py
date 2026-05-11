import hashlib
import os
from typing import Optional


def _classifier_prefer_translation() -> bool:
    return os.getenv("CLASSIFIER_PREFER_TRANSLATION", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _input_text_from_translation(doc: dict) -> Optional[str]:
    """Use ``translation.text_en`` / ``fields_en`` when the job was MT-prepared (e.g. Argentina)."""
    tr = doc.get("translation") if isinstance(doc.get("translation"), dict) else {}
    if not tr:
        return None
    text_en = (tr.get("text_en") or "").strip()
    if text_en:
        return text_en
    fe = tr.get("fields_en") if isinstance(tr.get("fields_en"), dict) else {}
    req_parts: list[str] = []
    for k in ("exclusive_requirement", "desirable_requirement"):
        v = (fe.get(k) or "").strip()
        if v:
            req_parts.append(v)
    requirements_en = " ".join(req_parts).strip()
    if requirements_en:
        return requirements_en
    t_en = (fe.get("title") or "").strip()
    d_en = (fe.get("description") or "").strip()
    if t_en and d_en:
        return f"{t_en}.\n{d_en}"
    if d_en or t_en:
        return d_en or t_en
    return None


def _input_text_spanish_scraper_style(doc: dict) -> Optional[str]:
    """Same ordering as ``run_classifier`` / scraped_jobs: requirements, then title + description."""
    title = (doc.get("title") or "").strip()
    description = (doc.get("description") or "").strip()
    requirements = (doc.get("requirements") or "").strip()
    if requirements:
        return requirements
    if title and description:
        return f"{title}.\n{description}"
    return description or title or None


def build_batch_classifier_input_text(doc: dict) -> Optional[str]:
    """Mongo batch job (``run_classifier.py``): prefer English translation, else Spanish scraper fields."""
    if _classifier_prefer_translation():
        t = _input_text_from_translation(doc)
        if t:
            return t
    return _input_text_spanish_scraper_style(doc)


def build_input_text(
    doc: dict,
    *,
    allow_text_field: bool = False,
) -> Optional[str]:
    """Build classifier input from a job dict.

    When *allow_text_field* is True, a non-empty ``text`` key is used as-is
    (the classify API accepts this).

    When ``translation.text_en`` / ``translation.fields_en`` are present
    (Horizon inline translation), they are preferred unless
    ``CLASSIFIER_PREFER_TRANSLATION=0``.

    Otherwise uses ``title`` + ``description`` (legacy worker / API behaviour).
    """
    if allow_text_field:
        text = doc.get("text")
        if text and isinstance(text, str) and text.strip():
            return text.strip()

    if _classifier_prefer_translation():
        t = _input_text_from_translation(doc)
        if t:
            return t

    title = (doc.get("title") or "").strip()
    description = (doc.get("description") or "").strip()

    if title and description:
        return f"{title}.\n{description}"
    return description or title or None


def compute_hash(text: str) -> str:
    """SHA-256 hex digest of *text*, used for classification dedup."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
