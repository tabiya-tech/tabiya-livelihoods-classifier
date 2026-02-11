"""
NER via Vertex AI (Gemini). Returns the same format as the RoBERTa NER pipeline:
List[dict] with keys "type" (Occupation|Skill|Qualification|Domain|Experience) and "tokens" (span text).
"""
import json
import os
import re
from typing import List

ENTITY_TYPES = ("Occupation", "Skill", "Qualification")
TYPE_MAP = {t.lower(): t for t in ENTITY_TYPES}

SYSTEM_PROMPT = """You are a named-entity recognition system for job descriptions.

Given ONE sentence, you must extract job-related entities and return them in STRICT JSON.

Rules:
- Only look at the given sentence (no external knowledge).
- Extract spans for these types ONLY (no others):
  - Occupation
  - Skill
  - Qualification
- Each entity is a contiguous span of the original sentence.
- Do NOT paraphrase or normalize: the span text must match the original sentence text exactly.

Output format:
- Return a JSON array of objects, and NOTHING else (no explanations, no comments).
- Each object has exactly two keys: "type" and "text".
- "type" must be exactly one of: "Occupation", "Skill", "Qualification".
- "text" is the exact span string from the sentence.
- If there are no entities, return [].

Example:
Sentence: We are looking for a Head Chef who can plan menus.
Output:
[
  {"type": "Occupation", "text": "Head Chef"},
  {"type": "Skill", "text": "plan menus"}
]

Now process the next sentence."""


def _get_vertex_client():
    import vertexai
    from vertexai.generative_models import GenerativeModel

    project = os.getenv("VERTEX_PROJECT")
    location = os.getenv("VERTEX_API_REGION", "us-west1")
    if not project:
        raise ValueError("VERTEX_PROJECT environment variable is required for LLM NER")
    vertexai.init(project=project, location=location)
    model_name = os.getenv("LLM_MODEL", "gemini-1.5-pro")
    return GenerativeModel(model_name)


def _parse_llm_response(raw: str) -> List[dict]:
    raw = raw.strip()
    # Strip common markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```\s*$", "", raw)
    # First try direct JSON parse
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to locate a JSON array substring
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []
        snippet = raw[start : end + 1]
        try:
            data = json.loads(snippet)
        except json.JSONDecodeError:
            return []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        t_raw = item.get("type") or item.get("Type")
        text = item.get("text") or item.get("text_span") or item.get("span")
        if not t_raw or not text or not isinstance(text, str):
            continue
        t_norm = TYPE_MAP.get(str(t_raw).strip().lower())
        if t_norm and text:
            text = text.strip()
            if text:
                out.append({"type": t_norm, "tokens": text})
    return out


def extract_entities_llm(sentence: str, model=None) -> List[dict]:
    """
    Run LLM-based NER on a single sentence. Returns list of {"type": str, "tokens": str}.
    """
    if model is None:
        model = _get_vertex_client()
    prompt = f"{SYSTEM_PROMPT}\n\nSentence: {sentence}"
    try:
        # Ask Vertex to return pure JSON
        response = model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
            },
        )
        text = getattr(response, "text", None)
        if callable(text):
            text = text()
        if not text and getattr(response, "candidates", None):
            try:
                cand = response.candidates[0]
                if cand.content and cand.content.parts:
                    text = cand.content.parts[0].text
            except Exception:
                text = None
    except Exception:
        return []
    if not text:
        return []
    return _parse_llm_response(text)


class VertexNERClient:
    """Reusable client for Vertex NER (holds model so vertexai is init'd once)."""

    def __init__(self, model_name: str = None, project: str = None, location: str = None):
        import vertexai
        from vertexai.generative_models import GenerativeModel

        self._project = project or os.getenv("VERTEX_PROJECT")
        self._location = location or os.getenv("VERTEX_API_REGION", "us-west1")
        self._model_name = model_name or os.getenv("LLM_MODEL", "gemini-1.5-pro")
        if not self._project:
            raise ValueError("VERTEX_PROJECT is required for LLM NER")
        vertexai.init(project=self._project, location=self._location)
        self._model = GenerativeModel(self._model_name)

    def extract(self, sentence: str) -> List[dict]:
        return extract_entities_llm(sentence, model=self._model)
