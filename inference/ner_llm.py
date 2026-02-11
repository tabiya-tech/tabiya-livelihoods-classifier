"""
NER via Vertex AI (Gemini). Returns the same format as the RoBERTa NER pipeline:
List[dict] with keys "type" (Occupation|Skill|Qualification|Domain|Experience) and "tokens" (span text).
"""
import json
import os
import re
from typing import List

ENTITY_TYPES = ("Occupation", "Skill", "Qualification", "Domain", "Experience")

SYSTEM_PROMPT = """You are a named-entity recognition system for job descriptions. Extract entities from the given sentence only. Output a JSON array of objects. Each object must have exactly two keys: "type" and "text". "type" must be one of: Occupation, Skill, Qualification, Domain, Experience. "text" is the exact span from the sentence (no paraphrasing). If there are no entities, output []. Output nothing else besides the JSON array."""


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
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```\s*$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        t = item.get("type") or item.get("Type")
        text = item.get("text") or item.get("text_span") or item.get("span")
        if t in ENTITY_TYPES and text and isinstance(text, str):
            text = text.strip()
            if text:
                out.append({"type": t, "tokens": text})
    return out


def extract_entities_llm(sentence: str, model=None) -> List[dict]:
    """
    Run LLM-based NER on a single sentence. Returns list of {"type": str, "tokens": str}.
    """
    if model is None:
        model = _get_vertex_client()
    prompt = f"{SYSTEM_PROMPT}\n\nSentence: {sentence}"
    try:
        response = model.generate_content(prompt)
        if hasattr(response, "text") and callable(getattr(response, "text")):
            text = response.text
        elif response.candidates and response.candidates[0].content.parts:
            text = response.candidates[0].content.parts[0].text
        else:
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
