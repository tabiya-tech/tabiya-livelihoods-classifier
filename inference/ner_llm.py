"""
LLM-based NER via Gemini (Vertex-backed) using google-genai and Pydantic schemas.
Returns the same format as the RoBERTa NER pipeline:
List[dict] with keys "type" (Occupation|Skill|Qualification) and "tokens" (span text).
"""
import os
from typing import List, Literal

from pydantic import BaseModel, Field

from google import genai
from google.genai.types import HttpOptions

ENTITY_TYPES = ("Occupation", "Skill", "Qualification")

SYSTEM_PROMPT = """System Message: You are a helpful information extraction system that works on job descriptions.

Prompt: Given a single sentence from a job posting, extract all entities and identify their entity types.

You must follow these rules:
- Only consider entities of the following types (no others):
  - Occupation
  - Skill
  - Qualification
- Use only spans that appear verbatim in the input sentence (no paraphrasing or normalization).
- Each entity span is contiguous in the token sequence.
- Do not invent entities that are not explicitly mentioned.

Output format:
- Return the result as a JSON object, and nothing else.
- The JSON object must have a single key \"entities\" whose value is a list of objects.
- Each entity object must have exactly two keys:
  - \"text\": the exact entity span string from the sentence
  - \"type\": the entity type, exactly one of \"Occupation\", \"Skill\", \"Qualification\"
- If there are no entities, return: {\"entities\": []}

Example:
Sentence: We are looking for a Head Chef who can plan menus.
Output:
{
  \"entities\": [
    {\"text\": \"Head Chef\", \"type\": \"Occupation\"},
    {\"text\": \"plan menus\", \"type\": \"Skill\"}
  ]
}

Now read the next sentence and output the JSON object as specified."""


class NerEntity(BaseModel):
    text: str = Field(description="Exact entity span from the sentence.")
    type: Literal["Occupation", "Skill", "Qualification"] = Field(
        description="Entity type."
    )


class NerResponse(BaseModel):
    entities: List[NerEntity] = Field(
        description="All entities extracted from the sentence."
    )


def _get_genai_client():
    """
    Build a google-genai client configured for Vertex-backed Gemini.
    Requires env like:
      - GOOGLE_CLOUD_PROJECT
      - GOOGLE_CLOUD_LOCATION
      - GOOGLE_GENAI_USE_VERTEXAI=True
    """
    client = genai.Client(http_options=HttpOptions(api_version="v1"))
    model_name = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    return client, model_name


def extract_entities_llm(sentence: str, client=None, model_name: str = None) -> List[dict]:
    """
    Run LLM-based NER on a single sentence. Returns list of {"type": str, "tokens": str}.
    """
    if client is None:
        client, default_model = _get_genai_client()
        model_name = model_name or default_model
    else:
        model_name = model_name or os.getenv("LLM_MODEL", "gemini-2.5-flash")

    prompt = f"{SYSTEM_PROMPT}\nSentence: {sentence}"
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": NerResponse.model_json_schema(),
            },
        )
        ner = NerResponse.model_validate_json(response.text)
    except Exception:
        return []

    return [{"type": ent.type, "tokens": ent.text} for ent in ner.entities]


class VertexNERClient:
    """Reusable client for Vertex-backed Gemini NER (holds google-genai client)."""

    def __init__(self, model_name: str = None):
        self._client, default_model = _get_genai_client()
        self._model_name = model_name or default_model

    def extract(self, sentence: str) -> List[dict]:
        return extract_entities_llm(sentence, client=self._client, model_name=self._model_name)
