"""
Local NER backend using a Gemma-style HF model as a control/baseline.

This is prompt-based: for each sentence, we ask the model to output a JSON
object of the form:

{
  "entities": [
    {"text": "...", "type": "Occupation" | "Skill" | "Qualification"},
    ...
  ]
}

and then map that into the same format used by the rest of the pipeline:
[{"type": "...", "tokens": "..."}, ...].
"""

import json
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ENTITY_TYPES = ("Occupation", "Skill", "Qualification")

PROMPT_TEMPLATE = """System Message: You are a helpful information extraction system that works on job descriptions.

Prompt: Given the following sentence from a job posting, extract all entities and identify their entity types.

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
- The JSON object must have a single key "entities" whose value is a list of objects.
- Each entity object must have exactly two keys:
  - "text": the exact entity span string from the sentence
  - "type": the entity type, exactly one of "Occupation", "Skill", "Qualification"
- If there are no entities, return: {{ "entities": [] }}

Sentence: {sentence}
Output:
"""


class GemmaNERClient:
    """
    Prompt-based NER using a local Gemma-style HF causal LM.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b-it",
        max_new_tokens: int = 128,
    ):
        self.model_name = model_name
        self.use_cuda = torch.cuda.is_available()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.use_cuda:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            self.device = next(self.model.parameters()).device
        else:
            self.device = torch.device("cpu")
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)
        self.max_new_tokens = max_new_tokens

    def extract(self, sentence: str) -> List[dict]:
        """
        Run NER on a single sentence. Returns list of {"type": str, "tokens": str}.
        """
        prompt = PROMPT_TEMPLATE.format(sentence=sentence)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )[0]

        gen_text = self.tokenizer.decode(
            output_ids[input_len:], skip_special_tokens=True
        ).strip()

        # Try to parse a JSON object with key "entities"
        try:
            # If the model wrapped it in extra text, try to find the first '{'...' }'
            text = gen_text
            if not text.strip().startswith("{"):
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    text = text[start : end + 1]
            data = json.loads(text)
        except Exception:
            return []

        entities = data.get("entities") or []
        out: List[dict] = []
        for ent in entities:
            if not isinstance(ent, dict):
                continue
            text = ent.get("text")
            etype = ent.get("type")
            if not text or not isinstance(text, str):
                continue
            if etype not in ENTITY_TYPES:
                continue
            span = text.strip()
            if not span:
                continue
            out.append({"type": etype, "tokens": span})
        return out

