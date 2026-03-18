"""NER model: extracts entity spans from job-related text."""

import os
from typing import List

import torch
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForTokenClassification, AutoTokenizer

from shared.bio_utils import extract_entities, fix_bio_tags, remove_special_tokens_and_tags
from shared.transformers_crf import AutoModelCrfForNer

HF_TOKEN = os.getenv("HF_TOKEN")


class NERModel:
    """Extracts entity spans from job-related text using a fine-tuned transformer."""

    def __init__(
        self,
        model_name: str = "tabiya/roberta-base-job-ner",
        crf: bool = False,
    ):
        self.model_name = model_name
        self.crf = crf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.crf:
            self.model = AutoModelCrfForNer.from_pretrained(model_name)
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name, token=HF_TOKEN
            )

        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)

    def extract(self, text: str) -> List[dict]:
        """Extract entities from text, returning entity_type, surface_form, and span."""
        text = text.replace("\n", " ")
        sentences = sent_tokenize(text)
        all_entities: List[dict] = []
        char_offset = 0

        for sentence in sentences:
            sent_start = text.find(sentence, char_offset)
            raw_entities = self._ner_pipeline(sentence)

            for entity in raw_entities:
                surface = entity["tokens"]
                entity_start = text.find(surface, sent_start)
                entity_end = entity_start + len(surface) if entity_start != -1 else sent_start

                all_entities.append(
                    {
                        "entity_type": entity["type"].lower(),
                        "surface_form": surface,
                        "span": {
                            "start": max(entity_start, 0),
                            "end": max(entity_end, 0),
                        },
                    }
                )

            char_offset = sent_start + len(sentence)

        return all_entities

    def _ner_pipeline(self, text: str) -> List[dict]:
        """Run NER on a single sentence."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)

        if self.crf:
            with torch.no_grad():
                logits = self.model(**inputs)
            predictions = logits[1][0]
        else:
            with torch.no_grad():
                logits = self.model(**inputs).logits
            predictions = torch.argmax(logits, dim=2)

        predicted_tags = [self.model.config.id2label[t.item()] for t in predictions[0]]
        predicted_tags = fix_bio_tags(predicted_tags)
        input_ids, predicted_tags = remove_special_tokens_and_tags(
            inputs["input_ids"][0], predicted_tags, self.tokenizer
        )
        result = extract_entities(input_ids, predicted_tags)

        for entry in result:
            sentence = self.tokenizer.decode(entry["tokens"])
            if sentence.startswith(" "):
                sentence = sentence[1:]
            entry["tokens"] = sentence

        return result
