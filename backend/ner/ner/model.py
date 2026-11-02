"""NER model: extracts entity spans from job-related text."""

import logging
import os
import re
from collections import Counter
from typing import List, NamedTuple, Optional, TypedDict

import torch
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForTokenClassification, AutoTokenizer

from ner.models import EntityType
from shared.bio_utils import extract_entities, fix_bio_tags
from shared.transformers_crf import AutoModelCrfForNer

HF_TOKEN = os.getenv("HF_TOKEN")

log = logging.getLogger(__name__)

# The entity types the pipeline speaks: NER's response model, NEL's request model and
# every consumer downstream accept exactly these. A checkpoint may tag more than this
# — `tabiya/roberta-base-job-ner` also emits Experience and Domain — and those spans
# cannot be linked to the taxonomy, so they are dropped rather than returned.
PIPELINE_ENTITY_TYPES = frozenset(t.value for t in EntityType)


# Words as the training corpora split them: runs of letters/digits (keeping inner
# hyphens and apostrophes) and each punctuation mark on its own.
_WORD_PATTERN = re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]")


class Word(NamedTuple):
    """One word of the input and its character offsets in the text it came from."""

    text: str
    start: int
    end: int


class SentenceEntity(TypedDict):
    """An entity found in one sentence, with offsets relative to that sentence."""

    type: str
    surface: str
    start: int
    end: int


def split_words(text: str) -> List[Word]:
    """Split text into words the way the NER training data was split.

    The model is given these words rather than raw text: some tokenizers (e.g. the
    SentencePiece ones of BSC-LT/mRoBERTa) do not mark word boundaries on raw text, so
    the model's per-word tags could not be read back. Character offsets keep every
    entity tied to the exact text it came from.
    """
    return [Word(match.group(), match.start(), match.end()) for match in _WORD_PATTERN.finditer(text)]


def first_piece_word_tags(
    piece_tags: List[str], word_ids: List[Optional[int]], word_count: int
) -> List[str]:
    """One tag per word, read from the word's first piece (the only piece trained).

    Words cut off by truncation have no pieces and are tagged ``O``.
    """
    word_tags = ["O"] * word_count
    tagged_words: set[int] = set()
    for piece_tag, word_index in zip(piece_tags, word_ids):
        if word_index is None or word_index in tagged_words:
            continue
        word_tags[word_index] = piece_tag
        tagged_words.add(word_index)
    return word_tags


def resolve_entity_type(raw_label: str, label_map: Optional[dict] = None) -> Optional[str]:
    """Map a checkpoint's own label onto a pipeline entity type.

    Returns ``None`` when the label has no pipeline equivalent — the caller drops the
    span. Returning it instead would fail ``NERResponse`` validation and turn the whole
    request into a 400, losing the entities that *were* usable.
    """
    label = raw_label.lower()
    mapped = (label_map or {}).get(label, label)
    return mapped if mapped in PIPELINE_ENTITY_TYPES else None


class NERModel:
    """Extracts entity spans from job-related text using a fine-tuned transformer.

    ``sentence_tokenizer_language`` picks the NLTK punkt model used to split the text
    into sentences (abbreviations and clause punctuation differ per language).

    ``label_map`` renames a checkpoint's own entity labels to the pipeline's types
    (occupation / skill / qualification), so a language can be pointed at a model with a
    different label vocabulary through config alone. Labels that still do not name a
    pipeline type after mapping are dropped — see ``resolve_entity_type``.
    """

    def __init__(
        self,
        model_name: str = "tabiya/roberta-base-job-ner",
        crf: bool = False,
        sentence_tokenizer_language: str = "english",
        label_map: Optional[dict] = None,
    ):
        self.model_name = model_name
        self.crf = crf
        self.sentence_tokenizer_language = sentence_tokenizer_language
        self.label_map = {k.lower(): v.lower() for k, v in (label_map or {}).items()}
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
        sentences = self._split_sentences(text)
        all_entities: List[dict] = []
        dropped: Counter = Counter()
        char_offset = 0

        for sentence in sentences:
            sent_start = text.find(sentence, char_offset)
            if sent_start == -1:  # the splitter altered the sentence; offsets stay approximate
                sent_start = char_offset
            raw_entities = self._ner_pipeline(sentence)

            for entity in raw_entities:
                entity_type = resolve_entity_type(entity["type"], self.label_map)
                if entity_type is None:
                    dropped[entity["type"].lower()] += 1
                    continue
                all_entities.append(
                    {
                        "entity_type": entity_type,
                        "surface_form": entity["surface"],
                        "span": {
                            "start": sent_start + entity["start"],
                            "end": sent_start + entity["end"],
                        },
                    }
                )

            char_offset = sent_start + len(sentence)

        if dropped:
            log.debug(
                "Dropped %d span(s) whose label has no pipeline entity type: %s",
                sum(dropped.values()),
                dict(dropped),
            )
        return all_entities

    def _split_sentences(self, text: str) -> List[str]:
        """Sentence-split with this language's punkt model, falling back to English.

        A missing punkt model for a language must not fail the request — the fallback
        splits slightly worse, it does not change the entity vocabulary.
        """
        try:
            return sent_tokenize(text, language=self.sentence_tokenizer_language)
        except LookupError:
            log.warning(
                "No punkt sentence tokenizer for %r; falling back to English",
                self.sentence_tokenizer_language,
            )
            return sent_tokenize(text)

    def _ner_pipeline(self, text: str) -> List[SentenceEntity]:
        """Run NER on a single sentence; entity offsets are relative to ``text``."""
        words = split_words(text)
        if not words:
            return []
        inputs = self.tokenizer(
            [word.text for word in words],
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)

        if self.crf:
            with torch.no_grad():
                logits = self.model(**inputs)
            predictions = logits[1][0]
        else:
            with torch.no_grad():
                logits = self.model(**inputs).logits
            predictions = torch.argmax(logits, dim=2)

        piece_tags = [self.model.config.id2label[prediction.item()] for prediction in predictions[0]]
        word_tags = fix_bio_tags(first_piece_word_tags(piece_tags, inputs.word_ids(0), len(words)))

        entities: List[SentenceEntity] = []
        for entity in extract_entities(words, word_tags):
            start, end = entity["tokens"][0].start, entity["tokens"][-1].end
            entities.append({"type": entity["type"], "surface": text[start:end], "start": start, "end": end})
        return entities
