"""Read the Peru job-ads corpus (Cardenas et al., CC BY 4.0) and convert its tags to ours.

Source: https://github.com/ronaldahmed/labor-market-demand-analysis (``annotated data.zip``).
800 ads in two folders: ``random`` (a random sample) and ``careers tagged`` (ads asking
for a degree). Each ad file has one word per line (word, part of speech, tag) and a blank
line between sentences.
"""

import io
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal

from tqdm.auto import tqdm

from ner_experiments.corpus.qualification_levels import is_qualification_phrase
from ner_experiments.schemas import TaggedSentence
from ner_experiments.tags import EntityType, extract_spans, repair_bio

AD_FOLDERS = ("random", "careers tagged")
_PERU_TAG_TO_OURS = {
    "JOB": EntityType.OCCUPATION.value,
    "REQ": EntityType.SKILL.value,
    "CARR": EntityType.QUALIFICATION.value,
}


@dataclass
class MovedPhrase:
    """A REQ phrase moved from Skill to Qualification, kept for human review."""

    ad_id: str
    sentence_index: int
    phrase: str


@dataclass
class PeruAd:
    ad_id: str
    folder: str
    sentences: list[TaggedSentence]


@dataclass
class ConversionReport:
    moved_to_qualification: list[MovedPhrase] = field(default_factory=list)


def download_corpus(corpus_url: str, raw_dir: Path) -> Path:
    """Download the corpus zip once and reuse the cached copy afterwards."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "peru_annotated_ads.zip"
    if zip_path.exists():
        print(f"Using cached corpus at {zip_path}")
        return zip_path
    print(f"Downloading the Peru corpus from {corpus_url}")
    with urllib.request.urlopen(corpus_url) as response:
        zip_path.write_bytes(response.read())
    return zip_path


def _convert_peru_tag(peru_tag: str, task_description_spans: Literal["ignore", "skill"]) -> str:
    """'B-REQ' -> 'B-Skill', 'I-JOB' -> 'I-Occupation', anything unmapped -> 'O'."""
    if len(peru_tag) < 3 or peru_tag[1] != "-" or peru_tag[0] not in "BI":
        return "O"
    prefix, peru_type = peru_tag[0], peru_tag[2:]
    if peru_type == "FUN" and task_description_spans == "skill":
        return f"{prefix}-{EntityType.SKILL.value}"
    our_type = _PERU_TAG_TO_OURS.get(peru_type)
    return f"{prefix}-{our_type}" if our_type else "O"


def _move_qualification_levels(
    sentence: TaggedSentence, peru_tags: list[str], report: ConversionReport
) -> TaggedSentence:
    """Retag REQ phrases that name a level of education as Qualification."""
    new_tags = list(sentence.tags)
    for entity_type, start_word, end_word in extract_spans(sentence.tags):
        came_from_req = all(peru_tag.endswith("-REQ") for peru_tag in peru_tags[start_word:end_word])
        if entity_type != EntityType.SKILL.value or not came_from_req:
            continue
        phrase = " ".join(sentence.tokens[start_word:end_word])
        if not is_qualification_phrase(phrase):
            continue
        new_tags[start_word] = f"B-{EntityType.QUALIFICATION.value}"
        for word_index in range(start_word + 1, end_word):
            new_tags[word_index] = f"I-{EntityType.QUALIFICATION.value}"
        report.moved_to_qualification.append(MovedPhrase(sentence.ad_id, sentence.sentence_index, phrase))
    return sentence.model_copy(update={"tags": new_tags})


def _read_ad(
    ad_id: str,
    ad_text: str,
    task_description_spans: Literal["ignore", "skill"],
    move_qualification_levels: bool,
    report: ConversionReport,
) -> list[TaggedSentence]:
    sentences: list[TaggedSentence] = []
    words: list[str] = []
    peru_tags: list[str] = []

    def close_sentence() -> None:
        if not words:
            return
        our_tags = repair_bio([_convert_peru_tag(peru_tag, task_description_spans) for peru_tag in peru_tags])
        sentence = TaggedSentence(ad_id=ad_id, sentence_index=len(sentences), tokens=list(words), tags=our_tags)
        if move_qualification_levels:
            sentence = _move_qualification_levels(sentence, list(peru_tags), report)
        sentences.append(sentence)
        words.clear()
        peru_tags.clear()

    for line in ad_text.splitlines():
        if not line.strip():
            close_sentence()
            continue
        columns = line.split("\t")
        words.append(columns[0])
        peru_tags.append(columns[-1].strip() if len(columns) >= 3 else "O")
    close_sentence()
    return sentences


def read_corpus(
    zip_path: Path,
    task_description_spans: Literal["ignore", "skill"],
    move_qualification_levels: bool,
) -> tuple[list[PeruAd], ConversionReport]:
    """Every ad in the corpus, converted to our tags, in a stable order."""
    report = ConversionReport()
    ads: list[PeruAd] = []
    with zipfile.ZipFile(io.BytesIO(zip_path.read_bytes())) as corpus_zip:
        ad_file_names = sorted(
            name
            for name in corpus_zip.namelist()
            if PurePosixPath(name).suffix == ".tsv" and PurePosixPath(name).parent.name in AD_FOLDERS
        )
        for file_name in tqdm(ad_file_names, desc="Reading and converting ads", unit="ad"):
            file_path = PurePosixPath(file_name)
            ad_id = f"{file_path.parent.name}/{file_path.stem}"
            ad_text = corpus_zip.read(file_name).decode("utf-8", errors="replace")
            sentences = _read_ad(ad_id, ad_text, task_description_spans, move_qualification_levels, report)
            if sentences:
                ads.append(PeruAd(ad_id=ad_id, folder=file_path.parent.name, sentences=sentences))
    return ads, report
