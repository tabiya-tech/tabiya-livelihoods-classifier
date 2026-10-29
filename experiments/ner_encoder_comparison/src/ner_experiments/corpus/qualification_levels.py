"""Find Peru ``REQ`` phrases that are really qualifications, not skills.

The Peru annotators tagged levels of education ("Técnico", "titulado", "postgrado",
"maestría") as requirements (``REQ``), which we otherwise map to Skill. This rule moves
those phrases to Qualification. Every moved phrase is written to
``data/processed/moved_to_qualification.csv`` so a person can check the rule's choices.

The phrase must START with a qualification word (optionally after "profesional"), so
"comunicación con equipos técnicos" (communicating with technical teams) stays a skill.
"""

import re

_QUALIFICATION_START = re.compile(
    r"^\s*(profesional(es)?\s+)?("
    r"t[eé]cnic[oa]s?"  # técnico / técnica (as a level of training)
    r"|bachiller(es)?"
    r"|universitari[oa]s?"
    r"|egresad[oa]s?"
    r"|titulad[oa]s?|t[ií]tulos?"
    r"|licenciad[oa]s?|licenciatura"
    r"|post\s*-?\s*grado|posgrado"
    r"|maestr[ií]as?|doctorado|diplomados?"
    r"|especializaci[oó]n(es)?"
    r"|secundaria"
    r"|colegiad[oa]s?"
    r"|carreras?"
    r")\b",
    re.IGNORECASE,
)

# "técnicas de venta" (sales techniques), "técnicas varias" (various techniques) are skills.
# "técnica en enfermería" (nursing technician) is a qualification, so only "en" keeps it.
_TECHNIQUES_PHRASE = re.compile(r"^\s*t[eé]cnicas?\b(?!\s+en\b)", re.IGNORECASE)


def is_qualification_phrase(phrase: str) -> bool:
    """True when a REQ phrase names a level of education rather than a skill."""
    if _TECHNIQUES_PHRASE.match(phrase):
        return False
    return _QUALIFICATION_START.match(phrase) is not None
