#!/usr/bin/env python3
"""Build a NEL v1 data pack (``nel/nel/files/<language>/``) from a taxonomy CSV export.

The v1 linker embeds label text from three CSVs. Those CSVs were committed by hand for
English; this script is how every language pack — including a refreshed English one — is
produced, so a contributor adding a language never has to reverse-engineer the format.

Input: a Tabiya taxonomy CSV export directory (what the taxonomy platform hands you, and
what ``empujar/empujar-taxonomy/`` is), containing at least:

    occupations.csv   ID, ORIGINURI, UUIDHISTORY, CODE, OCCUPATIONTYPE, PREFERREDLABEL, ALTLABELS
    skills.csv        ID, ORIGINURI, UUIDHISTORY, PREFERREDLABEL, ALTLABELS
    model_info.csv    LOCALE, NAME, VERSION            (optional — used to sanity-check the locale)

Output, written to ``nel/nel/files/<language>/``:

    occupations_augmented.csv   occupation, preffered_label, esco_code, uuid   (one row per label)
    skills.csv                  skills, uuid                                  (one row per skill)
    qualifications.csv          copied from the fallback pack (see --qualifications-from)

``uuid`` is ``UUIDHISTORY[-1]``, the oldest entry in the entity's UUID history. That
value is **identical across taxonomy locales** (verified: 13,896/13,896 skills and
3,007/3,007 ESCO occupations between the English model and AR-es), which is what lets a
Spanish-linked entity resolve to the same taxonomy entry in the reranker and the matching
service. Do not switch this column to ``ID``: those are per-model and share nothing
across locales.

Only ``OCCUPATIONTYPE=escooccupation`` rows are emitted, matching the committed English
pack. ``localoccupation`` rows have no ESCO code and differ per locale.

Usage:

    # Spanish (AR-es) pack from the Empujar taxonomy export
    python scripts/build_nel_files.py \
        --taxonomy-dir ../../../empujar/empujar-taxonomy --language es

    # See what would be written, and how it compares to what is committed
    python scripts/build_nel_files.py --taxonomy-dir … --language es --dry-run

Note on regenerating ``en``: the committed English pack was built from an earlier
taxonomy export, so a fresh run produces a slightly different file (newer labels). That
is expected — regenerate it deliberately, not as a side effect of adding a language.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

# Allow running as a plain script (no package install) from backend/nel/.
_NEL_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_ROOT = _NEL_ROOT.parent
for _p in (str(_NEL_ROOT), str(_BACKEND_ROOT / "shared")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.languages import (  # noqa: E402
    LANGUAGE_REGISTRY,
    get_language_config,
    normalise_language,
)

FILES_DIR = _NEL_ROOT / "nel" / "files"

OCCUPATIONS_OUT = "occupations_augmented.csv"
SKILLS_OUT = "skills.csv"
QUALIFICATIONS_OUT = "qualifications.csv"

# csv.field_size_limit default is too small for the DESCRIPTION column of some rows.
csv.field_size_limit(10_000_000)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _uuid(row: dict) -> str:
    """``UUIDHISTORY[-1]`` — the locale-stable identity of a taxonomy entity."""
    history = [u.strip() for u in (row.get("UUIDHISTORY") or "").split("\n") if u.strip()]
    return history[-1] if history else ""


def _labels(row: dict) -> list[str]:
    """Preferred label first, then alt labels, de-duplicated, order preserved."""
    out: list[str] = []
    seen: set[str] = set()
    for label in [row.get("PREFERREDLABEL") or ""] + (row.get("ALTLABELS") or "").split("\n"):
        text = label.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def build_occupation_rows(occupations: list[dict]) -> list[dict]:
    """One row per (label, occupation) pair for ESCO occupations."""
    rows: list[dict] = []
    skipped_no_uuid = 0
    for row in occupations:
        if (row.get("OCCUPATIONTYPE") or "").strip().lower() != "escooccupation":
            continue
        uuid = _uuid(row)
        if not uuid:
            skipped_no_uuid += 1
            continue
        preferred = (row.get("PREFERREDLABEL") or "").strip()
        code = (row.get("CODE") or "").strip()
        for label in _labels(row):
            rows.append(
                {
                    "occupation": label,
                    "preffered_label": preferred,  # column name is misspelled in the schema
                    "esco_code": code,
                    "uuid": uuid,
                }
            )
    if skipped_no_uuid:
        print(f"  ! skipped {skipped_no_uuid} occupation(s) with an empty UUIDHISTORY")
    return rows


def build_skill_rows(skills: list[dict]) -> list[dict]:
    """One row per skill, keyed on its preferred label (matches the English pack)."""
    rows: list[dict] = []
    skipped = 0
    for row in skills:
        uuid = _uuid(row)
        label = (row.get("PREFERREDLABEL") or "").strip()
        if not uuid or not label:
            skipped += 1
            continue
        rows.append({"skills": label, "uuid": uuid})
    if skipped:
        print(f"  ! skipped {skipped} skill(s) with an empty label or UUIDHISTORY")
    return rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _check_locale(taxonomy_dir: Path, language: str) -> None:
    """Warn when the export's LOCALE is not the one the language config declares."""
    info_path = taxonomy_dir / "model_info.csv"
    if not info_path.is_file():
        return
    rows = _read_csv(info_path)
    if not rows:
        return
    locale = (rows[0].get("LOCALE") or "").strip()
    name = (rows[0].get("NAME") or "").strip()
    expected = get_language_config(language).get("taxonomy_locale", "")
    print(f"  taxonomy model: {name or '?'} (locale {locale or '?'})")
    if locale and expected and normalise_language(locale, fallback=language) != language:
        print(
            f"  ! export locale {locale!r} does not resolve to language {language!r} "
            f"(expected something like {expected!r}) — check --language"
        )


def _compare_to_existing(path: Path, rows: list[dict], key: tuple[str, ...]) -> None:
    """Report how the generated file differs from one already on disk."""
    if not path.is_file():
        print(f"  {path.name}: new file ({len(rows)} rows)")
        return
    existing = _read_csv(path)
    if not existing or key[0] not in (existing[0] or {}):
        print(f"  {path.name}: {len(existing)} rows on disk (different schema) → {len(rows)} rows")
        return
    old = {tuple(r.get(k, "") for k in key) for r in existing}
    new = {tuple(r.get(k, "") for k in key) for r in rows}
    print(
        f"  {path.name}: {len(existing)} rows on disk → {len(rows)} generated "
        f"(+{len(new - old)} added, -{len(old - new)} removed)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a NEL data pack (nel/nel/files/<language>/) from a taxonomy CSV export.",
    )
    parser.add_argument(
        "--taxonomy-dir",
        required=True,
        type=Path,
        help="Taxonomy CSV export directory (occupations.csv, skills.csv, model_info.csv).",
    )
    parser.add_argument(
        "--language",
        required=True,
        help=f"Target language code. Registered: {', '.join(LANGUAGE_REGISTRY)}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: nel/nel/files/<language>/).",
    )
    parser.add_argument(
        "--qualifications-from",
        type=Path,
        default=None,
        help=(
            "Pack directory to copy qualifications.csv from (default: nel/nel/files/en/). "
            "Taxonomy exports carry no qualifications; the ESCO qualification list is "
            "language-independent in practice (country + EQF level)."
        ),
    )
    parser.add_argument(
        "--skip-qualifications",
        action="store_true",
        help="Do not write qualifications.csv (the linker falls back to the en pack at runtime).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be written without writing anything.",
    )
    args = parser.parse_args()

    language = normalise_language(args.language, fallback=args.language)
    if language not in LANGUAGE_REGISTRY:
        parser.error(
            f"unregistered language {args.language!r}. Add it to "
            f"shared/languages/__init__.py:LANGUAGE_REGISTRY and create "
            f"shared/languages/{language}_config.py first."
        )

    taxonomy_dir: Path = args.taxonomy_dir.expanduser().resolve()
    if not taxonomy_dir.is_dir():
        parser.error(f"--taxonomy-dir does not exist: {taxonomy_dir}")
    for required in ("occupations.csv", "skills.csv"):
        if not (taxonomy_dir / required).is_file():
            parser.error(f"missing {required} in {taxonomy_dir}")

    subdir = get_language_config(language).get("nel_files_subdir") or language
    out_dir: Path = (args.out_dir or FILES_DIR / subdir).expanduser().resolve()

    print(f"Building NEL data pack for {language!r} from {taxonomy_dir}")
    _check_locale(taxonomy_dir, language)

    occupations = _read_csv(taxonomy_dir / "occupations.csv")
    skills = _read_csv(taxonomy_dir / "skills.csv")
    occ_rows = build_occupation_rows(occupations)
    skill_rows = build_skill_rows(skills)

    print(
        f"  {len(occupations)} occupation(s) in export → {len(occ_rows)} label row(s) "
        f"({len({r['uuid'] for r in occ_rows})} ESCO occupations)"
    )
    print(f"  {len(skills)} skill(s) in export → {len(skill_rows)} row(s)")

    _compare_to_existing(out_dir / OCCUPATIONS_OUT, occ_rows, ("occupation", "uuid"))
    _compare_to_existing(out_dir / SKILLS_OUT, skill_rows, ("skills", "uuid"))

    if args.dry_run:
        print(f"Dry run — nothing written to {out_dir}")
        return 0

    _write_csv(out_dir / OCCUPATIONS_OUT, ["occupation", "preffered_label", "esco_code", "uuid"], occ_rows)
    _write_csv(out_dir / SKILLS_OUT, ["skills", "uuid"], skill_rows)

    if not args.skip_qualifications:
        source_dir = (args.qualifications_from or FILES_DIR / "en").expanduser().resolve()
        source = source_dir / QUALIFICATIONS_OUT
        target = out_dir / QUALIFICATIONS_OUT
        if source.is_file() and source != target:
            shutil.copyfile(source, target)
            print(f"  {QUALIFICATIONS_OUT}: copied from {source_dir.name}/")
        elif not source.is_file():
            print(f"  ! {QUALIFICATIONS_OUT} not found in {source_dir} — the linker will fall back to the en pack")

    print(f"Wrote {out_dir}")
    print(
        "Embedding caches (<model>/*.pkl) are not built here — the linker computes and "
        "caches them on first start, or run: python -m scripts.build_nel_embeddings "
        f"--language {language}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
