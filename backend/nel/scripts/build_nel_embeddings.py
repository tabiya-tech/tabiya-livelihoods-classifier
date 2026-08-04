#!/usr/bin/env python3
"""Pre-compute the NEL embedding caches for one language pack.

The linker does this itself on first start when the cache is missing, but a cold Cloud Run
instance then spends minutes encoding ~45k labels before it can serve. Run this in the
image build (or commit the .pkl files to LFS) so the service starts ready.

Writes ``nel/nel/files/<language>/<similarity-model>/{occupations,skills,qualifications}.pkl``
using the model from the language's ``LANGUAGE_CONFIG`` — override with
``LINKER_MODEL_<LANG>`` or ``--model``.

    python scripts/build_nel_embeddings.py --language es
    python scripts/build_nel_embeddings.py --language en --model all-MiniLM-L6-v2

Requires the ``inference`` extra (``poetry install --extras inference``) and downloads the
sentence-transformer on first use.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_NEL_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_ROOT = _NEL_ROOT.parent
for _p in (str(_NEL_ROOT), str(_BACKEND_ROOT / "shared")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.languages import LANGUAGE_REGISTRY, get_language_config, normalise_language  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-compute NEL embedding caches for a language pack.")
    parser.add_argument(
        "--language",
        required=True,
        help=f"Language code. Registered: {', '.join(LANGUAGE_REGISTRY)}",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Sentence-transformer to encode with (default: the language config's nel_similarity_model).",
    )
    args = parser.parse_args()

    language = normalise_language(args.language, fallback=args.language)
    if language not in LANGUAGE_REGISTRY:
        parser.error(f"unregistered language {args.language!r}")

    cfg = get_language_config(language)
    model = args.model or cfg["nel_similarity_model"]

    # Imported here so --help works without torch installed.
    from nel.linker import NELLinker, resolve_files_path

    files_path = resolve_files_path(language)
    print(f"Encoding {language!r} pack at {files_path} with {model!r} — this takes a few minutes")

    # from_cache=False computes the tensors and pickles them next to the CSVs.
    NELLinker(similarity_model=model, files_path=files_path, from_cache=False)

    print(f"Wrote {Path(files_path) / model}/*.pkl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
