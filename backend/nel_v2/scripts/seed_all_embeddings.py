#!/usr/bin/env python3
"""Prime the NEL embeddings cache for EVERY (nel_model, taxonomy_model) pair.

WHY
---
NEL needs a precomputed embeddings cache in MongoDB Atlas for each
(taxonomy_model_id, nel_model_id) combination a user might select. If a pair
isn't primed, switching to it makes NEL return 503 (cache not ready). This
driver primes them all so a public user can change their configuration to any
supported model/taxonomy without breaking NEL.

WHAT IT DOES
------------
  * Enumerates the supported NEL models (from seed_nel_models._MODELS).
  * Fetches the live taxonomy-model list (GET {TAXONOMY_API_BASE_URL}/api/app/models).
  * For each (nel_model × taxonomy_model) pair, runs the existing
    generate_embeddings.main() flow (streams items, embeds, inserts, marks
    cache ready, upserts Atlas vector indexes).
  * DRY-RUN by default — logs the full plan and what each pair would do, writes
    nothing. Pass --hot-run to actually generate + write.
  * Idempotent: generate_embeddings skips a pair/entity that is already
    "ready" unless --force is set, so re-runs only fill gaps.

COST WARNING
------------
Two NEL models are paid Google Vertex AI (text-embedding-005,
models/gemini-embedding-001). A full hot-run embeds thousands of items across
every taxonomy for those models — real cost + hours of runtime. Use
--sentence-transformer-only to skip the paid Vertex models, or --nel-models /
--taxonomy-ids to scope. Requires GCP ADC for the Vertex models.

REQUIRED ENV VARS (same as generate_embeddings.py)
--------------------------------------------------
  APPLICATION_MONGODB_URI, TAXONOMY_MONGODB_URI, TAXONOMY_API_KEY

EXAMPLES
--------
  # Dry-run the whole matrix (safe — no writes, shows the plan):
  python scripts/seed_all_embeddings.py

  # Free SentenceTransformer models only, all taxonomies, for real:
  python scripts/seed_all_embeddings.py --sentence-transformer-only --hot-run

  # The full matrix, for real (paid Vertex included):
  python scripts/seed_all_embeddings.py --hot-run

  # Scope to specific models / taxonomies:
  python scripts/seed_all_embeddings.py \\
      --nel-models all-MiniLM-L6-v2 text-embedding-005 \\
      --taxonomy-ids 68933862382aab4c7de13ec6 \\
      --hot-run
"""

import argparse
import asyncio
import logging
import logging.config
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

import httpx
import yaml

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_HERE.parent))  # nel_v2/ → `nel` importable
sys.path.insert(0, str(_HERE))         # scripts/ → sibling scripts importable

from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env")

_cfg_path = _HERE / "logging.cfg.yaml"
with open(_cfg_path) as _f:
    logging.config.dictConfig(yaml.safe_load(_f))

logger = logging.getLogger(__name__)

# Reuse the single-pair flow and the canonical model list — no duplication.
import generate_embeddings as gen  # noqa: E402
from seed_nel_models import _MODELS as _NEL_MODELS  # noqa: E402

# Vertex AI models are paid; everything else is a local SentenceTransformer.
_VERTEX_MODEL_IDS = {"text-embedding-005", "models/gemini-embedding-001"}


def _fetch_taxonomy_models() -> list[dict]:
    base = os.environ.get("TAXONOMY_API_BASE_URL", "https://taxonomy.tabiya.tech").rstrip("/")
    api_key = os.environ.get("TAXONOMY_API_KEY", "")
    headers = {"X-API-Key": api_key} if api_key else {}
    resp = httpx.get(f"{base}/api/app/models", headers=headers, timeout=30.0)
    resp.raise_for_status()
    return resp.json()


def _select_nel_models(args: argparse.Namespace) -> list[str]:
    all_ids = [m["model_id"] for m in _NEL_MODELS]
    if args.nel_models:
        unknown = [m for m in args.nel_models if m not in all_ids]
        if unknown:
            raise SystemExit(f"Unknown --nel-models: {unknown}. Supported: {all_ids}")
        selected = list(args.nel_models)
    else:
        selected = list(all_ids)
    if args.sentence_transformer_only:
        selected = [m for m in selected if m not in _VERTEX_MODEL_IDS]
    return selected


def _select_taxonomies(args: argparse.Namespace, taxonomy_models: list[dict]) -> list[dict]:
    selected = taxonomy_models
    if args.released_only:
        selected = [t for t in selected if t.get("released")]
    if args.taxonomy_ids:
        wanted = set(args.taxonomy_ids)
        selected = [t for t in selected if t["id"] in wanted]
        missing = wanted - {t["id"] for t in selected}
        if missing:
            raise SystemExit(f"Unknown --taxonomy-ids: {sorted(missing)}")
    return selected


def _pair_args(taxonomy_id: str, nel_model_id: str, args: argparse.Namespace) -> Namespace:
    """Build the Namespace that generate_embeddings.main() expects for one pair."""
    return Namespace(
        taxonomy_model_id=taxonomy_id,
        nel_model_id=nel_model_id,
        entity_types=args.entity_types,
        force=args.force,
        start_cursor=None,
        start_cursor_entity_type=None,
        indexes_only=False,
        hot_run=args.hot_run,
    )


async def _run(args: argparse.Namespace) -> None:
    nel_model_ids = _select_nel_models(args)
    taxonomy_models = _select_taxonomies(args, _fetch_taxonomy_models())

    pairs = [(t, n) for n in nel_model_ids for t in taxonomy_models]
    vertex_selected = [n for n in nel_model_ids if n in _VERTEX_MODEL_IDS]

    mode = "HOT-RUN (writes enabled)" if args.hot_run else "DRY-RUN (no writes)"
    logger.info("=" * 68)
    logger.info("Seed ALL embeddings — %s", mode)
    logger.info("  NEL models        : %d  %s", len(nel_model_ids), nel_model_ids)
    logger.info("  Taxonomy models   : %d", len(taxonomy_models))
    logger.info("  Pairs to process  : %d", len(pairs))
    logger.info("  Entity types      : %s", args.entity_types)
    logger.info("  Force re-generate : %s", args.force)
    if vertex_selected:
        logger.info("  ⚠ PAID Vertex AI models selected: %s", vertex_selected)
        logger.info("    (a hot-run embeds every taxonomy with these — real cost)")
    logger.info("=" * 68)

    if not args.hot_run:
        logger.info("Plan (dry-run) — each pair below will be dry-run through the generator:")
        for t, n in pairs:
            logger.info("  • nel=%-45s taxonomy=%s (%s)", n, t["id"], t.get("name", ""))

    succeeded: list[tuple[str, str]] = []
    failed: list[tuple[str, str, str]] = []
    t_start = time.monotonic()

    for index, (taxonomy, nel_model_id) in enumerate(pairs, start=1):
        taxonomy_id = taxonomy["id"]
        logger.info("")
        logger.info(
            "── pair %d/%d: nel=%s taxonomy=%s (%s) ──",
            index, len(pairs), nel_model_id, taxonomy_id, taxonomy.get("name", ""),
        )
        try:
            await gen.main(_pair_args(taxonomy_id, nel_model_id, args))
            succeeded.append((nel_model_id, taxonomy_id))
        except Exception as exc:  # noqa: BLE001 — keep going; report at the end
            logger.error("Pair FAILED (nel=%s taxonomy=%s): %s", nel_model_id, taxonomy_id, exc, exc_info=True)
            failed.append((nel_model_id, taxonomy_id, str(exc)))
            if not args.keep_going:
                logger.error("Stopping (pass --keep-going to continue past failures).")
                break

    elapsed = time.monotonic() - t_start
    logger.info("")
    logger.info("=" * 68)
    logger.info("Summary — %s", mode)
    logger.info("  Pairs succeeded : %d/%d", len(succeeded), len(pairs))
    if failed:
        logger.info("  Pairs FAILED    : %d", len(failed))
        for nel_model_id, taxonomy_id, err in failed:
            logger.info("    ✗ nel=%s taxonomy=%s — %s", nel_model_id, taxonomy_id, err[:120])
    logger.info("  Elapsed         : %.1fs", elapsed)
    if not args.hot_run:
        logger.info("  DRY-RUN — nothing was written. Re-run with --hot-run to apply.")
    logger.info("=" * 68)

    if failed:
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--nel-models",
        nargs="+",
        metavar="MODEL",
        help="Restrict to these NEL model ids (default: all supported).",
    )
    parser.add_argument(
        "--taxonomy-ids",
        nargs="+",
        metavar="UUID",
        help="Restrict to these taxonomy model ids (default: all from the taxonomy API).",
    )
    parser.add_argument(
        "--sentence-transformer-only",
        action="store_true",
        help="Skip the paid Vertex AI models (text-embedding-005, gemini-embedding-001).",
    )
    parser.add_argument(
        "--released-only",
        action="store_true",
        help="Only taxonomy models flagged released=true (excludes -rc versions).",
    )
    parser.add_argument(
        "--entity-types",
        nargs="+",
        choices=["occupation", "skill", "qualification"],
        default=["occupation", "skill", "qualification"],
        metavar="TYPE",
        help="Entity types to generate per pair (default: all three).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even pairs already marked 'ready' (deletes + rebuilds).",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue to the next pair if one fails (default: stop on first failure).",
    )
    parser.add_argument(
        "--hot-run",
        action="store_true",
        help="Perform actual writes. Without this, runs a dry-run of every pair.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(_run(parse_args()))
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
