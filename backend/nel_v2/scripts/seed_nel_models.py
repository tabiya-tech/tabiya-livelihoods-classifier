#!/usr/bin/env python3
"""Seed the nel_models collection with supported SentenceTransformer models.

DESCRIPTION
-----------
Upserts the set of supported NEL embedding models into the `nel_models`
MongoDB collection. This collection is read by GET /v2/nel/models.

Safe to re-run — existing records are updated, not duplicated.

By default runs in DRY-RUN mode. Pass --hot-run to write to MongoDB.

REQUIRED ENV VARS
-----------------
  APPLICATION_MONGODB_URI     App DB connection string

OPTIONAL ENV VARS
-----------------
  APPLICATION_DATABASE_NAME   Default: tabiya-classifier

EXAMPLES
--------
  python scripts/seed_nel_models.py
  python scripts/seed_nel_models.py --hot-run
"""

import argparse
import asyncio
import logging
import logging.config
import os
import sys
import time
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env")

_cfg_path = _HERE / "logging.cfg.yaml"
with open(_cfg_path) as _f:
    logging.config.dictConfig(yaml.safe_load(_f))

logger = logging.getLogger(__name__)

from motor.motor_asyncio import AsyncIOMotorClient

_MODELS = [
    {"model_id": "all-MiniLM-L6-v2", "dimensions": 384, "description": "Fast, lightweight — good for English text"},
    {"model_id": "paraphrase-multilingual-mpnet-base-v2", "dimensions": 768, "description": "Multilingual, higher quality"},
    {"model_id": "tabiya/all-MiniLM-L6-v2-occupation-fine-tuned", "dimensions": 384, "description": "Fine-tuned on occupation data"},
    {"model_id": "text-embedding-005", "dimensions": 768, "description": "Google Vertex AI text-embedding-005 (768d)"},
    {"model_id": "models/gemini-embedding-001", "dimensions": 3072, "description": "Google Gemini embedding-001 (3072d)"},
]


async def main(args: argparse.Namespace) -> None:
    hot_run: bool = args.hot_run
    uri = os.environ["APPLICATION_MONGODB_URI"]
    db_name = os.environ.get("APPLICATION_DATABASE_NAME", "tabiya-classifier")

    mode_label = "HOT-RUN (writes enabled)" if hot_run else "DRY-RUN (no writes)"
    logger.info("=" * 60)
    logger.info("NEL Models Seed — %s", mode_label)
    logger.info("  app DB : %s", db_name)
    logger.info("  models : %d", len(_MODELS))
    logger.info("=" * 60)

    if not hot_run:
        for m in _MODELS:
            logger.info("[DRY-RUN] Would upsert: %s (dims=%d)", m["model_id"], m["dimensions"])
        logger.info("Re-run with --hot-run to perform actual writes.")
        return

    client = AsyncIOMotorClient(uri, tlsAllowInvalidCertificates=True)
    col = client.get_database(db_name)["nel_models"]
    t0 = time.monotonic()

    inserted = updated = unchanged = 0
    for m in _MODELS:
        result = await col.update_one(
            {"model_id": {"$eq": m["model_id"]}},
            {"$set": m},
            upsert=True,
        )
        if result.upserted_id:
            inserted += 1
        elif result.modified_count:
            updated += 1
        else:
            unchanged += 1
        logger.info("  upserted: %s", m["model_id"])

    client.close()
    logger.info("Done in %.1fs — inserted: %d, updated: %d, unchanged: %d", time.monotonic() - t0, inserted, updated, unchanged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hot-run", action="store_true", help="Perform actual writes to MongoDB")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(main(parse_args()))
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    except Exception as exc:
        logger.error("Script failed: %s", exc, exc_info=True)
        sys.exit(1)
