#!/usr/bin/env python3
"""Generate and store NEL embeddings for a (taxonomy_model_id, nel_model_id) pair.

DESCRIPTION
-----------
This script fetches occupations and skills from the Taxonomy API and qualifications
from MongoDB, embeds them using a SentenceTransformer model, and inserts the
resulting embedding vectors into MongoDB Atlas for vector search.

By default the script runs in DRY-RUN mode — it connects to all services,
validates config, and logs what *would* happen, but writes nothing to the database.
Pass --hot-run to perform actual writes.

STEPS (per entity type)
-----------------------
  [1] Check current cache status
  [2] Delete existing embeddings     (only when --force is set)
  [3] Stream items from source API / MongoDB
  [4] Embed each page using SentenceTransformer
  [5] Insert embedding documents into Atlas
  [6] Mark cache status as "ready"
  [7] Create/update Atlas vector search indexes (skipped with --no-indexes)

REQUIRED ENV VARS
-----------------
  APPLICATION_MONGODB_URI      App DB connection string (cache status, qualifications)
  TAXONOMY_MONGODB_URI         Atlas DB connection string (embedding storage)
  TAXONOMY_API_KEY             X-API-Key for taxonomy.tabiya.tech

OPTIONAL ENV VARS
-----------------
  APPLICATION_DATABASE_NAME    Default: tabiya-classifier
  TAXONOMY_DATABASE_NAME       Default: tabiya-taxonomy
  TAXONOMY_API_BASE_URL        Default: https://taxonomy.tabiya.tech

EXAMPLES
--------
  # Dry-run (safe, no writes):
  python scripts/generate_embeddings.py \\
      --taxonomy-model-id <uuid> \\
      --nel-model-id all-MiniLM-L6-v2

  # Hot-run — actually write embeddings:
  python scripts/generate_embeddings.py \\
      --taxonomy-model-id <uuid> \\
      --nel-model-id all-MiniLM-L6-v2 \\
      --hot-run

  # Only occupations, force re-generation:
  python scripts/generate_embeddings.py \\
      --taxonomy-model-id <uuid> \\
      --nel-model-id all-MiniLM-L6-v2 \\
      --entity-types occupation \\
      --force \\
      --hot-run

  # Create/update Atlas vector search indexes only (no embedding generation):
  python scripts/generate_embeddings.py \\
      --taxonomy-model-id <uuid> \\
      --nel-model-id all-MiniLM-L6-v2 \\
      --indexes-only \\
      --hot-run
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
from pymongo.operations import SearchIndexModel
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent  # scripts → nel_v2 → backend → repo root
sys.path.insert(0, str(_HERE.parent))   # nel_v2/ → `nel` package importable
sys.path.insert(0, str(_HERE))          # scripts/ → `_logging_handler` importable

from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env")

# ── Logging setup ─────────────────────────────────────────────────────────────
_cfg_path = _HERE / "logging.cfg.yaml"
with open(_cfg_path) as _f:
    logging.config.dictConfig(yaml.safe_load(_f))

logger = logging.getLogger(__name__)

# ── Service imports (after path and logging are set up) ───────────────────────
from motor.motor_asyncio import AsyncIOMotorClient

from nel.app.embedding.service.service import SentenceTransformerEmbeddingService
from nel.app.embeddings_cache.repository.repository import EmbeddingsCacheRepository
from nel.app.embeddings_cache.service.generation_service import (
    EmbeddingGenerationService,
    _ENTITY_TYPES as _ALL_ENTITY_TYPES,
)
import nel.app.embeddings_cache.service.generation_service as _gen_mod
from nel.app.embeddings_cache.service.taxonomy_source import (
    TaxonomyAPISource,
    QualificationsMongoSource,
)
from nel.app.embeddings_cache.types import CacheStatus


# ── Dry-run aware generation service wrapper ──────────────────────────────────

class DryRunGenerationService:
    """Wraps EmbeddingGenerationService to log-only in dry-run mode.

    In dry-run mode we still connect to both databases and the taxonomy API,
    but no documents are inserted and no cache status is written.
    """

    def __init__(self, inner: EmbeddingGenerationService, entity_types: list[str], hot_run: bool):
        self._inner = inner
        self._entity_types = entity_types
        self._hot_run = hot_run

    async def run(
        self, taxonomy_model_id: str, nel_model_id: str, force: bool
    ) -> None:
        original = _gen_mod._ENTITY_TYPES
        _gen_mod._ENTITY_TYPES = self._entity_types
        try:
            if self._hot_run:
                await self._inner.generate_for_combination(
                    taxonomy_model_id=taxonomy_model_id,
                    nel_model_id=nel_model_id,
                    force=force,
                )
            else:
                await self._dry_run(taxonomy_model_id, nel_model_id, force)
        finally:
            _gen_mod._ENTITY_TYPES = original

    async def _dry_run(self, taxonomy_model_id: str, nel_model_id: str, force: bool) -> None:
        repo = self._inner._cache_repo
        source_map = {
            et: self._inner._get_source(et, taxonomy_model_id)
            for et in self._entity_types
        }
        for entity_type in self._entity_types:
            status = await repo.get_cache_status(taxonomy_model_id, nel_model_id, entity_type)
            if status and status.status == CacheStatus.ready and not force:
                logger.info("[DRY-RUN] %s is already ready — would skip (use --force to regenerate)", entity_type)
                continue

            if force:
                logger.info("[DRY-RUN] Would delete existing embeddings for %s", entity_type)

            logger.info("[DRY-RUN] Streaming %s items from source …", entity_type)
            count = 0
            t0 = time.monotonic()
            progress = tqdm(desc=f"[dry-run] counting {entity_type}", unit="item", leave=False)
            async for _ in source_map[entity_type]:
                count += 1
                progress.update(1)
            progress.close()
            elapsed = time.monotonic() - t0

            logger.info(
                "[DRY-RUN] Would embed and insert %d %s items (%.1fs to stream)",
                count, entity_type, elapsed,
            )
            logger.info("[DRY-RUN] Would mark %s cache status as 'ready'", entity_type)


# ── Vector search index management ───────────────────────────────────────────

_INDEX_NAME = "nel_embedding_index_384"
_INDEX_DEFINITION = {
    "fields": [
        {
            "type": "vector",
            "path": "embedding",
            "numDimensions": 384,
            "similarity": "cosine",
        },
        {"type": "filter", "path": "taxonomy_model_id"},
        {"type": "filter", "path": "nel_model_id"},
    ]
}


async def _upsert_vector_search_index(collection, hot_run: bool) -> None:
    exists = False
    async for idx in collection.list_search_indexes():
        if idx["name"] == _INDEX_NAME:
            exists = True
            break

    if exists:
        if hot_run:
            logger.info("  Updating vector search index '%s' on %s", _INDEX_NAME, collection.name)
            await collection.update_search_index(_INDEX_NAME, _INDEX_DEFINITION)
        else:
            logger.info("  [DRY-RUN] Would update vector search index '%s' on %s", _INDEX_NAME, collection.name)
    else:
        model = SearchIndexModel(definition=_INDEX_DEFINITION, name=_INDEX_NAME, type="vectorSearch")
        if hot_run:
            logger.info("  Creating vector search index '%s' on %s", _INDEX_NAME, collection.name)
            await collection.create_search_index(model=model)
        else:
            logger.info("  [DRY-RUN] Would create vector search index '%s' on %s", _INDEX_NAME, collection.name)


async def create_vector_search_indexes(taxonomy_db, hot_run: bool) -> None:
    from nel.app.server_dependencies.database_collections import Collections
    collections = [
        Collections.OCCUPATION_EMBEDDINGS,
        Collections.SKILL_EMBEDDINGS,
        Collections.QUALIFICATION_EMBEDDINGS,
    ]
    # list_search_indexes() can be flaky when called concurrently — run sequentially
    for col_name in collections:
        await _upsert_vector_search_index(taxonomy_db[col_name], hot_run)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    hot_run: bool = args.hot_run
    indexes_only: bool = args.indexes_only

    app_uri = os.environ["APPLICATION_MONGODB_URI"]
    app_db_name = os.environ.get("APPLICATION_DATABASE_NAME", "tabiya-classifier")
    taxonomy_uri = os.environ["TAXONOMY_MONGODB_URI"]
    taxonomy_db_name = os.environ.get("TAXONOMY_DATABASE_NAME", "tabiya-taxonomy")
    taxonomy_base_url = os.environ.get("TAXONOMY_API_BASE_URL", "https://taxonomy.tabiya.tech")
    taxonomy_api_key = os.environ.get("TAXONOMY_API_KEY", "")

    mode_label = "HOT-RUN (writes enabled)" if hot_run else "DRY-RUN (no writes)"
    logger.info("=" * 60)
    logger.info("NEL Embedding Generation — %s", mode_label)
    logger.info("  taxonomy_model_id : %s", args.taxonomy_model_id)
    logger.info("  nel_model_id      : %s", args.nel_model_id)
    logger.info("  indexes_only      : %s", indexes_only)
    if not indexes_only:
        logger.info("  entity_types      : %s", args.entity_types)
        logger.info("  force             : %s", args.force)
    logger.info("  taxonomy API      : %s", taxonomy_base_url)
    logger.info("  app DB            : %s", app_db_name)
    logger.info("  taxonomy DB       : %s", taxonomy_db_name)
    logger.info("=" * 60)

    # ── [1] Connect ───────────────────────────────────────────────────────────
    logger.info("[1] Connecting to MongoDB …")
    app_client = AsyncIOMotorClient(app_uri, tlsAllowInvalidCertificates=True)
    taxonomy_client = AsyncIOMotorClient(taxonomy_uri, tlsAllowInvalidCertificates=True)
    app_db = app_client.get_database(app_db_name)
    taxonomy_db = taxonomy_client.get_database(taxonomy_db_name)

    repo = EmbeddingsCacheRepository(app_db=app_db, taxonomy_db=taxonomy_db)

    if not indexes_only:
        if hot_run:
            logger.info("  Ensuring MongoDB indexes …")
            await repo.ensure_indexes()
        else:
            logger.info("  [DRY-RUN] Would ensure MongoDB indexes")

        # ── [2] Load embedding model ──────────────────────────────────────────
        logger.info("[2] Loading SentenceTransformer model: %s …", args.nel_model_id)
        t0 = time.monotonic()
        embedding_svc = SentenceTransformerEmbeddingService(args.nel_model_id)
        logger.info(
            "  Model loaded in %.1fs — dimensions: %d",
            time.monotonic() - t0,
            embedding_svc.dimensions,
        )

        # ── [3] Build services ────────────────────────────────────────────────
        logger.info("[3] Initialising taxonomy source and generation service …")
        taxonomy_source = TaxonomyAPISource(
            base_url=taxonomy_base_url,
            api_key=taxonomy_api_key,
        )
        qual_source = QualificationsMongoSource(app_db=app_db)

        inner_svc = EmbeddingGenerationService(
            cache_repository=repo,
            embedding_service=embedding_svc,
            taxonomy_source=taxonomy_source,
            qualifications_source=qual_source,
            page_size=100,
        )
        svc = DryRunGenerationService(inner_svc, entity_types=args.entity_types, hot_run=hot_run)

        # ── [4] Generate ──────────────────────────────────────────────────────
        logger.info("[4] Running embedding generation for: %s", args.entity_types)
        t_start = time.monotonic()
        await svc.run(
            taxonomy_model_id=args.taxonomy_model_id,
            nel_model_id=args.nel_model_id,
            force=args.force,
        )
        elapsed = time.monotonic() - t_start
        if hot_run:
            logger.info("Embeddings generated in %.1fs", elapsed)
        else:
            logger.info("Dry-run complete in %.1fs — no writes were made", elapsed)

    # ── [5] Vector search indexes ─────────────────────────────────────────────
    logger.info("[5] Creating/updating Atlas vector search indexes …")
    await create_vector_search_indexes(taxonomy_db, hot_run=hot_run)

    logger.info("=" * 60)
    if hot_run:
        logger.info("Done.")
    else:
        logger.info("Dry-run complete — no writes were made. Re-run with --hot-run to apply.")
    logger.info("=" * 60)

    app_client.close()
    taxonomy_client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--taxonomy-model-id",
        required=True,
        metavar="UUID",
        help="Taxonomy model UUID (from taxonomy.tabiya.tech /api/app/modelInfo)",
    )
    parser.add_argument(
        "--nel-model-id",
        default="all-MiniLM-L6-v2",
        metavar="MODEL",
        help="SentenceTransformer model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--entity-types",
        nargs="+",
        choices=["occupation", "skill", "qualification"],
        default=["occupation", "skill", "qualification"],
        metavar="TYPE",
        help="Entity types to generate: occupation skill qualification (default: all three)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing embeddings and regenerate even if already 'ready'",
    )
    parser.add_argument(
        "--indexes-only",
        action="store_true",
        help="Skip embedding generation — only create/update Atlas vector search indexes.",
    )
    parser.add_argument(
        "--hot-run",
        action="store_true",
        help=(
            "Perform actual writes (inserts, cache status updates, index creation).\n"
            "Without this flag the script runs in DRY-RUN mode: it reads from all\n"
            "sources and logs what would happen, but writes nothing."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(main(parse_args()))
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as exc:
        logger.error("Script failed: %s", exc, exc_info=True)
        sys.exit(1)
