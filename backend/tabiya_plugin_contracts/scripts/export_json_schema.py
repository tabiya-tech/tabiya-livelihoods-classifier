"""Emit JSON Schemas + version marker from the Pydantic contract models.

Run from the package root:

    python scripts/export_json_schema.py

Writes:
    contracts/manifest.schema.json
    contracts/invoke_request.schema.json
    contracts/invoke_response.schema.json
    contracts/error_envelope.schema.json
    contracts/health.schema.json
    contracts/slots.schema.json         # bundles all slot payloads under $defs
    contracts/version.txt

Both files are committed. CI re-runs this script and asserts a clean git diff
so drift between Pydantic definitions and the generated schemas fails the build.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from pydantic import BaseModel  # noqa: E402
from pydantic.json_schema import models_json_schema  # noqa: E402

from tabiya_plugin_contracts import (  # noqa: E402
    CONTRACT_VERSION,
    Entities,
    ErrorEnvelope,
    Health,
    InvokeRequest,
    InvokeResponse,
    LinkedEntities,
    Manifest,
    NoneSlot,
    RawText,
    RawTextStream,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACTS_DIR = REPO_ROOT / "contracts"


def _dump(model_cls: type[BaseModel], filename: str) -> None:
    schema = model_cls.model_json_schema(by_alias=True)
    path = CONTRACTS_DIR / filename
    path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")


def _dump_slots() -> None:
    # Bundle all slot payloads under a single file so the TS side can generate
    # a discriminated union without importing five separate schema files.
    # Use pydantic.json_schema.models_json_schema so nested subdefs
    # (EntitySpan, Entity, Match, …) are hoisted to the top-level $defs and
    # $ref pointers are rewritten accordingly.
    _, combined = models_json_schema(
        [
            (NoneSlot, "validation"),
            (RawText, "validation"),
            (RawTextStream, "validation"),
            (Entities, "validation"),
            (LinkedEntities, "validation"),
        ],
        by_alias=True,
        title="SlotPayloads",
    )
    (CONTRACTS_DIR / "slots.schema.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    CONTRACTS_DIR.mkdir(exist_ok=True)
    _dump(Manifest, "manifest.schema.json")
    _dump(InvokeRequest, "invoke_request.schema.json")
    _dump(InvokeResponse, "invoke_response.schema.json")
    _dump(ErrorEnvelope, "error_envelope.schema.json")
    _dump(Health, "health.schema.json")
    _dump_slots()
    (CONTRACTS_DIR / "version.txt").write_text(CONTRACT_VERSION + "\n")


if __name__ == "__main__":
    main()
