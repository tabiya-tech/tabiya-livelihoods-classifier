"""Generate TypeScript types from the exported JSON Schemas.

Prerequisite: `contracts/*.schema.json` produced by `export_json_schema.py`.

Runs `npx json-schema-to-typescript` for each schema and concatenates the
results into a single `.generated.ts` at
`app/src/lib/api/pluginContracts.generated.ts`. A header comment marks the
file as generated and points readers at this script.

Committed. CI re-runs and asserts a clean git diff.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONTRACTS_DIR = REPO_ROOT / "contracts"
CLASSIFIER_ROOT = REPO_ROOT.parent.parent
FRONTEND_OUTPUT = (
    CLASSIFIER_ROOT / "app" / "src" / "lib" / "api" / "pluginContracts.generated.ts"
)

SCHEMA_FILES = [
    ("manifest.schema.json", "ManifestNs"),
    ("invoke_request.schema.json", "InvokeRequestNs"),
    ("invoke_response.schema.json", "InvokeResponseNs"),
    ("error_envelope.schema.json", "ErrorEnvelopeNs"),
    ("health.schema.json", "HealthNs"),
    ("slots.schema.json", "SlotsNs"),
]

HEADER = """/**
 * AUTO-GENERATED — do not edit by hand.
 *
 * Source: backend/tabiya_plugin_contracts/contracts/*.schema.json
 * Generator: backend/tabiya_plugin_contracts/scripts/export_ts_types.py
 *
 * To regenerate:
 *   cd backend/tabiya_plugin_contracts
 *   python scripts/export_json_schema.py
 *   python scripts/export_ts_types.py
 *
 * CI re-runs both scripts and fails on a non-empty git diff.
 *
 * Each schema is emitted into its own namespace so field-level types
 * (e.g. `Detail`, `Message`) don't collide. Consumers typically import
 * the root types re-exported at the bottom of this file.
 */

/* eslint-disable */
"""


_MODULE_REEXPORTS = """
// ─── Root type re-exports ─────────────────────────────────────────────
// Import these directly; the namespaces above are an implementation detail
// of the codegen and may be reorganised without a breaking change.

export type Manifest = ManifestNs.Manifest;
export type InvokeRequest = InvokeRequestNs.InvokeRequest;
export type InvokeResponse = InvokeResponseNs.InvokeResponse;
export type ErrorEnvelope = ErrorEnvelopeNs.ErrorEnvelope;
export type Health = HealthNs.Health;
export type SlotPayloads = SlotsNs.SlotPayloads;
"""


def _resolve_json2ts() -> list[str]:
    # The frontend uses Yarn Berry with Plug'n'Play, so there is no
    # node_modules/.bin. `yarn json2ts` (from the app dir) resolves the
    # binary through PnP.
    if shutil.which("yarn") is None:
        raise RuntimeError("Yarn is required to run json2ts under PnP. Install Yarn first.")
    return ["yarn", "json2ts"]


APP_DIR = CLASSIFIER_ROOT / "app"


def _generate(schema_path: Path, json2ts: list[str]) -> str:
    # json-schema-to-typescript v15 CLI expects flags via `--flag=value`.
    # `--additionalProperties=false` matches the old `--no-additionalProperties`.
    result = subprocess.run(
        [
            *json2ts,
            "--input",
            str(schema_path),
            "--additionalProperties=false",
            "--unreachableDefinitions",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=APP_DIR,
    )
    return result.stdout


def main() -> None:
    if not CONTRACTS_DIR.exists():
        print("Run export_json_schema.py first.", file=sys.stderr)
        sys.exit(1)

    json2ts = _resolve_json2ts()
    parts: list[str] = [HEADER]
    for filename, namespace in SCHEMA_FILES:
        schema_path = CONTRACTS_DIR / filename
        raw = _generate(schema_path, json2ts)
        # Wrap each generated block in its own namespace so that field-level
        # types like `Detail` (present in multiple schemas) don't collide.
        # Root exports (Manifest, InvokeRequest, …) get re-exported at the
        # module level under a stable name in _MODULE_REEXPORTS below.
        parts.append(
            f"\n// ─── from {filename} ─────────────────────────────────────────\n"
        )
        parts.append(f"export namespace {namespace} {{\n{raw}\n}}\n")

    parts.append(_MODULE_REEXPORTS)

    FRONTEND_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    FRONTEND_OUTPUT.write_text("".join(parts))
    print(f"Wrote {FRONTEND_OUTPUT.relative_to(CLASSIFIER_ROOT)}")


if __name__ == "__main__":
    main()
