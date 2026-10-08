"""Post-deploy manifest smoke test.

Boots nothing — it hits the *live* plugin bundles that were just deployed
and asserts every installed plugin serves a well-formed manifest over
HTTP. This complements the in-process bundle unit tests (which validate
the manifest objects before serialization) by catching problems that only
appear against the real deployment: a bundle that failed to start, an env
var that left a plugin unwired, a container image built from stale code,
or a serialization break that the unit tests can't see.

For each bundle it:
  1. Reads the bundle's Cloud Run URL from the backend Pulumi stack output.
  2. Fetches `GET {bundle_url}/health` to learn the installed plugin ids.
  3. Fetches `GET {bundle_url}/plugin/{plugin_id}/manifest` for each one and
     asserts the required fields are present and the declared slot types are
     drawn from the known slot vocabulary.

The manifest checks are intentionally kept as lightweight structural
assertions rather than importing the `tabiya_plugin_contracts` Pydantic
model: this script runs inside the IaC deploy job, which installs only the
`iac/` requirements and does not have the backend package on its path. The
bundle unit tests already validate the full Pydantic contract in CI; here
we only need to confirm the deployed surface is reachable and well-shaped.

The bundles are private Cloud Run services, so every request carries a GCP
identity token minted for the bundle's URL as its audience. The deploy
service account is already granted `roles/run.invoker` on both bundles.

Exit code is non-zero if any bundle is unreachable or any manifest is
malformed, so the deploy job fails loudly instead of shipping a broken
plugin surface.
"""

from __future__ import annotations

import argparse
import sys

import google.auth.transport.requests
import google.oauth2.id_token
import httpx
import pulumi.automation as auto


_BACKEND_STACK_WORK_DIR = "iac/backend"
_BUNDLE_URL_OUTPUTS = {
    "tabiya_core": "tabiyaCoreUrl",
    "tabiya_io": "tabiyaIoUrl",
}
# Must match tabiya_plugin_contracts.SlotType. Duplicated here (not imported)
# because the deploy job doesn't install the backend package — see module docstring.
_VALID_SLOT_TYPES = {"None", "RawText", "RawTextStream", "Entities", "LinkedEntities"}
_REQUIRED_MANIFEST_FIELDS = ("plugin_id", "name", "version", "category", "input_slot", "output_slot")
_REQUEST_TIMEOUT_SECONDS = 30.0


def _identity_token_for(audience_url: str) -> str:
    """Mint a GCP identity token whose audience is the bundle's base URL."""

    auth_request = google.auth.transport.requests.Request()
    return google.oauth2.id_token.fetch_id_token(auth_request, audience_url)


def _read_bundle_urls(stack_name: str) -> dict[str, str]:
    """Read the deployed bundle URLs from the backend Pulumi stack outputs."""

    stack = auto.select_stack(stack_name=stack_name, work_dir=_BACKEND_STACK_WORK_DIR)
    outputs = stack.outputs()
    bundle_urls: dict[str, str] = {}
    for bundle_name, output_key in _BUNDLE_URL_OUTPUTS.items():
        if output_key not in outputs:
            raise SystemExit(
                f"Backend stack '{stack_name}' has no output '{output_key}' "
                f"for bundle '{bundle_name}'."
            )
        bundle_urls[bundle_name] = str(outputs[output_key].value).rstrip("/")
    return bundle_urls


def _installed_plugin_ids(client: httpx.Client, bundle_url: str, headers: dict) -> list[str]:
    """Ask the bundle's /health which plugins it mounted."""

    response = client.get(f"{bundle_url}/health", headers=headers)
    response.raise_for_status()
    payload = response.json()
    installed = payload.get("installed")
    if not isinstance(installed, list) or not installed:
        raise SystemExit(
            f"Bundle at {bundle_url} reported no installed plugins: {payload!r}"
        )
    return [str(plugin_id) for plugin_id in installed]


def _slot_type_value(manifest: dict, slot_field: str) -> str:
    slot = manifest.get(slot_field)
    if not isinstance(slot, dict) or "type" not in slot:
        raise SystemExit(
            f"Manifest '{slot_field}' is missing or has no 'type': {slot!r}"
        )
    return str(slot["type"])


def _check_manifest(client: httpx.Client, bundle_url: str, plugin_id: str, headers: dict) -> None:
    """Fetch and validate one plugin's live manifest, raising on any problem."""

    response = client.get(f"{bundle_url}/plugin/{plugin_id}/manifest", headers=headers)
    response.raise_for_status()
    manifest = response.json()

    missing_fields = [field for field in _REQUIRED_MANIFEST_FIELDS if field not in manifest]
    if missing_fields:
        raise SystemExit(
            f"Plugin '{plugin_id}' manifest is missing required field(s): "
            f"{', '.join(missing_fields)}."
        )
    if manifest["plugin_id"] != plugin_id:
        raise SystemExit(
            f"{bundle_url}/plugin/{plugin_id}/manifest returned mismatched "
            f"plugin_id '{manifest['plugin_id']}'."
        )

    input_slot_type = _slot_type_value(manifest, "input_slot")
    if input_slot_type not in _VALID_SLOT_TYPES:
        raise SystemExit(
            f"Plugin '{plugin_id}' declares unknown input slot '{input_slot_type}'."
        )
    output_slot_type = _slot_type_value(manifest, "output_slot")
    if output_slot_type not in _VALID_SLOT_TYPES:
        raise SystemExit(
            f"Plugin '{plugin_id}' declares unknown output slot '{output_slot_type}'."
        )


def _smoke_bundle(bundle_name: str, bundle_url: str) -> int:
    """Smoke-test every plugin in one bundle. Returns the count checked."""

    headers = {"Authorization": f"Bearer {_identity_token_for(bundle_url)}"}
    with httpx.Client(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
        plugin_ids = _installed_plugin_ids(client, bundle_url, headers)
        for plugin_id in plugin_ids:
            _check_manifest(client, bundle_url, plugin_id, headers)
            print(f"  ✓ {bundle_name}: {plugin_id}")
    return len(plugin_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stack",
        required=True,
        help="Pulumi stack name whose backend outputs hold the bundle URLs.",
    )
    args = parser.parse_args()

    bundle_urls = _read_bundle_urls(args.stack)
    total_checked = 0
    for bundle_name, bundle_url in bundle_urls.items():
        print(f"Smoke-testing {bundle_name} at {bundle_url}")
        total_checked += _smoke_bundle(bundle_name, bundle_url)

    print(f"\nManifest smoke test passed: {total_checked} manifest(s) validated.")


if __name__ == "__main__":
    try:
        main()
    except (httpx.HTTPError, SystemExit) as exc:
        print(f"Manifest smoke test FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
