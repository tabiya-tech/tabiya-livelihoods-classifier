"""Manifest smoke test — every installed plugin serves a well-formed manifest.

Unit tests elsewhere check the in-process `MANIFEST` Python objects. This
suite instead boots the *real* bundle FastAPI app and hits each plugin's
live `GET /plugin/{plugin_id}/manifest` endpoint over ASGI, then re-parses
the JSON through the `Manifest` Pydantic model. That catches serialization
drift, bad slot enums, and missing fields that a pure-Python assertion on
the manifest object would never see, because those only surface once the
manifest crosses the HTTP boundary.

Every test uses GIVEN/WHEN/THEN inline comments and named `given*` /
`expected*` variables.
"""

from __future__ import annotations

import httpx
import pytest

from tabiya_plugin_contracts import Manifest, SlotType
from tabiya_plugin_contracts.version import CONTRACT_VERSION

from tabiya_io.main import app as bundle_app
from tabiya_io.plugins import INSTALLED_PLUGINS


_VALID_SLOT_TYPES = {slot_type.value for slot_type in SlotType}
_INSTALLED_PLUGIN_IDS = [manifest.plugin_id for manifest, _, _ in INSTALLED_PLUGINS]


@pytest.fixture(autouse=True)
def _force_local_mode(monkeypatch):
    # The bundle-wide auth dependency short-circuits in local mode; the smoke
    # test never exercises real GCP identity tokens.
    monkeypatch.setenv("TARGET_ENVIRONMENT_TYPE", "local")
    yield


def _make_client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=bundle_app)
    return httpx.AsyncClient(transport=transport, base_url="http://bundle.test")


@pytest.mark.parametrize("givenPluginId", _INSTALLED_PLUGIN_IDS)
async def test_installed_plugin_serves_a_valid_manifest(givenPluginId: str) -> None:
    # GIVEN a plugin installed in the running tabiya_io bundle
    expectedStatus = 200

    # WHEN we fetch its live manifest over HTTP
    async with _make_client() as client:
        response = await client.get(f"/plugin/{givenPluginId}/manifest")

    # THEN the endpoint responds 200 and the payload round-trips back into the
    # Manifest contract model without loss
    assert response.status_code == expectedStatus
    parsedManifest = Manifest.model_validate(response.json())

    # AND the manifest identifies itself as the plugin we asked for
    assert parsedManifest.plugin_id == givenPluginId

    # AND both slots declare types drawn from the shared SlotType enum
    assert parsedManifest.input_slot.type.value in _VALID_SLOT_TYPES
    assert parsedManifest.output_slot.type.value in _VALID_SLOT_TYPES

    # AND the adapter auto-stamped the contract version it was built against
    assert parsedManifest.x_tabiya_contract_version == CONTRACT_VERSION


async def test_bundle_installs_at_least_one_plugin() -> None:
    # GIVEN the installed-plugins list the bundle mounts at startup
    givenInstalledPluginIds = _INSTALLED_PLUGIN_IDS

    # WHEN we count them
    actualCount = len(givenInstalledPluginIds)

    # THEN the bundle is not empty — an empty bundle would make the
    # parametrized smoke test above vacuously pass, so guard against it
    expectedMinimumCount = 1
    assert actualCount >= expectedMinimumCount


async def test_manifest_plugin_ids_are_unique() -> None:
    # GIVEN every installed plugin id
    givenPluginIds = _INSTALLED_PLUGIN_IDS

    # WHEN we compare the raw list against its de-duplicated set
    expectedUniqueCount = len(set(givenPluginIds))

    # THEN no two plugins share an id (a collision would silently shadow a
    # route under `/plugin/{plugin_id}`)
    assert len(givenPluginIds) == expectedUniqueCount
