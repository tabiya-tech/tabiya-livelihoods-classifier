"""Plugin registry tests.

Deterministic — a fake `IHttpClient` returns pre-canned Response objects
keyed by URL, so the tests never touch the network and never depend on
respx. Every test uses GIVEN/WHEN/THEN inline comments and
`given*` / `expected*` named variables.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Callable, Optional

import httpx
import pytest
from tabiya_plugin_contracts import CONTRACT_VERSION, Manifest, PluginCategory, Slot, SlotType

from classify_v2.app.pipelines.registry import (
    CatalogEntry,
    PluginRegistry,
    PluginStatus,
    PluginUnreachableError,
    load_catalog,
)


class _FakeHttp:
    """Test double for httpx.AsyncClient.get.

    Configure with `on(url, response_or_factory)`; `get(url)` returns the
    configured Response, calls the factory, or raises the configured
    exception.
    """

    def __init__(self) -> None:
        self._responses: dict[str, Callable[[], httpx.Response]] = {}
        self._exceptions: dict[str, Exception] = {}
        self.calls: list[str] = []
        self.headers_by_url: dict[str, dict[str, str]] = {}

    def on(self, url: str, response: httpx.Response) -> "_FakeHttp":
        self._responses[url] = lambda: response
        return self

    def on_factory(self, url: str, factory: Callable[[], httpx.Response]) -> "_FakeHttp":
        self._responses[url] = factory
        return self

    def raise_on(self, url: str, exc: Exception) -> "_FakeHttp":
        self._exceptions[url] = exc
        return self

    async def get(
        self,
        url: str,
        timeout: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        self.calls.append(url)
        self.headers_by_url[url] = dict(headers or {})
        if url in self._exceptions:
            raise self._exceptions[url]
        if url in self._responses:
            return self._responses[url]()
        return httpx.Response(status_code=404, text="not configured for test")


def _valid_manifest_dict(plugin_id: str = "tabiya.ner.v1") -> dict:
    """Build a minimal but complete manifest JSON body — matches Manifest schema."""

    manifest = Manifest(
        plugin_id=plugin_id,
        name=plugin_id,
        version="0.1.0",
        category=PluginCategory.CORE,
        summary="Test manifest.",
        icon="ner",
        input_slot=Slot(type=SlotType.RAW_TEXT),
        output_slot=Slot(type=SlotType.ENTITIES),
        config_schema={},
        timeout_ms=5_000,
        **{"x-tabiya-contract-version": CONTRACT_VERSION},
    )
    return manifest.model_dump(by_alias=True, exclude_none=True)


def _core_ner_and_two_coming_soon() -> list[CatalogEntry]:
    return [
        CatalogEntry(
            plugin_id="tabiya.ner.v1",
            url_env="TABIYA_CORE_BUNDLE_URL",
            path="/plugin/tabiya.ner.v1",
        ),
        CatalogEntry(plugin_id="tabiya.source.scraper.v1", coming_soon=True),
        CatalogEntry(plugin_id="tabiya.branching.language_router.v1", coming_soon=True),
    ]


def _valid_manifest_response(plugin_id: str = "tabiya.ner.v1") -> httpx.Response:
    return httpx.Response(status_code=200, json=_valid_manifest_dict(plugin_id))


def test_load_catalog_parses_shipped_json_into_typed_entries() -> None:
    # GIVEN the shipped catalog.json
    givenPath = (
        Path(__file__).parent / "catalog.json"
    )

    # WHEN we load it
    entries = load_catalog(givenPath)

    # THEN every entry is a CatalogEntry and the shipped ids are present
    expectedIds = {
        "tabiya.ner.v1",
        "tabiya.nel.v1",
        "tabiya.source.text.v1",
        "tabiya.source.json.v1",
        "tabiya.source.json_entities.v1",
        "tabiya.sink.results.v1",
        "tabiya.source.scraper.v1",
        "tabiya.transform.stopwords.v1",
        "tabiya.sink.database.v1",
        "tabiya.branching.language_router.v1",
    }
    assert {entry.plugin_id for entry in entries} == expectedIds


def test_load_catalog_raises_when_shape_is_not_a_list(tmp_path) -> None:
    # GIVEN a catalog that is a JSON object instead of a list
    givenPath = tmp_path / "catalog.json"
    givenPath.write_text(json.dumps({"not": "a list"}))

    # WHEN we load it, THEN we get a ValueError with a helpful message
    with pytest.raises(ValueError, match="must be a JSON list"):
        load_catalog(givenPath)


async def test_refresh_marks_coming_soon_entries_unavailable() -> None:
    # GIVEN a catalog whose only entry is coming_soon
    givenCatalog = [CatalogEntry(plugin_id="tabiya.source.scraper.v1", coming_soon=True)]
    fake_http = _FakeHttp()
    registry = PluginRegistry(givenCatalog, http_client=fake_http, env={})

    # WHEN we refresh
    await registry.refresh()

    # THEN status is UNAVAILABLE with the coming_soon reason
    expectedStatus = PluginStatus.UNAVAILABLE
    expectedReason = "coming_soon"
    entry = await registry.get("tabiya.source.scraper.v1")
    assert entry is not None
    assert entry.status == expectedStatus
    assert entry.last_error == expectedReason
    assert entry.coming_soon is True
    # AND we never hit the network for coming_soon entries
    assert fake_http.calls == []


async def test_refresh_marks_missing_env_var_unavailable() -> None:
    # GIVEN a catalog whose url_env is unset
    givenCatalog = _core_ner_and_two_coming_soon()
    fake_http = _FakeHttp()
    registry = PluginRegistry(givenCatalog, http_client=fake_http, env={})

    # WHEN we refresh
    await registry.refresh()

    # THEN the NER plugin is UNAVAILABLE with a clear reason
    expectedStatus = PluginStatus.UNAVAILABLE
    entry = await registry.get("tabiya.ner.v1")
    assert entry is not None
    assert entry.status == expectedStatus
    assert entry.last_error == "env var TABIYA_CORE_BUNDLE_URL is unset"
    assert entry.resolved_url is None


async def test_refresh_marks_reachable_plugin_enabled() -> None:
    # GIVEN a resolved URL and a valid manifest response
    givenBase = "http://tabiya-core:5010"
    givenManifestUrl = f"{givenBase}/plugin/tabiya.ner.v1/manifest"
    fake_http = _FakeHttp().on(givenManifestUrl, _valid_manifest_response())
    givenCatalog = [
        CatalogEntry(
            plugin_id="tabiya.ner.v1",
            url_env="TABIYA_CORE_BUNDLE_URL",
            path="/plugin/tabiya.ner.v1",
        )
    ]
    registry = PluginRegistry(
        givenCatalog,
        http_client=fake_http,
        env={"TABIYA_CORE_BUNDLE_URL": givenBase},
    )

    # WHEN we refresh
    await registry.refresh()

    # THEN the plugin is ENABLED and the manifest is cached
    expectedStatus = PluginStatus.ENABLED
    entry = await registry.get("tabiya.ner.v1")
    assert entry is not None
    assert entry.status == expectedStatus
    assert entry.manifest is not None
    assert entry.manifest.plugin_id == "tabiya.ner.v1"
    assert entry.resolved_url == f"{givenBase}/plugin/tabiya.ner.v1"
    assert fake_http.calls == [givenManifestUrl]


async def test_manifest_declaring_coming_soon_is_cached_but_unavailable() -> None:
    # GIVEN a reachable plugin whose manifest declares x-tabiya-coming-soon
    givenBase = "http://tabiya-io:5011"
    givenManifestUrl = f"{givenBase}/plugin/tabiya.sink.database.v1/manifest"
    comingSoonManifest = Manifest(
        plugin_id="tabiya.sink.database.v1",
        name="Database Sink",
        version="0.1.0",
        category=PluginCategory.CORE,
        summary="Writes results to a database.",
        icon="download",
        input_slot=Slot(type=SlotType.LINKED_ENTITIES),
        output_slot=Slot(type=SlotType.NONE),
        config_schema={},
        timeout_ms=5_000,
        **{
            "x-tabiya-contract-version": CONTRACT_VERSION,
            "x-tabiya-coming-soon": True,
        },
    )
    fake_http = _FakeHttp().on(
        givenManifestUrl,
        httpx.Response(200, json=comingSoonManifest.model_dump(by_alias=True, exclude_none=True)),
    )
    givenCatalog = [
        CatalogEntry(
            plugin_id="tabiya.sink.database.v1",
            url_env="TABIYA_IO_BUNDLE_URL",
            path="/plugin/tabiya.sink.database.v1",
        )
    ]
    registry = PluginRegistry(
        givenCatalog,
        http_client=fake_http,
        env={"TABIYA_IO_BUNDLE_URL": givenBase},
    )

    # WHEN we refresh
    await registry.refresh()

    # THEN the manifest is cached (so the palette can show name/category), but
    # the plugin is flagged coming_soon + UNAVAILABLE (so it stays undroppable).
    entry = await registry.get("tabiya.sink.database.v1")
    assert entry is not None
    assert entry.manifest is not None
    assert entry.manifest.name == "Database Sink"
    assert entry.coming_soon is True
    assert entry.status == PluginStatus.UNAVAILABLE


async def test_refresh_attaches_identity_token_to_manifest_fetch() -> None:
    # GIVEN a reachable bundle and an identity provider that mints a token
    givenBase = "http://tabiya-core:5010"
    givenManifestUrl = f"{givenBase}/plugin/tabiya.ner.v1/manifest"
    givenToken = "eyJ.header.sig"
    expectedAuthHeader = f"Bearer {givenToken}"
    fake_http = _FakeHttp().on(givenManifestUrl, _valid_manifest_response())

    class _FakeIdentity:
        async def get_id_token(self, url: str) -> str:
            return givenToken

    givenCatalog = [
        CatalogEntry(
            plugin_id="tabiya.ner.v1",
            url_env="TABIYA_CORE_BUNDLE_URL",
            path="/plugin/tabiya.ner.v1",
        )
    ]
    registry = PluginRegistry(
        givenCatalog,
        http_client=fake_http,
        env={"TABIYA_CORE_BUNDLE_URL": givenBase},
        identity_token_provider=_FakeIdentity(),
    )

    # WHEN we refresh
    await registry.refresh()

    # THEN the manifest fetch carried the bearer token
    assert fake_http.headers_by_url[givenManifestUrl].get("Authorization") == expectedAuthHeader


async def test_refresh_sends_no_auth_header_without_provider() -> None:
    # GIVEN a reachable bundle and NO identity provider (local mode)
    givenBase = "http://tabiya-core:5010"
    givenManifestUrl = f"{givenBase}/plugin/tabiya.ner.v1/manifest"
    expectedAuthHeader = None
    fake_http = _FakeHttp().on(givenManifestUrl, _valid_manifest_response())
    givenCatalog = [
        CatalogEntry(
            plugin_id="tabiya.ner.v1",
            url_env="TABIYA_CORE_BUNDLE_URL",
            path="/plugin/tabiya.ner.v1",
        )
    ]
    registry = PluginRegistry(
        givenCatalog,
        http_client=fake_http,
        env={"TABIYA_CORE_BUNDLE_URL": givenBase},
    )

    # WHEN we refresh
    await registry.refresh()

    # THEN no Authorization header was attached
    assert fake_http.headers_by_url[givenManifestUrl].get("Authorization") is expectedAuthHeader


async def test_refresh_marks_unreachable_plugin_unavailable() -> None:
    # GIVEN an httpx transport error at the manifest URL
    givenBase = "http://tabiya-core:5010"
    givenManifestUrl = f"{givenBase}/plugin/tabiya.ner.v1/manifest"
    fake_http = _FakeHttp().raise_on(
        givenManifestUrl, httpx.ConnectError("Connection refused")
    )
    givenCatalog = [
        CatalogEntry(
            plugin_id="tabiya.ner.v1",
            url_env="TABIYA_CORE_BUNDLE_URL",
            path="/plugin/tabiya.ner.v1",
        )
    ]
    registry = PluginRegistry(
        givenCatalog,
        http_client=fake_http,
        env={"TABIYA_CORE_BUNDLE_URL": givenBase},
    )

    # WHEN we refresh
    await registry.refresh()

    # THEN the plugin is UNAVAILABLE with the transport error as reason
    expectedStatus = PluginStatus.UNAVAILABLE
    entry = await registry.get("tabiya.ner.v1")
    assert entry is not None
    assert entry.status == expectedStatus
    assert entry.manifest is None
    assert "unreachable" in (entry.last_error or "")


async def test_refresh_marks_non_200_manifest_unavailable() -> None:
    # GIVEN a plugin URL that returns HTTP 500
    givenBase = "http://tabiya-core:5010"
    givenManifestUrl = f"{givenBase}/plugin/tabiya.ner.v1/manifest"
    fake_http = _FakeHttp().on(givenManifestUrl, httpx.Response(status_code=500))
    givenCatalog = [
        CatalogEntry(
            plugin_id="tabiya.ner.v1",
            url_env="TABIYA_CORE_BUNDLE_URL",
            path="/plugin/tabiya.ner.v1",
        )
    ]
    registry = PluginRegistry(
        givenCatalog,
        http_client=fake_http,
        env={"TABIYA_CORE_BUNDLE_URL": givenBase},
    )

    # WHEN we refresh
    await registry.refresh()

    # THEN UNAVAILABLE with the HTTP status in the reason
    expectedStatus = PluginStatus.UNAVAILABLE
    entry = await registry.get("tabiya.ner.v1")
    assert entry is not None
    assert entry.status == expectedStatus
    assert "HTTP 500" in (entry.last_error or "")


async def test_refresh_marks_invalid_manifest_unavailable() -> None:
    # GIVEN a plugin that returns a valid HTTP 200 but a body missing required fields
    givenBase = "http://tabiya-core:5010"
    givenManifestUrl = f"{givenBase}/plugin/tabiya.ner.v1/manifest"
    fake_http = _FakeHttp().on(givenManifestUrl, httpx.Response(200, json={"not": "a manifest"}))
    givenCatalog = [
        CatalogEntry(
            plugin_id="tabiya.ner.v1",
            url_env="TABIYA_CORE_BUNDLE_URL",
            path="/plugin/tabiya.ner.v1",
        )
    ]
    registry = PluginRegistry(
        givenCatalog,
        http_client=fake_http,
        env={"TABIYA_CORE_BUNDLE_URL": givenBase},
    )

    # WHEN we refresh
    await registry.refresh()

    # THEN UNAVAILABLE with "invalid manifest" reason
    expectedStatus = PluginStatus.UNAVAILABLE
    entry = await registry.get("tabiya.ner.v1")
    assert entry is not None
    assert entry.status == expectedStatus
    assert "invalid manifest" in (entry.last_error or "")


async def test_refresh_rejects_manifest_whose_plugin_id_does_not_match_catalog() -> None:
    # GIVEN a manifest that declares a different plugin_id
    givenBase = "http://tabiya-core:5010"
    givenManifestUrl = f"{givenBase}/plugin/tabiya.ner.v1/manifest"
    manifestDict = _valid_manifest_dict("tabiya.impostor.v1")
    fake_http = _FakeHttp().on(givenManifestUrl, httpx.Response(200, json=manifestDict))
    givenCatalog = [
        CatalogEntry(
            plugin_id="tabiya.ner.v1",
            url_env="TABIYA_CORE_BUNDLE_URL",
            path="/plugin/tabiya.ner.v1",
        )
    ]
    registry = PluginRegistry(
        givenCatalog,
        http_client=fake_http,
        env={"TABIYA_CORE_BUNDLE_URL": givenBase},
    )

    # WHEN we refresh
    await registry.refresh()

    # THEN UNAVAILABLE with the mismatch called out
    entry = await registry.get("tabiya.ner.v1")
    assert entry is not None
    assert entry.status == PluginStatus.UNAVAILABLE
    assert "plugin_id mismatch" in (entry.last_error or "")


async def test_refresh_rejects_contract_version_major_mismatch() -> None:
    # GIVEN a manifest built against a different major contract version
    givenBase = "http://tabiya-core:5010"
    givenManifestUrl = f"{givenBase}/plugin/tabiya.ner.v1/manifest"
    manifestDict = _valid_manifest_dict("tabiya.ner.v1")
    manifestDict["x-tabiya-contract-version"] = "99.0.0"
    fake_http = _FakeHttp().on(givenManifestUrl, httpx.Response(200, json=manifestDict))
    givenCatalog = [
        CatalogEntry(
            plugin_id="tabiya.ner.v1",
            url_env="TABIYA_CORE_BUNDLE_URL",
            path="/plugin/tabiya.ner.v1",
        )
    ]
    registry = PluginRegistry(
        givenCatalog,
        http_client=fake_http,
        env={"TABIYA_CORE_BUNDLE_URL": givenBase},
    )

    # WHEN we refresh
    await registry.refresh()

    # THEN UNAVAILABLE with the contract-version mismatch called out
    entry = await registry.get("tabiya.ner.v1")
    assert entry is not None
    assert entry.status == PluginStatus.UNAVAILABLE
    assert "contract-version" in (entry.last_error or "").lower()


async def test_get_manifest_raises_when_plugin_not_in_catalog() -> None:
    # GIVEN an empty catalog
    fake_http = _FakeHttp()
    registry = PluginRegistry([], http_client=fake_http, env={})

    # WHEN we ask for a plugin manifest, THEN we get PluginUnreachableError
    with pytest.raises(PluginUnreachableError, match="not in catalog"):
        await registry.get_manifest("tabiya.ner.v1")


async def test_get_manifest_raises_when_plugin_is_unavailable() -> None:
    # GIVEN a catalog entry with no URL configured
    givenCatalog = [
        CatalogEntry(
            plugin_id="tabiya.ner.v1",
            url_env="TABIYA_CORE_BUNDLE_URL",
            path="/plugin/tabiya.ner.v1",
        )
    ]
    fake_http = _FakeHttp()
    registry = PluginRegistry(givenCatalog, http_client=fake_http, env={})
    await registry.refresh()

    # WHEN we ask for the manifest, THEN we get PluginUnreachableError
    with pytest.raises(PluginUnreachableError):
        await registry.get_manifest("tabiya.ner.v1")


async def test_list_manifests_returns_every_catalog_entry_in_declared_order() -> None:
    # GIVEN a mixed catalog
    givenCatalog = _core_ner_and_two_coming_soon()
    fake_http = _FakeHttp()
    registry = PluginRegistry(givenCatalog, http_client=fake_http, env={})

    # WHEN we ask for the list
    resolved = registry.list_manifests()

    # THEN every catalog entry is represented, in order
    expectedIds = [entry.plugin_id for entry in givenCatalog]
    assert [plugin.plugin_id for plugin in resolved] == expectedIds


async def test_refresh_can_flip_a_plugin_from_unavailable_to_enabled_between_cycles() -> None:
    # GIVEN a plugin that first fails, then succeeds
    givenBase = "http://tabiya-core:5010"
    givenManifestUrl = f"{givenBase}/plugin/tabiya.ner.v1/manifest"
    responses = iter(
        [
            httpx.Response(status_code=503),  # first refresh fails
            _valid_manifest_response(),  # second refresh succeeds
        ]
    )
    fake_http = _FakeHttp().on_factory(givenManifestUrl, lambda: next(responses))
    givenCatalog = [
        CatalogEntry(
            plugin_id="tabiya.ner.v1",
            url_env="TABIYA_CORE_BUNDLE_URL",
            path="/plugin/tabiya.ner.v1",
        )
    ]
    registry = PluginRegistry(
        givenCatalog,
        http_client=fake_http,
        env={"TABIYA_CORE_BUNDLE_URL": givenBase},
    )

    # WHEN we refresh twice
    await registry.refresh()
    firstStatus = await registry.get_status("tabiya.ner.v1")
    await registry.refresh()
    secondStatus = await registry.get_status("tabiya.ner.v1")

    # THEN status transitions UNAVAILABLE → ENABLED
    assert firstStatus == PluginStatus.UNAVAILABLE
    assert secondStatus == PluginStatus.ENABLED
