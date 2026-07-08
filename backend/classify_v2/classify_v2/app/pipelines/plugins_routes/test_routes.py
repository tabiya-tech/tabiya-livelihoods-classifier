"""Tests for /v2/plugins read-only routes."""

from __future__ import annotations

from typing import Any, Callable

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tabiya_plugin_contracts import (
    CONTRACT_VERSION,
    Manifest,
    PluginCategory,
    Slot,
    SlotType,
)

from classify_v2.app.pipelines.plugins_routes.routes import (
    get_plugin_http,
    get_plugin_registry,
    router as plugins_router,
)
from classify_v2.app.pipelines.registry import (
    CatalogEntry,
    PluginRegistry,
    PluginStatus,
)


class _FakeHttp:
    """Test double for httpx.AsyncClient — pattern shared with registry tests."""

    def __init__(self) -> None:
        self._responses: dict[str, Callable[[], httpx.Response]] = {}
        self._exceptions: dict[str, Exception] = {}
        self.calls: list[str] = []

    def on(self, url: str, response: httpx.Response) -> "_FakeHttp":
        self._responses[url] = lambda: response
        return self

    def raise_on(self, url: str, exc: Exception) -> "_FakeHttp":
        self._exceptions[url] = exc
        return self

    async def get(self, url: str, timeout: float | None = None) -> httpx.Response:
        self.calls.append(url)
        if url in self._exceptions:
            raise self._exceptions[url]
        if url in self._responses:
            return self._responses[url]()
        return httpx.Response(status_code=404, text="not configured for test")


def _ner_manifest_dict(x_source: str = "/v2/plugins/tabiya.ner.v1/options/model_id") -> dict:
    manifest = Manifest(
        plugin_id="tabiya.ner.v1",
        name="Tabiya NER",
        version="0.1.0",
        category=PluginCategory.CORE,
        summary="Named entity recognition over job ad prose.",
        icon="ner",
        input_slot=Slot(type=SlotType.RAW_TEXT),
        output_slot=Slot(type=SlotType.ENTITIES),
        config_schema={
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "string",
                    "title": "Model",
                    "x-source": x_source,
                },
                "entity_types": {"type": "array", "items": {"type": "string"}},
            },
        },
        timeout_ms=15_000,
        **{"x-tabiya-contract-version": CONTRACT_VERSION},
    )
    return manifest.model_dump(by_alias=True, exclude_none=True)


def _local_mode_env(monkeypatch) -> None:
    monkeypatch.setenv("TARGET_ENVIRONMENT_TYPE", "local")


async def _build_client(
    catalog: list[CatalogEntry],
    env: dict[str, str],
    fake_http: _FakeHttp,
    monkeypatch,
    do_initial_refresh: bool = True,
) -> tuple[TestClient, PluginRegistry]:
    """Assemble a minimal FastAPI app with the plugins router mounted."""

    _local_mode_env(monkeypatch)
    registry = PluginRegistry(catalog, http_client=fake_http, env=env)
    if do_initial_refresh:
        await registry.refresh()

    app = FastAPI()
    app.state.plugin_registry = registry
    app.state.plugin_http = fake_http
    app.include_router(plugins_router)
    # In-app overrides ensure the fake HTTP client is used even when the
    # dependency reaches back to app.state.
    app.dependency_overrides[get_plugin_registry] = lambda: registry
    app.dependency_overrides[get_plugin_http] = lambda: fake_http
    return TestClient(app), registry


CORE_URL = "http://tabiya-core:5010"
IO_URL = "http://tabiya-io:5011"


def _six_entry_catalog() -> list[CatalogEntry]:
    return [
        CatalogEntry(
            plugin_id="tabiya.ner.v1",
            url_env="TABIYA_CORE_BUNDLE_URL",
            path="/plugin/tabiya.ner.v1",
        ),
        CatalogEntry(
            plugin_id="tabiya.source.text.v1",
            url_env="TABIYA_IO_BUNDLE_URL",
            path="/plugin/tabiya.source.text.v1",
        ),
        CatalogEntry(plugin_id="tabiya.source.scraper.v1", coming_soon=True),
    ]


def _env_with_bundles() -> dict[str, str]:
    return {
        "TABIYA_CORE_BUNDLE_URL": CORE_URL,
        "TABIYA_IO_BUNDLE_URL": IO_URL,
    }


def _seed_manifests(fake_http: _FakeHttp) -> None:
    fake_http.on(
        f"{CORE_URL}/plugin/tabiya.ner.v1/manifest",
        httpx.Response(200, json=_ner_manifest_dict()),
    )
    text_manifest = Manifest(
        plugin_id="tabiya.source.text.v1",
        name="Text Input",
        version="0.1.0",
        category=PluginCategory.SOURCE,
        summary="Feeds raw text into the pipeline.",
        icon="text",
        input_slot=Slot(type=SlotType.NONE, cardinality="none"),
        output_slot=Slot(type=SlotType.RAW_TEXT),
        config_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
        },
        timeout_ms=5_000,
        **{"x-tabiya-contract-version": CONTRACT_VERSION},
    )
    fake_http.on(
        f"{IO_URL}/plugin/tabiya.source.text.v1/manifest",
        httpx.Response(200, json=text_manifest.model_dump(by_alias=True, exclude_none=True)),
    )


async def test_list_returns_one_entry_per_catalog_row(monkeypatch) -> None:
    # GIVEN a three-entry catalog with two live and one coming_soon
    givenCatalog = _six_entry_catalog()
    fake_http = _FakeHttp()
    _seed_manifests(fake_http)
    client, _ = await _build_client(givenCatalog, _env_with_bundles(), fake_http, monkeypatch)

    # WHEN we list plugins
    response = client.get("/v2/plugins")

    # THEN each catalog entry appears
    expectedStatus = 200
    expectedIds = [entry.plugin_id for entry in givenCatalog]
    assert response.status_code == expectedStatus
    body = response.json()
    assert [plugin["plugin_id"] for plugin in body["plugins"]] == expectedIds


async def test_list_projects_status_for_each_plugin(monkeypatch) -> None:
    # GIVEN a catalog with one enabled, one coming_soon
    givenCatalog = _six_entry_catalog()
    fake_http = _FakeHttp()
    _seed_manifests(fake_http)
    client, _ = await _build_client(givenCatalog, _env_with_bundles(), fake_http, monkeypatch)

    # WHEN we list
    response = client.get("/v2/plugins")

    # THEN status matches what the registry decided
    plugins_by_id = {plugin["plugin_id"]: plugin for plugin in response.json()["plugins"]}
    assert plugins_by_id["tabiya.ner.v1"]["status"] == PluginStatus.ENABLED.value
    assert plugins_by_id["tabiya.source.scraper.v1"]["status"] == PluginStatus.UNAVAILABLE.value
    assert plugins_by_id["tabiya.source.scraper.v1"]["coming_soon"] is True


async def test_list_returns_placeholder_summary_when_manifest_not_loaded(monkeypatch) -> None:
    # GIVEN a plugin whose manifest fetch failed
    givenCatalog = [
        CatalogEntry(
            plugin_id="tabiya.ner.v1",
            url_env="TABIYA_CORE_BUNDLE_URL",
            path="/plugin/tabiya.ner.v1",
        )
    ]
    fake_http = _FakeHttp().raise_on(
        f"{CORE_URL}/plugin/tabiya.ner.v1/manifest",
        httpx.ConnectError("refused"),
    )
    client, _ = await _build_client(givenCatalog, _env_with_bundles(), fake_http, monkeypatch)

    # WHEN we list
    response = client.get("/v2/plugins")

    # THEN we still get an entry with the plugin_id as fallback name + last_error surfaced
    assert response.status_code == 200
    body = response.json()
    assert len(body["plugins"]) == 1
    assert body["plugins"][0]["plugin_id"] == "tabiya.ner.v1"
    assert body["plugins"][0]["name"] == "tabiya.ner.v1"
    assert body["plugins"][0]["status"] == PluginStatus.UNAVAILABLE.value
    assert "unreachable" in body["plugins"][0]["last_error"]


async def test_detail_returns_full_manifest_when_enabled(monkeypatch) -> None:
    # GIVEN an enabled plugin
    givenCatalog = _six_entry_catalog()
    fake_http = _FakeHttp()
    _seed_manifests(fake_http)
    client, _ = await _build_client(givenCatalog, _env_with_bundles(), fake_http, monkeypatch)

    # WHEN we fetch the detail
    response = client.get("/v2/plugins/tabiya.ner.v1")

    # THEN the full manifest comes back with the runtime status
    expectedStatus = 200
    assert response.status_code == expectedStatus
    body = response.json()
    assert body["status"] == PluginStatus.ENABLED.value
    assert body["manifest"]["plugin_id"] == "tabiya.ner.v1"
    assert body["manifest"]["config_schema"]["properties"]["model_id"]["x-source"] == (
        "/v2/plugins/tabiya.ner.v1/options/model_id"
    )


async def test_detail_returns_404_when_plugin_not_in_catalog(monkeypatch) -> None:
    # GIVEN an empty catalog
    fake_http = _FakeHttp()
    client, _ = await _build_client([], {}, fake_http, monkeypatch, do_initial_refresh=False)

    # WHEN we ask for an unknown plugin
    response = client.get("/v2/plugins/tabiya.ghost.v1")

    # THEN 404
    expectedStatus = 404
    assert response.status_code == expectedStatus


async def test_detail_for_coming_soon_returns_null_manifest_with_unavailable_status(
    monkeypatch,
) -> None:
    # GIVEN a coming_soon entry in the catalog
    givenCatalog = [CatalogEntry(plugin_id="tabiya.source.scraper.v1", coming_soon=True)]
    fake_http = _FakeHttp()
    client, _ = await _build_client(givenCatalog, {}, fake_http, monkeypatch)

    # WHEN we hit the detail endpoint
    response = client.get("/v2/plugins/tabiya.source.scraper.v1")

    # THEN we get a 200 with manifest=null and status=UNAVAILABLE
    expectedStatus = 200
    assert response.status_code == expectedStatus
    body = response.json()
    assert body["manifest"] is None
    assert body["status"] == PluginStatus.UNAVAILABLE.value
    assert body["coming_soon"] is True


async def test_options_proxies_upstream_and_returns_normalised_items(monkeypatch) -> None:
    # GIVEN a plugin whose model_id x-source points at an in-app endpoint
    xSource = "/v2/nel/models"
    fake_http = _FakeHttp()
    fake_http.on(
        f"{CORE_URL}/plugin/tabiya.ner.v1/manifest",
        httpx.Response(200, json=_ner_manifest_dict(x_source=xSource)),
    )
    # And that endpoint returns nel-shaped model list
    fake_http.on(
        f"http://testserver{xSource}",
        httpx.Response(
            200,
            json=[
                {"id": "model-1", "name": "All MiniLM"},
                {"id": "model-2", "name": "Sentence-BERT"},
            ],
        ),
    )
    givenCatalog = [
        CatalogEntry(
            plugin_id="tabiya.ner.v1",
            url_env="TABIYA_CORE_BUNDLE_URL",
            path="/plugin/tabiya.ner.v1",
        )
    ]
    client, _ = await _build_client(givenCatalog, _env_with_bundles(), fake_http, monkeypatch)

    # WHEN we hit the options endpoint
    response = client.get("/v2/plugins/tabiya.ner.v1/options/model_id")

    # THEN the response is normalised into {value, label}
    expectedStatus = 200
    assert response.status_code == expectedStatus
    body = response.json()
    assert body["field"] == "model_id"
    assert body["options"] == [
        {"value": "model-1", "label": "All MiniLM"},
        {"value": "model-2", "label": "Sentence-BERT"},
    ]


async def test_options_accepts_already_shaped_value_label_payload(monkeypatch) -> None:
    # GIVEN an upstream that already returns {value, label} entries
    xSource = "http://external.example.com/some-options"
    fake_http = _FakeHttp()
    fake_http.on(
        f"{CORE_URL}/plugin/tabiya.ner.v1/manifest",
        httpx.Response(200, json=_ner_manifest_dict(x_source=xSource)),
    )
    fake_http.on(
        xSource,
        httpx.Response(
            200,
            json=[
                {"value": "en", "label": "English"},
                {"value": "fr", "label": "French"},
            ],
        ),
    )
    givenCatalog = [
        CatalogEntry(
            plugin_id="tabiya.ner.v1",
            url_env="TABIYA_CORE_BUNDLE_URL",
            path="/plugin/tabiya.ner.v1",
        )
    ]
    client, _ = await _build_client(givenCatalog, _env_with_bundles(), fake_http, monkeypatch)

    # WHEN we hit options
    response = client.get("/v2/plugins/tabiya.ner.v1/options/model_id")

    # THEN the response passes {value, label} through unchanged
    assert response.status_code == 200
    body = response.json()
    assert body["options"] == [
        {"value": "en", "label": "English"},
        {"value": "fr", "label": "French"},
    ]


async def test_options_returns_404_for_field_without_x_source(monkeypatch) -> None:
    # GIVEN a manifest whose entity_types field has no x-source
    fake_http = _FakeHttp()
    _seed_manifests(fake_http)
    givenCatalog = _six_entry_catalog()
    client, _ = await _build_client(givenCatalog, _env_with_bundles(), fake_http, monkeypatch)

    # WHEN we ask for options of a field that has no x-source
    response = client.get("/v2/plugins/tabiya.ner.v1/options/entity_types")

    # THEN 404 with a helpful message
    expectedStatus = 404
    assert response.status_code == expectedStatus
    assert "x-source" in response.json()["detail"]


async def test_options_returns_404_for_unknown_field(monkeypatch) -> None:
    # GIVEN a plugin with a config_schema but not the requested field
    fake_http = _FakeHttp()
    _seed_manifests(fake_http)
    givenCatalog = _six_entry_catalog()
    client, _ = await _build_client(givenCatalog, _env_with_bundles(), fake_http, monkeypatch)

    # WHEN we ask for an unknown field
    response = client.get("/v2/plugins/tabiya.ner.v1/options/does_not_exist")

    # THEN 404
    expectedStatus = 404
    assert response.status_code == expectedStatus


async def test_options_returns_404_when_plugin_is_unavailable(monkeypatch) -> None:
    # GIVEN a coming_soon plugin
    givenCatalog = [CatalogEntry(plugin_id="tabiya.source.scraper.v1", coming_soon=True)]
    fake_http = _FakeHttp()
    client, _ = await _build_client(givenCatalog, {}, fake_http, monkeypatch)

    # WHEN we ask for options
    response = client.get("/v2/plugins/tabiya.source.scraper.v1/options/anything")

    # THEN 404 — we don't proxy for plugins that aren't ENABLED
    expectedStatus = 404
    assert response.status_code == expectedStatus


async def test_options_maps_upstream_5xx_to_502(monkeypatch) -> None:
    # GIVEN an upstream that fails with 500
    xSource = "/v2/nel/models"
    fake_http = _FakeHttp()
    fake_http.on(
        f"{CORE_URL}/plugin/tabiya.ner.v1/manifest",
        httpx.Response(200, json=_ner_manifest_dict(x_source=xSource)),
    )
    fake_http.on(f"http://testserver{xSource}", httpx.Response(500))
    givenCatalog = [
        CatalogEntry(
            plugin_id="tabiya.ner.v1",
            url_env="TABIYA_CORE_BUNDLE_URL",
            path="/plugin/tabiya.ner.v1",
        )
    ]
    client, _ = await _build_client(givenCatalog, _env_with_bundles(), fake_http, monkeypatch)

    # WHEN we hit options
    response = client.get("/v2/plugins/tabiya.ner.v1/options/model_id")

    # THEN 502
    expectedStatus = 502
    assert response.status_code == expectedStatus
    assert "HTTP 500" in response.json()["detail"]


async def test_options_maps_upstream_transport_error_to_502(monkeypatch) -> None:
    # GIVEN an upstream that raises a ConnectError
    xSource = "/v2/nel/models"
    fake_http = _FakeHttp()
    fake_http.on(
        f"{CORE_URL}/plugin/tabiya.ner.v1/manifest",
        httpx.Response(200, json=_ner_manifest_dict(x_source=xSource)),
    )
    fake_http.raise_on(f"http://testserver{xSource}", httpx.ConnectError("boom"))
    givenCatalog = [
        CatalogEntry(
            plugin_id="tabiya.ner.v1",
            url_env="TABIYA_CORE_BUNDLE_URL",
            path="/plugin/tabiya.ner.v1",
        )
    ]
    client, _ = await _build_client(givenCatalog, _env_with_bundles(), fake_http, monkeypatch)

    # WHEN we hit options
    response = client.get("/v2/plugins/tabiya.ner.v1/options/model_id")

    # THEN 502
    expectedStatus = 502
    assert response.status_code == expectedStatus


async def test_options_maps_string_list_upstream_to_value_equals_label(monkeypatch) -> None:
    # GIVEN an upstream that returns a bare list of strings
    xSource = "/v2/nel/models"
    fake_http = _FakeHttp()
    fake_http.on(
        f"{CORE_URL}/plugin/tabiya.ner.v1/manifest",
        httpx.Response(200, json=_ner_manifest_dict(x_source=xSource)),
    )
    fake_http.on(
        f"http://testserver{xSource}",
        httpx.Response(200, json=["alpha", "beta"]),
    )
    givenCatalog = [
        CatalogEntry(
            plugin_id="tabiya.ner.v1",
            url_env="TABIYA_CORE_BUNDLE_URL",
            path="/plugin/tabiya.ner.v1",
        )
    ]
    client, _ = await _build_client(givenCatalog, _env_with_bundles(), fake_http, monkeypatch)

    # WHEN we hit options
    response = client.get("/v2/plugins/tabiya.ner.v1/options/model_id")

    # THEN each string becomes {value: "s", label: "s"}
    assert response.status_code == 200
    body = response.json()
    assert body["options"] == [
        {"value": "alpha", "label": "alpha"},
        {"value": "beta", "label": "beta"},
    ]
