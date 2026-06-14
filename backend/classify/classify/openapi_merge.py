"""Merge OpenAPI schemas from peer services into a single document.

Classify v1 serves the unified /docs at the gateway. At startup we fetch each
peer's /openapi.json over the internal Cloud Run URL (with a GCP identity
token when running on GCP), then merge them into a single OpenAPI 3 document
that FastAPI exposes via app.openapi_schema.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import httpx

log = logging.getLogger("classify-api.openapi-merge")


# Each peer is a service whose schema we want to fold into the merged doc.
# `tag` is what Swagger UI will group its operations under.
# `prefix` namespaces component schema names so they don't collide.
class PeerSpec:
    def __init__(self, name: str, base_url: str, tag: str, prefix: str):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.tag = tag
        self.prefix = prefix


def _gcp_identity_token(audience: str) -> str | None:
    """Fetch a GCP identity token for the given audience via the metadata server.
    Returns None when running outside GCP."""
    try:
        resp = httpx.get(
            f"http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience={audience}",
            headers={"Metadata-Flavor": "Google"},
            timeout=2.0,
        )
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None


async def fetch_peer_schema(peer: PeerSpec) -> dict | None:
    """Fetch one peer's /openapi.json. Returns None if unreachable — caller logs."""
    headers: dict[str, str] = {}
    token = _gcp_identity_token(peer.base_url)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{peer.base_url}/openapi.json", headers=headers)
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        log.warning("Failed to fetch %s openapi.json from %s: %s", peer.name, peer.base_url, exc)
        return None


async def fetch_all_peer_schemas(peers: list[PeerSpec]) -> dict[str, dict]:
    """Fetch every peer's schema in parallel; missing peers are simply omitted."""
    import asyncio
    results = await asyncio.gather(*(fetch_peer_schema(peer) for peer in peers))
    return {peer.name: schema for peer, schema in zip(peers, results) if schema is not None}


def _rewrite_refs(node: Any, ref_map: dict[str, str]) -> Any:
    """Walk a JSON tree and rewrite every $ref string per ref_map.

    A ref is rewritten when the OLD ref appears as a key in ref_map; the value
    is the NEW ref string. Anything not in the map is left untouched (so
    securitySchemes refs, for instance, survive untouched after we dedupe them).
    """
    if isinstance(node, dict):
        rewritten: dict[str, Any] = {}
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str) and value in ref_map:
                rewritten[key] = ref_map[value]
            else:
                rewritten[key] = _rewrite_refs(value, ref_map)
        return rewritten
    if isinstance(node, list):
        return [_rewrite_refs(item, ref_map) for item in node]
    return node


def _namespace_components(peer: PeerSpec, peer_schema: dict) -> tuple[dict, dict[str, str]]:
    """Prefix every component schema name with the peer's prefix.

    Returns (renamed_schemas_dict, ref_map) where ref_map maps every old
    $ref string to its new $ref string. The peer_schema's paths/components
    must then be walked with _rewrite_refs(ref_map) to fix up references.

    Only the 'schemas' bucket inside components is namespaced. Other buckets
    (securitySchemes, parameters, etc.) are handled separately by the caller.
    """
    components = peer_schema.get("components") or {}
    schemas = components.get("schemas") or {}
    renamed: dict[str, Any] = {}
    ref_map: dict[str, str] = {}
    for name, definition in schemas.items():
        new_name = f"{peer.prefix}_{name}"
        renamed[new_name] = definition
        ref_map[f"#/components/schemas/{name}"] = f"#/components/schemas/{new_name}"
    return renamed, ref_map


def _tag_operations(paths: dict, tag: str) -> dict:
    """Prepend `tag` to the tags list of every operation in `paths`."""
    tagged: dict[str, Any] = {}
    http_methods = {"get", "post", "put", "patch", "delete", "options", "head", "trace"}
    for path, item in paths.items():
        if not isinstance(item, dict):
            tagged[path] = item
            continue
        new_item: dict[str, Any] = {}
        for key, value in item.items():
            if key in http_methods and isinstance(value, dict):
                operation = dict(value)
                existing_tags = operation.get("tags") or []
                operation["tags"] = [tag] + [t for t in existing_tags if t != tag]
                new_item[key] = operation
            else:
                new_item[key] = value
        tagged[path] = new_item
    return tagged


def merge_schemas(base_schema: dict, peers: list[PeerSpec], peer_schemas: dict[str, dict], servers: list[dict] | None) -> dict:
    """Produce a single OpenAPI 3 document combining base + every peer.

    base_schema  — classify v1's own FastAPI-generated OpenAPI dict
    peers        — descriptors (in display order)
    peer_schemas — results of fetch_all_peer_schemas
    servers      — value for the top-level `servers` field (typically the gateway base URL)
    """
    merged = copy.deepcopy(base_schema)

    if servers is not None:
        merged["servers"] = servers

    merged.setdefault("paths", {})
    merged.setdefault("components", {})
    merged["components"].setdefault("schemas", {})
    merged["components"].setdefault("securitySchemes", {})
    merged.setdefault("tags", [])

    classify_v1_tag = "classify v1"
    merged["paths"] = _tag_operations(merged["paths"], classify_v1_tag)
    existing_tag_names = {tag["name"] for tag in merged["tags"] if isinstance(tag, dict) and "name" in tag}
    if classify_v1_tag not in existing_tag_names:
        merged["tags"].append({"name": classify_v1_tag})
        existing_tag_names.add(classify_v1_tag)

    for peer in peers:
        peer_schema = peer_schemas.get(peer.name)
        if not peer_schema:
            continue

        renamed_schemas, ref_map = _namespace_components(peer, peer_schema)

        peer_paths = peer_schema.get("paths") or {}
        peer_paths_rewritten = _rewrite_refs(peer_paths, ref_map)
        peer_paths_tagged = _tag_operations(peer_paths_rewritten, peer.tag)

        for path, item in peer_paths_tagged.items():
            if path in merged["paths"]:
                log.warning("Path collision while merging %s: %s — peer overrides base", peer.name, path)
            merged["paths"][path] = item

        for new_name, definition in renamed_schemas.items():
            merged["components"]["schemas"][new_name] = _rewrite_refs(definition, ref_map)

        peer_security = (peer_schema.get("components") or {}).get("securitySchemes") or {}
        for sec_name, sec_def in peer_security.items():
            merged["components"]["securitySchemes"].setdefault(sec_name, sec_def)

        if peer.tag not in existing_tag_names:
            merged["tags"].append({"name": peer.tag})
            existing_tag_names.add(peer.tag)

    return merged
