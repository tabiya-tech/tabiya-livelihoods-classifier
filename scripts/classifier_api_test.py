#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone classifier HTTP test — no packages from this repo, no pip required.

Depends only on Python 3.8+ stdlib. Optionally uses ``certifi`` if installed (helps macOS TLS).

  python3 classifier_api_test.py
  python3 classifier_api_test.py --url https://OTHER.run.app
  python3 classifier_api_test.py --payload-file body.json

Environment (optional overrides): CLASSIFIER_URL, CLASSIFIER_BEARER, CLASSIFIER_SMOKE_TIMEOUT, CLASSIFIER_INSECURE_SSL
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from typing import Any

# Default Cloud Run base (no trailing slash). Override with --url or CLASSIFIER_URL.
DEFAULT_CLASSIFIER_BASE_URL = "https://livelihoods-classifier-api-kntejm5pra-uc.a.run.app"


def _ssl_context(*, insecure: bool) -> ssl.SSLContext:
    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    try:
        import certifi  # type: ignore[import-not-found]

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def http_json(
    method: str,
    url: str,
    *,
    json_body: dict[str, Any] | None = None,
    bearer: str | None = None,
    timeout: float,
    insecure: bool,
) -> Any:
    payload = None
    headers = {"Accept": "application/json"}
    if json_body is not None:
        payload = json.dumps(json_body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if bearer:
        headers["Authorization"] = f"Bearer {bearer.strip()}"

    req = urllib.request.Request(url, data=payload, headers=headers, method=method)
    ctx = _ssl_context(insecure=insecure)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            text = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise SystemExit(
            f"Request failed: HTTP {e.code} {e.reason}\n--- body ---\n{detail}\n--- end ---"
        ) from e
    except urllib.error.URLError as e:
        raise SystemExit(f"Connection error: {e.reason}") from e

    if not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _load_payload(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit("--payload-file must contain a JSON object (e.g. {\"text\": \"...\"})")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Call Tabiya classifier /v1/health and /v1/classify (standalone test)."
    )
    parser.add_argument(
        "--url",
        default=(os.environ.get("CLASSIFIER_URL") or DEFAULT_CLASSIFIER_BASE_URL).rstrip("/"),
        help="Service base URL (default: baked-in dev URL; env CLASSIFIER_URL overrides)",
    )
    parser.add_argument(
        "--bearer",
        default=os.environ.get("CLASSIFIER_BEARER"),
        help="Optional Bearer token (CLASSIFIER_BEARER) for IAM-only services",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("CLASSIFIER_SMOKE_TIMEOUT", "600")),
        help="HTTP timeout seconds per request (default: 600)",
    )
    parser.add_argument(
        "--text",
        default="Head Chef with menu planning.",
        help='Used as {"text": ...} when --payload-file is omitted',
    )
    parser.add_argument(
        "--payload-file",
        metavar="PATH",
        help="JSON object file for POST /v1/classify body (overrides --text)",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification (macOS python.org CA issues; testing only)",
    )
    args = parser.parse_args()
    insecure = args.insecure or (
        os.environ.get("CLASSIFIER_INSECURE_SSL", "").strip().lower() in ("1", "true", "yes")
    )
    if insecure:
        print(
            "WARNING: TLS verification is OFF (--insecure). Use only for local smoke tests.\n",
            file=sys.stderr,
        )

    base = args.url.rstrip("/")
    health_url = f"{base}/v1/health"
    classify_url = f"{base}/v1/classify"

    body: dict[str, Any]
    if args.payload_file:
        body = _load_payload(args.payload_file)
    else:
        body = {"text": args.text}

    print(f"Base URL: {base}\nTimeout:  {args.timeout}s\n", flush=True)

    print("→ GET /v1/health", flush=True)
    health = http_json(
        "GET", health_url, bearer=args.bearer, timeout=args.timeout, insecure=insecure
    )
    print(json.dumps(health, indent=2, ensure_ascii=False))

    print("\n→ POST /v1/classify", flush=True)
    print(json.dumps({"request_body": body}, indent=2, ensure_ascii=False), flush=True)
    result = http_json(
        "POST",
        classify_url,
        json_body=body,
        bearer=args.bearer,
        timeout=args.timeout,
        insecure=insecure,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\nDone — both calls succeeded.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
