#!/usr/bin/env python3
"""set_stack_config_key.py — set a single key in a stack-config Secret Manager
secret, non-interactively.

`configure.py` edits a whole stack-config secret in $EDITOR. This is the
targeted alternative: fetch the current stack-config, set/update ONE key
(preserving every other key), show a redacted diff, and — only with --apply —
add a new secret version.

Why this exists: some config (e.g. `minInstances`) lives in the
`stack-config-backend` secret, NOT in the gitignored, generated Pulumi.<stack>.yaml.
Editing the local yaml doesn't persist — CI's prepare.py regenerates it from
the secret. This writes the durable source of truth.

Usage:
    # Dry-run (default): show what would change, write nothing
    python iac/scripts/set_stack_config_key.py \\
        --project classifier-dev-492912 \\
        --key tabiya-classifier-backend:minInstances --value 0

    # Apply: add a new secret version
    python iac/scripts/set_stack_config_key.py \\
        --project classifier-dev-492912 \\
        --key tabiya-classifier-backend:minInstances --value 0 --apply

    # Target a different stack-config secret (default: stack-config-backend)
    python iac/scripts/set_stack_config_key.py ... --secret stack-config-frontend

After applying, run prepare.py + pulumi up so the new value reaches Cloud Run.
"""

import argparse
import sys

import yaml
from google.cloud import secretmanager
from google.api_core.exceptions import NotFound


def _client():
    return secretmanager.SecretManagerServiceClient()


def _secret_path(project: str, secret_id: str) -> str:
    return f"projects/{project}/secrets/{secret_id}"


def _fetch_latest(client, project: str, secret_id: str) -> str:
    name = f"{_secret_path(project, secret_id)}/versions/latest"
    try:
        resp = client.access_secret_version(request={"name": name})
        return resp.payload.data.decode("utf-8")
    except NotFound:
        print(
            f"error: secret '{secret_id}' not found in project '{project}'. "
            f"Create it first with configure.py.",
            file=sys.stderr,
        )
        sys.exit(1)


def _add_version(client, project: str, secret_id: str, data: str):
    client.add_secret_version(
        request={
            "parent": _secret_path(project, secret_id),
            "payload": {"data": data.encode("utf-8")},
        }
    )


def _main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument(
        "--secret",
        default="stack-config-backend",
        help="Stack-config secret id (default: stack-config-backend)",
    )
    parser.add_argument(
        "--key",
        required=True,
        help="Config key to set, e.g. tabiya-classifier-backend:minInstances",
    )
    parser.add_argument(
        "--value",
        required=True,
        help="Value to set. Pulumi config values are strings — it is written "
        "quoted in the YAML.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Add a new secret version. Without this, runs as a dry-run.",
    )
    args = parser.parse_args()

    client = _client()

    # Fetch + parse the current stack-config secret.
    current_raw = _fetch_latest(client, args.project, args.secret)
    parsed = yaml.safe_load(current_raw) or {}
    config = parsed.setdefault("config", {})

    old_value = config.get(args.key, "<unset>")
    # Pulumi config values are strings; store as a string so YAML emits it
    # quoted (matches how numeric-looking config like minInstances is stored).
    new_value = str(args.value)

    if str(old_value) == new_value:
        print(
            f"no change: {args.key} is already {new_value!r} in "
            f"'{args.secret}'. Nothing to do."
        )
        return

    config[args.key] = new_value
    updated_raw = yaml.dump(parsed, default_flow_style=False, allow_unicode=True)

    print(f"secret : {args.secret} (project {args.project})")
    print(f"key    : {args.key}")
    print(f"change : {old_value!r} -> {new_value!r}")
    print(f"keys in config after update: {len(config)}")

    if not args.apply:
        print(
            "\ndry-run — no version written. Re-run with --apply to add a new "
            "secret version."
        )
        return

    _add_version(client, args.project, args.secret, updated_raw)
    print(f"\napplied: added a new version to '{args.secret}'.")
    print(
        "next: run prepare.py to regenerate Pulumi.<stack>.yaml, then "
        "`pulumi up` in iac/backend to roll it out."
    )


if __name__ == "__main__":
    _main()
