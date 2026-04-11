#!/usr/bin/env python3
"""configure.py — create or update GCP Secret Manager secrets for a stack.

Used locally to bootstrap a new environment or update existing config.
Never run in CI — CI uses prepare.py to fetch, not write.

Usage:
    # Interactive: prompts for each value
    python iac/scripts/configure.py --stack dev --project tabiya-classifier-dev

    # Show what secrets exist (doesn't print values)
    python iac/scripts/configure.py --stack dev --project tabiya-classifier-dev --list

    # Upload a specific secret from a local file
    python iac/scripts/configure.py --stack dev --project tabiya-classifier-dev \\
        --secret stack-config-backend --file /tmp/backend.yaml

Secrets managed:
    env-vars                     — MONGODB_URI, HF_TOKEN (shared by all stacks)
    stack-config-enable-services — Pulumi config for the enable-services stack
    stack-config-dns             — Pulumi config for the dns stack
    stack-config-auth            — Pulumi config for the auth stack
    stack-config-backend         — Pulumi config for the backend stack
    stack-config-common          — Pulumi config for the common stack
    stack-config-aws-ns          — Pulumi config for the aws-ns stack
"""

import argparse
import os
import sys
import subprocess
import tempfile

import yaml
from google.cloud import secretmanager
from google.api_core.exceptions import NotFound, AlreadyExists

IAC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMPLATES_DIR = os.path.join(IAC_DIR, "templates")

ALL_SECRETS = [
    "env-vars",
    "stack-config-enable-services",
    "stack-config-dns",
    "stack-config-auth",
    "stack-config-backend",
    "stack-config-frontend",
    "stack-config-common",
    "stack-config-aws-ns",
]

# Maps secret name to its template file (for showing defaults)
TEMPLATE_MAP = {
    "env-vars":                      "env.template",
    "stack-config-enable-services":  "stack_config.enable-services.template.yaml",
    "stack-config-dns":              "stack_config.dns.template.yaml",
    "stack-config-auth":             "stack_config.auth.template.yaml",
    "stack-config-backend":          "stack_config.backend.template.yaml",
    "stack-config-frontend":         "stack_config.frontend.template.yaml",
    "stack-config-common":           "stack_config.common.template.yaml",
    "stack-config-aws-ns":           "stack_config.aws-ns.template.yaml",
}


# ── Secret Manager helpers ─────────────────────────────────────────────────

def _client():
    return secretmanager.SecretManagerServiceClient()


def _secret_path(project: str, secret_id: str) -> str:
    return f"projects/{project}/secrets/{secret_id}"


def _ensure_secret_exists(client, project: str, secret_id: str):
    """Create the secret resource if it doesn't exist yet."""
    try:
        client.create_secret(
            request={
                "parent": f"projects/{project}",
                "secret_id": secret_id,
                "secret": {"replication": {"automatic": {}}},
            }
        )
        print(f"  created secret '{secret_id}'")
    except AlreadyExists:
        pass


def _add_version(client, project: str, secret_id: str, data: str):
    path = _secret_path(project, secret_id)
    client.add_secret_version(
        request={
            "parent": path,
            "payload": {"data": data.encode("utf-8")},
        }
    )
    print(f"  added new version to '{secret_id}'")


def _secret_exists(client, project: str, secret_id: str) -> bool:
    try:
        client.get_secret(request={"name": _secret_path(project, secret_id)})
        return True
    except NotFound:
        return False


def _fetch_latest(client, project: str, secret_id: str) -> str | None:
    try:
        resp = client.access_secret_version(
            request={"name": f"{_secret_path(project, secret_id)}/versions/latest"}
        )
        return resp.payload.data.decode("utf-8")
    except NotFound:
        return None


# ── Editor helpers ─────────────────────────────────────────────────────────

def _open_in_editor(content: str, suffix: str) -> str:
    """Open content in $EDITOR, return the edited content."""
    editor = os.environ.get("EDITOR", "nano")
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
        f.write(content)
        tmpfile = f.name
    try:
        subprocess.run([editor, tmpfile], check=True)
        with open(tmpfile) as f:
            return f.read()
    finally:
        os.unlink(tmpfile)


def _load_template(secret_id: str) -> str:
    template_file = TEMPLATE_MAP.get(secret_id)
    if not template_file:
        return ""
    path = os.path.join(TEMPLATES_DIR, template_file)
    if os.path.exists(path):
        return open(path).read()
    return ""


# ── Commands ───────────────────────────────────────────────────────────────

def cmd_list(project: str):
    client = _client()
    print(f"\nSecrets in project '{project}':\n")
    for secret_id in ALL_SECRETS:
        exists = _secret_exists(client, project, secret_id)
        status = "✓" if exists else "✗ (missing)"
        print(f"  {status}  {secret_id}")
    print()


def cmd_edit(project: str, secret_id: str):
    client = _client()
    print(f"\nEditing '{secret_id}' in project '{project}'...")

    # Start from existing value or template
    existing = _fetch_latest(client, project, secret_id)
    if existing:
        print("  (loaded existing value — editing current version)")
        start_content = existing
    else:
        print("  (no existing value — starting from template)")
        start_content = _load_template(secret_id)

    suffix = ".yaml" if "stack-config" in secret_id else ".env"
    edited = _open_in_editor(start_content, suffix)

    if edited.strip() == start_content.strip():
        print("  no changes made, skipping upload.")
        return

    _ensure_secret_exists(client, project, secret_id)
    _add_version(client, project, secret_id, edited)
    print(f"  done.")


def cmd_upload_file(project: str, secret_id: str, file_path: str):
    if not os.path.exists(file_path):
        print(f"error: file not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    with open(file_path) as f:
        content = f.read()

    client = _client()
    _ensure_secret_exists(client, project, secret_id)
    _add_version(client, project, secret_id, content)
    print(f"Uploaded '{file_path}' to secret '{secret_id}' in project '{project}'.")


def cmd_interactive(project: str, stack: str):
    """Walk through all secrets interactively, editing each one."""
    print(f"\nInteractive configuration for stack '{stack}' in project '{project}'")
    print("You will be prompted to edit each secret in your $EDITOR.\n")

    client = _client()
    for secret_id in ALL_SECRETS:
        exists = _secret_exists(client, project, secret_id)
        status = "exists" if exists else "missing"
        answer = input(f"  [{status}] Edit '{secret_id}'? [y/N] ").strip().lower()
        if answer == "y":
            cmd_edit(project, secret_id)
        else:
            print(f"  skipped '{secret_id}'")

    print(f"\nDone. Run prepare.py to generate local Pulumi config files:\n")
    print(f"  python iac/scripts/prepare.py \\")
    print(f"    --stack {stack} \\")
    print(f"    --project {project} \\")
    print(f"    --ner-image <uri> --nel-image <uri> --classify-image <uri>\n")


# ── Entry point ────────────────────────────────────────────────────────────

def _main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stack", required=True, help="Stack name (dev / staging / prod)")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--list", action="store_true", help="List secrets and their status")
    parser.add_argument("--secret", help="Specific secret to edit/upload")
    parser.add_argument("--file", help="Upload content from this file (use with --secret)")
    args = parser.parse_args()

    if args.list:
        cmd_list(args.project)
    elif args.secret and args.file:
        cmd_upload_file(args.project, args.secret, args.file)
    elif args.secret:
        cmd_edit(args.project, args.secret)
    else:
        cmd_interactive(args.project, args.stack)


if __name__ == "__main__":
    _main()
