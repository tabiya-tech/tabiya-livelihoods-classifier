#!/usr/bin/env python3
"""prepare.py — fetch environment config from GCP Secret Manager and write
local files that pulumi up needs for each stack.

Usage (called by CI before pulumi up):
    python iac/scripts/prepare.py \\
        --stack dev \\
        --project tabiya-classifier-dev \\
        --ner-image us-central1-docker.pkg.dev/.../ner:SHA \\
        --nel-image us-central1-docker.pkg.dev/.../nel:SHA \\
        --classify-image us-central1-docker.pkg.dev/.../classify:SHA

What it does:
  1. Fetches "env-vars" from GCP Secret Manager → writes iac/backend/.env.{stack}
     (shared by all stacks that need env vars at pulumi up time)
  2. Fetches one secret per stack ("stack-config-dns", "stack-config-auth", etc.)
  3. Validates each against its template in iac/templates/
  4. Merges image URIs into the backend stack config
  5. Writes Pulumi.{stack}.yaml into each stack's directory

GCP project must be authenticated before running.
"""

import argparse
import os
import re
import sys

import yaml
from dotenv import dotenv_values
from google.cloud import secretmanager
from google.api_core.exceptions import NotFound

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
IAC_DIR = os.path.join(REPO_ROOT, "iac")
TEMPLATES_DIR = os.path.join(IAC_DIR, "templates")

# Logical stack name → (directory, template file, Pulumi project name)
STACKS = {
    "dns":     ("iac/dns",     "stack_config.dns.template.yaml",     "tabiya-classifier-dns"),
    "auth":    ("iac/auth",    "stack_config.auth.template.yaml",    "tabiya-classifier-auth"),
    "backend": ("iac/backend", "stack_config.backend.template.yaml", "tabiya-classifier-backend"),
    "common":  ("iac/common",  "stack_config.common.template.yaml",  "tabiya-classifier-common"),
    "aws-ns":  ("iac/aws-ns",  "stack_config.aws-ns.template.yaml",  "tabiya-classifier-aws-ns"),
}

ENV_VARS_SECRET = "env-vars"


# ── Secret Manager ─────────────────────────────────────────────────────────

def _fetch_secret(project: str, secret_id: str) -> str:
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project}/secrets/{secret_id}/versions/latest"
    try:
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("utf-8")
    except NotFound:
        print(f"error: secret '{secret_id}' not found in project '{project}'.", file=sys.stderr)
        print(f"       Create it with:", file=sys.stderr)
        print(f"         gcloud secrets create {secret_id} --project {project}", file=sys.stderr)
        print(f"         gcloud secrets versions add {secret_id} --project {project} --data-file=-", file=sys.stderr)
        sys.exit(1)


# ── Template validation ────────────────────────────────────────────────────

def _is_regex(value: str) -> bool:
    return value.startswith("/") and value.endswith("/")


def _validate_against_template(template: dict, actual: dict, parent: str = "root") -> bool:
    ok = True
    for key, template_value in template.items():
        actual_value = actual.get(key, "")
        if isinstance(template_value, dict):
            if not isinstance(actual_value, dict):
                print(f"error: {parent}.{key} must be a mapping.")
                ok = False
            else:
                ok = _validate_against_template(template_value, actual_value, f"{parent}.{key}") and ok
        elif isinstance(template_value, str):
            if _is_regex(template_value):
                pattern = re.compile(template_value[1:-1], re.DOTALL)
            else:
                pattern = re.compile(r"[\s\S]+")
            if not re.fullmatch(pattern, str(actual_value)):
                print(f"error: {parent}.{key} value {actual_value!r} does not match expected pattern {template_value!r}.")
                ok = False
    for key in actual:
        if key not in template:
            print(f"warning: {parent}.{key} is present but not in the template (will be passed through).")
    return ok


def _validate_env_vars(env_content: str) -> dict:
    template_path = os.path.join(TEMPLATES_DIR, "env.template")
    template = {k: v for k, v in dotenv_values(template_path).items() if not k.startswith("#")}
    actual = dict(dotenv_values(stream=env_content))
    if not _validate_against_template(template, actual):
        print("error: env-vars secret does not satisfy the template. Aborting.", file=sys.stderr)
        sys.exit(1)
    print("info: env-vars validated.")
    return actual


def _validate_stack_config(logical_stack: str, config_content: str) -> dict:
    _, template_file, _ = STACKS[logical_stack]
    template_path = os.path.join(TEMPLATES_DIR, template_file)
    template = yaml.safe_load(open(template_path))
    actual = yaml.safe_load(config_content)
    if not _validate_against_template(template, actual):
        print(f"error: stack-config-{logical_stack} does not satisfy the template. Aborting.", file=sys.stderr)
        sys.exit(1)
    print(f"info: stack-config-{logical_stack} validated.")
    return actual


# ── File writers ───────────────────────────────────────────────────────────

def _write_env_file(stack: str, env_content: str):
    # Written into backend/ — sourced by all stacks that need env vars
    path = os.path.join(REPO_ROOT, "iac/backend", f".env.{stack}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(env_content)
    print(f"info: wrote {path}")


def _write_pulumi_yaml(
    logical_stack: str,
    stack: str,
    config: dict,
    ner_image: str = "",
    nel_image: str = "",
    classify_image: str = "",
):
    stack_dir, _, _ = STACKS[logical_stack]
    if logical_stack == "backend":
        config.setdefault("config", {})
        config["config"]["tabiya-classifier-backend:nerImage"] = ner_image
        config["config"]["tabiya-classifier-backend:nelImage"] = nel_image
        config["config"]["tabiya-classifier-backend:classifyImage"] = classify_image

    path = os.path.join(REPO_ROOT, stack_dir, f"Pulumi.{stack}.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"info: wrote {path}")


# ── Entry point ────────────────────────────────────────────────────────────

def _main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stack", required=True, help="Pulumi stack name (dev / staging / prod)")
    parser.add_argument("--project", required=True, help="GCP project ID for this environment")
    parser.add_argument("--ner-image", required=True, help="NER Docker image URI")
    parser.add_argument("--nel-image", required=True, help="NEL Docker image URI")
    parser.add_argument("--classify-image", required=True, help="Classify Docker image URI")
    parser.add_argument(
        "--stacks",
        default="all",
        help="Comma-separated logical stacks to prepare (default: all). "
             "Options: dns,auth,backend,common,aws-ns",
    )
    args = parser.parse_args()

    logical_stacks = list(STACKS.keys()) if args.stacks == "all" else args.stacks.split(",")
    for s in logical_stacks:
        if s not in STACKS:
            print(f"error: unknown logical stack '{s}'. Valid options: {list(STACKS)}", file=sys.stderr)
            sys.exit(1)

    print(f"info: preparing stack '{args.stack}' from project '{args.project}'")
    print(f"info: logical stacks: {logical_stacks}")

    # 1. Fetch and write env-vars (written once, shared by all stacks)
    env_content = _fetch_secret(args.project, ENV_VARS_SECRET)
    _validate_env_vars(env_content)
    _write_env_file(args.stack, env_content)

    # 2. Fetch, validate, and write each stack's Pulumi config
    for logical_stack in logical_stacks:
        secret_id = f"stack-config-{logical_stack}"
        config_content = _fetch_secret(args.project, secret_id)
        config = _validate_stack_config(logical_stack, config_content)
        _write_pulumi_yaml(
            logical_stack,
            args.stack,
            config,
            ner_image=args.ner_image,
            nel_image=args.nel_image,
            classify_image=args.classify_image,
        )

    print(f"info: preparation complete for stack '{args.stack}'.")


if __name__ == "__main__":
    _main()
