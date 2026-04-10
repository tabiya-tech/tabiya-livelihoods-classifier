#!/usr/bin/env python3
"""prepare.py — fetch environment config from GCP Secret Manager and write
local files that pulumi up needs.

Usage (called by CI before pulumi up):
    python iac/scripts/prepare.py \\
        --stack dev \\
        --project tabiya-classifier-dev \\
        --ner-image us-central1-docker.pkg.dev/.../ner:SHA \\
        --nel-image us-central1-docker.pkg.dev/.../nel:SHA \\
        --classify-image us-central1-docker.pkg.dev/.../classify:SHA

What it does:
  1. Fetches the "env-vars" secret from GCP Secret Manager in the given project.
  2. Fetches the "stack-config" secret from GCP Secret Manager in the given project.
  3. Validates both against their templates in iac/templates/.
  4. Merges the image URIs into the stack config.
  5. Writes iac/pulumi/.env.{stack}          (git-ignored)
  6. Writes iac/pulumi/Pulumi.{stack}.yaml   (git-ignored)

The GCP project must already be authenticated (via GOOGLE_APPLICATION_CREDENTIALS
or gcloud auth) before this script runs.
"""

import argparse
import os
import sys
import re

import yaml
from dotenv import dotenv_values
from google.cloud import secretmanager
from google.api_core.exceptions import NotFound

IAC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMPLATES_DIR = os.path.join(IAC_DIR, "templates")
PULUMI_DIR = os.path.join(IAC_DIR, "pulumi")

ENV_VARS_SECRET_NAME = "env-vars"
STACK_CONFIG_SECRET_NAME = "stack-config"


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
    template = {
        k: v for k, v in dotenv_values(template_path).items()
        if not k.startswith("#")
    }
    actual = {
        k: v for k, v in dotenv_values(stream=env_content).items()
    }
    if not _validate_against_template(template, actual):
        print("error: env-vars secret does not satisfy the template. Aborting.", file=sys.stderr)
        sys.exit(1)
    print("info: env-vars validated against template.")
    return actual


def _validate_stack_config(config_content: str) -> dict:
    template_path = os.path.join(TEMPLATES_DIR, "stack_config.template.yaml")
    template = yaml.safe_load(open(template_path))
    actual = yaml.safe_load(config_content)
    if not _validate_against_template(template, actual):
        print("error: stack-config secret does not satisfy the template. Aborting.", file=sys.stderr)
        sys.exit(1)
    print("info: stack-config validated against template.")
    return actual


# ── File writers ───────────────────────────────────────────────────────────

def _write_env_file(stack: str, env_content: str) -> str:
    path = os.path.join(PULUMI_DIR, f".env.{stack}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(env_content)
    print(f"info: wrote {path}")
    return path


def _write_pulumi_yaml(stack: str, config: dict, ner_image: str, nel_image: str, classify_image: str):
    # Merge image URIs into the config block — these come from CI, not Secret Manager
    config.setdefault("config", {})
    config["config"]["tabiya-classifier:nerImage"] = ner_image
    config["config"]["tabiya-classifier:nelImage"] = nel_image
    config["config"]["tabiya-classifier:classifyImage"] = classify_image

    path = os.path.join(PULUMI_DIR, f"Pulumi.{stack}.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"info: wrote {path}")


# ── Entry point ────────────────────────────────────────────────────────────

def _main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stack", required=True, help="Pulumi stack name (e.g. dev, staging, prod)")
    parser.add_argument("--project", required=True, help="GCP project ID for this environment")
    parser.add_argument("--ner-image", required=True, help="NER Docker image URI")
    parser.add_argument("--nel-image", required=True, help="NEL Docker image URI")
    parser.add_argument("--classify-image", required=True, help="Classify Docker image URI")
    args = parser.parse_args()

    print(f"info: preparing stack '{args.stack}' from project '{args.project}'")

    # 1. Fetch secrets
    env_content = _fetch_secret(args.project, ENV_VARS_SECRET_NAME)
    stack_config_content = _fetch_secret(args.project, STACK_CONFIG_SECRET_NAME)

    # 2. Validate
    _validate_env_vars(env_content)
    stack_config = _validate_stack_config(stack_config_content)

    # 3. Write files
    _write_env_file(args.stack, env_content)
    _write_pulumi_yaml(args.stack, stack_config, args.ner_image, args.nel_image, args.classify_image)

    print(f"info: preparation complete. Run: cd iac/pulumi && pulumi up --stack {args.stack}")


if __name__ == "__main__":
    _main()
