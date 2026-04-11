#!/usr/bin/env python3
"""up.py — run pulumi up for one or all modules using the Automation API.

Creates the stack if it doesn't exist, selects it if it does.
Config is read from Pulumi.<stack>.yaml and set explicitly on the stack —
the Automation API does not read this file automatically (unlike the CLI).

The env file (iac/backend/.env.{stack}) is loaded once and applies to all
modules that need it (backend, frontend).

Usage:
    # Deploy a single module
    python iac/scripts/up.py --stack dev --module backend

    # Deploy all modules in order
    python iac/scripts/up.py --stack dev

    # Deploy all modules, passing Firebase outputs to the frontend build
    APP_DIST_DIR=app/dist DOCS_DIST_DIR=docs/dist \\
        python iac/scripts/up.py --stack dev

Must be run from the repo root.
"""

import argparse
import os
import sys
from pathlib import Path

import yaml
import pulumi.automation as auto

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
IAC_DIR = REPO_ROOT / "iac"

# Deployment order matters — each module may depend on outputs from the previous.
DEPLOY_ORDER = [
    "enable-services",
    "dns",
    "auth",
    "backend",
    "frontend",  # must come before common (common has a StackReference to frontend)
    "common",
    "aws-ns",
]


def _load_stack_config(work_dir: Path, stack: str) -> dict:
    config_file = work_dir / f"Pulumi.{stack}.yaml"
    if not config_file.exists():
        print(f"error: config file not found: {config_file}", file=sys.stderr)
        sys.exit(1)
    with open(config_file) as f:
        data = yaml.safe_load(f)
    return data.get("config", {})


def _load_env_file(stack: str):
    env_path = IAC_DIR / "backend" / f".env.{stack}"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path, override=True)
        print(f"info: loaded env vars from {env_path}")
    else:
        print(f"warning: env file not found: {env_path}", file=sys.stderr)


def run_up(stack: str, module: str):
    work_dir = IAC_DIR / module
    if not work_dir.is_dir():
        print(f"error: module directory not found: {work_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\ninfo: ── deploying {module} (stack: {stack}) ──")

    stack_obj = auto.create_or_select_stack(
        stack_name=stack,
        work_dir=str(work_dir),
    )

    config = _load_stack_config(work_dir, stack)
    for key, value in config.items():
        stack_obj.set_config(key, auto.ConfigValue(value=str(value)))
    print(f"info: set {len(config)} config value(s)")

    stack_obj.up(on_output=print, color="always")
    print(f"info: {module} deployed successfully")


def _main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stack", required=True, help="Pulumi stack name (dev / staging / prod)")
    parser.add_argument(
        "--module",
        choices=DEPLOY_ORDER,
        help="Single module to deploy. Omit to deploy all modules in order.",
    )
    args = parser.parse_args()

    # Always load the env file — modules that don't need it simply ignore the vars.
    _load_env_file(args.stack)

    modules = [args.module] if args.module else DEPLOY_ORDER
    for module in modules:
        run_up(stack=args.stack, module=module)

    print(f"\ninfo: all done ({', '.join(modules)})")


if __name__ == "__main__":
    _main()
