#!/usr/bin/env python3
"""up.py — run pulumi up for a single stack using the Automation API.

Creates the stack if it doesn't exist, selects it if it does.
Config is read from Pulumi.<stack>.yaml and set explicitly on the stack —
the Automation API does not read this file automatically (unlike the CLI).
The Pulumi org is resolved from the active PULUMI_ACCESS_TOKEN — no hardcoding needed.

Usage:
    python iac/scripts/up.py --stack dev --module enable-services
    python iac/scripts/up.py --stack dev --module backend --env-file iac/backend/.env.dev

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

MODULES = [
    "enable-services",
    "dns",
    "auth",
    "backend",
    "common",
    "aws-ns",
]


def _load_stack_config(work_dir: Path, stack: str) -> dict[str, str]:
    """Read Pulumi.<stack>.yaml and return the flat config dict."""
    config_file = work_dir / f"Pulumi.{stack}.yaml"
    if not config_file.exists():
        print(f"error: config file not found: {config_file}", file=sys.stderr)
        sys.exit(1)
    with open(config_file) as f:
        data = yaml.safe_load(f)
    return data.get("config", {})


def run_up(stack: str, module: str, env_file: str | None = None):
    work_dir = IAC_DIR / module

    if not work_dir.is_dir():
        print(f"error: module directory not found: {work_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"info: deploying {module} (stack: {stack})")

    if env_file:
        env_path = REPO_ROOT / env_file
        if not env_path.exists():
            print(f"error: env file not found: {env_path}", file=sys.stderr)
            sys.exit(1)
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path, override=True)
        print(f"info: loaded env vars from {env_path}")

    stack_obj = auto.create_or_select_stack(
        stack_name=stack,
        work_dir=str(work_dir),
    )

    # The Automation API does not read Pulumi.<stack>.yaml automatically.
    # We must set config explicitly from the file before running up.
    config = _load_stack_config(work_dir, stack)
    for key, value in config.items():
        stack_obj.set_config(key, auto.ConfigValue(value=str(value)))
    print(f"info: set {len(config)} config value(s)")

    stack_obj.up(
        on_output=print,
        color="always",
    )

    print(f"info: {module} deployed successfully")


def _main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stack", required=True, help="Pulumi stack name (dev / staging / prod)")
    parser.add_argument("--module", required=True, choices=MODULES, help="IAC module to deploy")
    parser.add_argument("--env-file", help="Path to .env file to source before pulumi up (relative to repo root)")
    args = parser.parse_args()

    run_up(stack=args.stack, module=args.module, env_file=args.env_file)


if __name__ == "__main__":
    _main()
