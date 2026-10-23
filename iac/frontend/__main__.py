"""Tabiya Livelihoods Classifier — Frontend Stack

Builds the app frontend with environment-specific config injected
from the auth stack outputs, then uploads it to a GCS bucket using the
BucketContent custom resource (efficient diffing, Brotli compression, correct
cache headers).

Required Pulumi config:
  tabiya-classifier-frontend:project    — GCP project ID
  tabiya-classifier-frontend:env        — Stack name (dev / staging / prod)
  tabiya-classifier-frontend:appDomain  — e.g. "app.dev.classifier.tabiya.tech"
  tabiya-classifier-frontend:apiBaseUrl — e.g. "https://dev.classifier.tabiya.tech/api"

Required environment variables (from .env.{stack}):
  APP_DIST_DIR   — absolute path to the built app dist directory
"""

import os
import sys

import pulumi
import pulumi_gcp as gcp

from bucket_content import BucketContent

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PULUMI_ORG = "tabiya-tech"

config = pulumi.Config()
project = config.require("project")
env = config.require("env")
app_domain = config.require("appDomain")
api_base_url = config.require("apiBaseUrl")

app_dist_dir = os.environ.get("APP_DIST_DIR") or os.path.join(REPO_ROOT, "app/dist")

if not os.path.isdir(app_dist_dir):
    print(f"error: APP_DIST_DIR directory not found: {app_dist_dir}", file=sys.stderr)
    sys.exit(1)

# ── Bucket ────────────────────────────────────────────────────────────────

def _make_bucket(name: str) -> gcp.storage.Bucket:
    bucket = gcp.storage.Bucket(
        name,
        project=project,
        name=name,
        location="US",
        uniform_bucket_level_access=True,
        website=gcp.storage.BucketWebsiteArgs(
            main_page_suffix="index.html",
            not_found_page="index.html",
        ),
        cors=[gcp.storage.BucketCorArgs(
            origins=["*"],
            methods=["GET", "HEAD", "OPTIONS"],
            response_headers=["Content-Type"],
            max_age_seconds=3600,
        )],
    )
    gcp.storage.BucketIAMMember(
        f"{name}-public",
        bucket=bucket.name,
        role="roles/storage.objectViewer",
        member="allUsers",
    )
    return bucket

app_bucket = _make_bucket(f"app-{env}-classifier-tabiya-tech")

# ── Upload content ─────────────────────────────────────────────────────────
# index.html must not be cached so users always get the latest shell.
# Vite hashes all other assets in their filenames — those can be cached forever.

BucketContent(
    "app-bucket-content",
    bucket_name=app_bucket.name,
    source_dir=app_dist_dir,
    no_cache_paths=["index.html"],
    opts=pulumi.ResourceOptions(depends_on=[app_bucket]),
)

# ── Exports (consumed by common stack) ────────────────────────────────────
pulumi.export("appBucketName", app_bucket.name)
