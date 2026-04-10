"""Tabiya Livelihoods Classifier — GCP Infrastructure

Stacks: dev, staging, prod

Stack config is fetched from GCP Secret Manager at deploy time by scripts/prepare.py,
which writes a Pulumi.{stack}.yaml file. That file is never committed to git.

Secret values (mongodbUri, hfToken) are loaded from the .env.{stack} file written
by prepare.py and injected into os.environ before pulumi up runs.

Required Pulumi config keys (in the generated Pulumi.{stack}.yaml):
  tabiya-classifier:project       — GCP project ID
  tabiya-classifier:region        — GCP region (default: us-central1)
  tabiya-classifier:nerImage      — NER Docker image URI
  tabiya-classifier:nelImage      — NEL Docker image URI
  tabiya-classifier:classifyImage — Classify Docker image URI
  tabiya-classifier:mongodbDbName — MongoDB database name

Required environment variables (from .env.{stack}, sourced from Secret Manager):
  MONGODB_URI   — MongoDB Atlas connection URI
  HF_TOKEN      — HuggingFace access token
"""

import os

import pulumi

from registry_and_iam import create_artifact_registry
from cloud_run import create_cloud_run_services
from api_gateway import create_api_gateway
from memorystore import create_redis
from storage import create_frontend_buckets
from secrets import create_secrets

config = pulumi.Config()
project = config.require("project")
region = config.get("region") or "us-central1"
mongodb_db_name = config.get("mongodbDbName") or "tabiya-classifier"
ner_image = config.require("nerImage")
nel_image = config.require("nelImage")
classify_image = config.require("classifyImage")

# Secret values come from the .env.{stack} file loaded into the environment
# by prepare.py before pulumi up runs. They are never stored in Pulumi config.
def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Required environment variable {name} is not set. "
                         "Run scripts/prepare.py before pulumi up.")
    return value

mongodb_uri = _require_env("MONGODB_URI")
hf_token = _require_env("HF_TOKEN")

# ── Artifact Registry ──────────────────────────────────────────────────────
registry, service_accounts = create_artifact_registry(project=project, region=region)
pulumi.export(
    "artifactRegistryUrl",
    registry.location.apply(lambda loc: f"{loc}-docker.pkg.dev/{project}/tabiya-classifier"),
)

# ── Secret Manager ─────────────────────────────────────────────────────────
# Creates the Secret resources and pins their initial values. Subsequent rotations
# are managed outside Pulumi (via gcloud or CI) to avoid storing secrets in state.
secrets = create_secrets(project=project, mongodb_uri=mongodb_uri, hf_token=hf_token)

# ── Memorystore Redis ──────────────────────────────────────────────────────
redis, vpc_connector = create_redis(project=project, region=region)

# ── Cloud Run Services ─────────────────────────────────────────────────────
ner, nel, classify = create_cloud_run_services(
    project=project,
    region=region,
    service_accounts=service_accounts,
    ner_image=ner_image,
    nel_image=nel_image,
    classify_image=classify_image,
    hf_token_secret=secrets["hf_token"],
    mongodb_uri_secret=secrets["mongodb_uri"],
    mongodb_db_name=mongodb_db_name,
    redis_host=redis.host,
    vpc_connector=vpc_connector,
)
pulumi.export("nerUrl", ner.uri)
pulumi.export("nelUrl", nel.uri)
pulumi.export("classifyUrl", classify.uri)

# ── API Gateway ────────────────────────────────────────────────────────────
_api, _api_config, gateway = create_api_gateway(
    project=project,
    region=region,
    classify_url=classify.uri,
)
pulumi.export("apiGatewayUrl", gateway.default_hostname)

# ── Frontend Buckets (app + docs) ──────────────────────────────────────────
app_bucket, docs_bucket = create_frontend_buckets(project=project, region=region)
pulumi.export("appBucketUrl", app_bucket.url)
pulumi.export("docsBucketUrl", docs_bucket.url)
