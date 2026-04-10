"""Tabiya Livelihoods Classifier — GCP Infrastructure

Stacks: dev, staging, prod

Required Pulumi config (set via `pulumi config set`):
  tabiya-classifier:project       — GCP project ID
  tabiya-classifier:region        — GCP region (default: us-central1)
  tabiya-classifier:nerImage      — NER Docker image URI
  tabiya-classifier:nelImage      — NEL Docker image URI
  tabiya-classifier:classifyImage — Classify Docker image URI
  tabiya-classifier:mongodbUri    — MongoDB Atlas URI (secret)
  tabiya-classifier:mongodbDbName — MongoDB database name
  tabiya-classifier:hfToken       — HuggingFace token (secret)

Required environment variables (for CI/CD):
  PULUMI_ACCESS_TOKEN             — Pulumi Cloud token
  GOOGLE_CREDENTIALS              — GCP service account JSON key
"""

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
mongodb_uri = config.require_secret("mongodbUri")
mongodb_db_name = config.get("mongodbDbName") or "tabiya-classifier"
hf_token = config.require_secret("hfToken")
ner_image = config.require("nerImage")
nel_image = config.require("nelImage")
classify_image = config.require("classifyImage")

# ── Artifact Registry ──────────────────────────────────────────────────────
registry, service_accounts = create_artifact_registry(project=project, region=region)
pulumi.export(
    "artifactRegistryUrl",
    registry.location.apply(lambda loc: f"{loc}-docker.pkg.dev/{project}/tabiya-classifier"),
)

# ── Secret Manager ─────────────────────────────────────────────────────────
secrets = create_secrets(project=project, mongodb_uri=mongodb_uri, hf_token=hf_token)

# ── Memorystore Redis ──────────────────────────────────────────────────────
redis, vpc_connector = create_redis(project=project, region=region)
pulumi.export("redisHost", redis.host)

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
