"""Tabiya Livelihoods Classifier — Backend Stack

Deploys:
  - Artifact Registry + service accounts
  - Secret Manager secrets (mongodb_uri, hf_token)
  - Cloud Run services (NER, NEL, Classify)
  - API Gateway

Required Pulumi config:
  tabiya-classifier-backend:project           — GCP project ID
  tabiya-classifier-backend:region            — GCP region (default: us-central1)
  tabiya-classifier-backend:env               — Stack name (dev / staging / prod)
  tabiya-classifier-backend:firebaseProjectId — Firebase project ID for dashboard auth
  tabiya-classifier-backend:envSubdomain      — e.g. "dev.classifier.tabiya.tech"
  tabiya-classifier-backend:nerImage          — NER Docker image URI (injected by CI)
  tabiya-classifier-backend:nelImage          — NEL Docker image URI (injected by CI)
  tabiya-classifier-backend:classifyImage     — Classify Docker image URI (injected by CI)

Required environment variables (from .env.{stack}, sourced from Secret Manager):
  MONGODB_URI      — MongoDB Atlas connection URI
  MONGODB_DB_NAME  — MongoDB database name
  HF_TOKEN         — HuggingFace access token
"""

import os

import pulumi
import pulumi_gcp as gcp

from registry_and_iam import create_artifact_registry
from cloud_run import create_cloud_run_services
from api_gateway import create_api_gateway
from secrets import create_secrets

config = pulumi.Config()
project = config.require("project")
region = config.get("region") or "us-central1"
env = config.require("env")
firebase_project_id = config.require("firebaseProjectId")
env_subdomain = config.require("envSubdomain")
ner_image = config.require("nerImage")
nel_image = config.require("nelImage")
classify_image = config.require("classifyImage")


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(
            f"Required environment variable {name} is not set. "
            "Run iac/scripts/prepare.py before pulumi up."
        )
    return value


mongodb_uri = _require_env("MONGODB_URI")
mongodb_db_name = _require_env("MONGODB_DB_NAME")
hf_token = _require_env("HF_TOKEN")

# ── Artifact Registry ──────────────────────────────────────────────────────
registry, service_accounts = create_artifact_registry(project=project, region=region)
pulumi.export(
    "artifactRegistryUrl",
    registry.location.apply(lambda loc: f"{loc}-docker.pkg.dev/{project}/tabiya-classifier"),
)

# ── Secret Manager ─────────────────────────────────────────────────────────
secrets = create_secrets(project=project, mongodb_uri=mongodb_uri, hf_token=hf_token)

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
    firebase_project_id=firebase_project_id,
)
pulumi.export("nerUrl", ner.uri)
pulumi.export("nelUrl", nel.uri)
pulumi.export("classifyUrl", classify.uri)

# ── API Gateway ────────────────────────────────────────────────────────────
_api, _api_config, gateway, gateway_sa = create_api_gateway(
    project=project,
    region=region,
    classify_url=classify.uri,
    ner_url=ner.uri,
    nel_url=nel.uri,
    firebase_project_id=firebase_project_id,
    env_subdomain=env_subdomain,
)
pulumi.export("apiGatewayUrl", gateway.default_hostname)
pulumi.export("apiGatewayId", gateway.gateway_id)

# Allow the gateway service account to invoke all three services.
gcp.cloudrunv2.ServiceIamMember(
    "classify-invoker",
    project=project,
    location=region,
    name=classify.name,
    role="roles/run.invoker",
    member=gateway_sa.email.apply(lambda e: f"serviceAccount:{e}"),
    opts=pulumi.ResourceOptions(depends_on=[gateway_sa]),
)
gcp.cloudrunv2.ServiceIamMember(
    "ner-gw-invoker",
    project=project,
    location=region,
    name=ner.name,
    role="roles/run.invoker",
    member=gateway_sa.email.apply(lambda e: f"serviceAccount:{e}"),
    opts=pulumi.ResourceOptions(depends_on=[gateway_sa]),
)
gcp.cloudrunv2.ServiceIamMember(
    "nel-gw-invoker",
    project=project,
    location=region,
    name=nel.name,
    role="roles/run.invoker",
    member=gateway_sa.email.apply(lambda e: f"serviceAccount:{e}"),
    opts=pulumi.ResourceOptions(depends_on=[gateway_sa]),
)
