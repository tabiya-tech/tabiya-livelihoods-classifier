"""Tabiya Livelihoods Classifier — Backend Stack

Deploys:
  - Artifact Registry + service accounts
  - Secret Manager secrets (mongodb_uri, hf_token, taxonomy_mongodb_uri)
  - Cloud Run services (NER, NEL, Classify, NEL v2, Classify v2)
  - API Gateway

Required Pulumi config:
  tabiya-classifier-backend:project               — GCP project ID
  tabiya-classifier-backend:region                — GCP region (default: us-central1)
  tabiya-classifier-backend:env                   — Stack name (dev / staging / prod)
  tabiya-classifier-backend:firebaseProjectId     — Firebase project ID for dashboard auth
  tabiya-classifier-backend:envSubdomain          — e.g. "dev.classifier.tabiya.tech"
  tabiya-classifier-backend:nerImage              — NER Docker image URI (injected by CI)
  tabiya-classifier-backend:nelImage              — NEL Docker image URI (injected by CI)
  tabiya-classifier-backend:classifyImage         — Classify Docker image URI (injected by CI)
  tabiya-classifier-backend:nelV2Image            — NEL v2 Docker image URI (injected by CI)
  tabiya-classifier-backend:classifyV2Image       — Classify v2 Docker image URI (injected by CI)
  tabiya-classifier-backend:taxonomyMongoDbName   — MongoDB database name for taxonomy/embeddings Atlas cluster
  tabiya-classifier-backend:taxonomyApiBaseUrl    — Base URL for the taxonomy REST API
  tabiya-classifier-backend:defaultNELModelId     — Default NEL model ID (e.g. all-MiniLM-L6-v2)
  tabiya-classifier-backend:defaultTaxonomyModelId — Default taxonomy model ID

Required environment variables (from .env.{stack}, sourced from Secret Manager):
  MONGODB_URI          — MongoDB Atlas connection URI (app DB)
  MONGODB_DB_NAME      — MongoDB database name
  HF_TOKEN             — HuggingFace access token
  TAXONOMY_MONGODB_URI — MongoDB Atlas connection URI (taxonomy/embeddings DB)
"""

import os

import pulumi
import pulumi_gcp as gcp

from registry_and_iam import create_artifact_registry
from cloud_run import create_cloud_run_services
from api_gateway import create_api, create_api_gateway
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
nel_v2_image = config.require("nelV2Image")
classify_v2_image = config.require("classifyV2Image")
taxonomy_mongodb_db_name = config.require("taxonomyMongoDbName")
taxonomy_api_base_url = config.require("taxonomyApiBaseUrl")
default_nel_model_id = config.get("defaultNELModelId") or "all-MiniLM-L6-v2"
default_taxonomy_model_id = config.get("defaultTaxonomyModelId") or ""


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
taxonomy_mongodb_uri = _require_env("TAXONOMY_MONGODB_URI")

# ── Artifact Registry ──────────────────────────────────────────────────────
registry, service_accounts = create_artifact_registry(project=project, region=region)
pulumi.export(
    "artifactRegistryUrl",
    registry.location.apply(lambda loc: f"{loc}-docker.pkg.dev/{project}/tabiya-classifier"),
)

# ── Secret Manager ─────────────────────────────────────────────────────────
secrets = create_secrets(project=project, mongodb_uri=mongodb_uri, hf_token=hf_token, taxonomy_mongodb_uri=taxonomy_mongodb_uri)

# ── API resource (declared before Cloud Run so managed_service is available) ─
# The Api resource provides the managed_service name that the Classify service
# needs to restrict generated GCP API keys to this gateway.
api = create_api(project=project)

# Enable the managed service in the project — required for API keys to work.
# The managed service name is dynamic (GCP-generated hash), so it can't be
# enabled in the enable-services stack. Must be done here after the Api exists.
gcp.projects.Service(
    "enable-api-managed-service",
    project=project,
    service=api.managed_service,
    disable_dependent_services=False,
    disable_on_destroy=False,
)

# ── Cloud Run Services ─────────────────────────────────────────────────────
ner, nel, classify, nel_v2, classify_v2 = create_cloud_run_services(
    project=project,
    region=region,
    service_accounts=service_accounts,
    ner_image=ner_image,
    nel_image=nel_image,
    classify_image=classify_image,
    nel_v2_image=nel_v2_image,
    classify_v2_image=classify_v2_image,
    hf_token_secret=secrets["hf_token"],
    mongodb_uri_secret=secrets["mongodb_uri"],
    taxonomy_mongodb_uri_secret=secrets["taxonomy_mongodb_uri"],
    mongodb_db_name=mongodb_db_name,
    taxonomy_mongodb_db_name=taxonomy_mongodb_db_name,
    firebase_project_id=firebase_project_id,
    managed_service=api.managed_service,
    taxonomy_api_base_url=taxonomy_api_base_url,
    default_nel_model_id=default_nel_model_id,
    default_taxonomy_model_id=default_taxonomy_model_id,
)
pulumi.export("nerUrl", ner.uri)
pulumi.export("nelUrl", nel.uri)
pulumi.export("classifyUrl", classify.uri)
pulumi.export("nelV2Url", nel_v2.uri)
pulumi.export("classifyV2Url", classify_v2.uri)

# ── API Gateway (config + gateway, uses Cloud Run URLs) ────────────────────
_api_config, gateway, gateway_sa = create_api_gateway(
    project=project,
    region=region,
    api=api,
    classify_url=classify.uri,
    ner_url=ner.uri,
    nel_url=nel.uri,
    nel_v2_url=nel_v2.uri,
    classify_v2_url=classify_v2.uri,
    firebase_project_id=firebase_project_id,
    env_subdomain=env_subdomain,
)
pulumi.export("apiGatewayUrl", gateway.default_hostname)
pulumi.export("apiGatewayId", gateway.gateway_id)

# Grant classify SA permission to create/delete GCP API keys.
gcp.projects.IAMMember(
    "classify-sa-apikeys-admin",
    project=project,
    role="roles/serviceusage.apiKeysAdmin",
    member=service_accounts["classify_sa"].email.apply(lambda e: f"serviceAccount:{e}"),
)

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
gcp.cloudrunv2.ServiceIamMember(
    "nel-v2-gw-invoker",
    project=project,
    location=region,
    name=nel_v2.name,
    role="roles/run.invoker",
    member=gateway_sa.email.apply(lambda e: f"serviceAccount:{e}"),
    opts=pulumi.ResourceOptions(depends_on=[gateway_sa]),
)
gcp.cloudrunv2.ServiceIamMember(
    "classify-v2-gw-invoker",
    project=project,
    location=region,
    name=classify_v2.name,
    role="roles/run.invoker",
    member=gateway_sa.email.apply(lambda e: f"serviceAccount:{e}"),
    opts=pulumi.ResourceOptions(depends_on=[gateway_sa]),
)
