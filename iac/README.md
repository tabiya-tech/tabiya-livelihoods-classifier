# Infrastructure as Code

Pulumi Python — GCP.

**Never run `pulumi up` from this repo directly without first running `prepare.py`.
Deployments are triggered by GitHub Actions (deploy.yml).**

## How configuration works

Stack config and secrets are **never committed to git**. They live in GCP Secret Manager
in each environment's own GCP project and are fetched at deploy time.

| Secret name in GCP | Content | Written to |
|---|---|---|
| `env-vars` | `.env` file with secret values (`MONGODB_URI`, `HF_TOKEN`) | `iac/pulumi/.env.{stack}` |
| `stack-config` | YAML with Pulumi config (`project`, `region`, etc.) | `iac/pulumi/Pulumi.{stack}.yaml` |

Both generated files are git-ignored. The committed templates in `iac/templates/`
define the required keys and are used by `prepare.py` to validate what it fetches
before writing anything.

## Directory structure

```
iac/
├── pulumi/
│   ├── __main__.py         — entry point
│   ├── registry_and_iam.py — Artifact Registry + service accounts
│   ├── secrets.py          — Secret Manager resources
│   ├── memorystore.py      — Redis + VPC connector
│   ├── cloud_run.py        — NER, NEL, Classify Cloud Run services
│   ├── api_gateway.py      — GCP API Gateway with API key auth
│   ├── storage.py          — GCS buckets + CDN for app and docs
│   ├── requirements.txt    — Pulumi Python dependencies
│   └── Pulumi.yaml         — project metadata only (no stack config)
├── scripts/
│   ├── prepare.py          — fetches secrets, validates, writes local files
│   └── requirements.txt    — dependencies for prepare.py
└── templates/
    ├── env.template                 — required env var keys (committed)
    └── stack_config.template.yaml  — required Pulumi config keys (committed)
```

## Prerequisites

- [Pulumi CLI](https://www.pulumi.com/docs/install/)
- Python ≥ 3.11
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud`)
- A Pulumi account and access token

## One-time bootstrap (per environment)

Before deploying an environment for the first time, create its two secrets in the
environment's GCP project:

```bash
PROJECT=tabiya-classifier-dev   # or staging / prod
STACK=dev

# 1. Create and populate env-vars (secret values)
gcloud secrets create env-vars --project $PROJECT
cat <<'EOF' | gcloud secrets versions add env-vars --project $PROJECT --data-file=-
MONGODB_URI=mongodb+srv://...
HF_TOKEN=hf_...
EOF

# 2. Create and populate stack-config (Pulumi config YAML)
gcloud secrets create stack-config --project $PROJECT
cat <<'EOF' | gcloud secrets versions add stack-config --project $PROJECT --data-file=-
config:
  tabiya-classifier:project: tabiya-classifier-dev
  tabiya-classifier:region: us-central1
  tabiya-classifier:mongodbDbName: tabiya-classifier-dev
EOF
```

## Local deployment (manual)

```bash
# Authenticate to GCP
gcloud auth application-default login

# Install script deps
pip install -r iac/scripts/requirements.txt

# Fetch config and secrets, write local files
python iac/scripts/prepare.py \
  --stack dev \
  --project tabiya-classifier-dev \
  --ner-image us-central1-docker.pkg.dev/tabiya-classifier-dev/tabiya-classifier/ner:SHA \
  --nel-image us-central1-docker.pkg.dev/tabiya-classifier-dev/tabiya-classifier/nel:SHA \
  --classify-image us-central1-docker.pkg.dev/tabiya-classifier-dev/tabiya-classifier/classify:SHA

# Install Pulumi deps
pip install -r iac/pulumi/requirements.txt

# Source env vars and deploy
cd iac/pulumi
set -a && source .env.dev && set +a
pulumi up --stack dev
```

## Rotating secrets

Secret values are managed outside Pulumi. To rotate:

```bash
echo "new-value" | gcloud secrets versions add env-vars \
  --project tabiya-classifier-dev \
  --data-file=-
```

Then re-deploy via GitHub Actions or locally (prepare.py will pick up the new `latest` version).

## Required GCP APIs

Enable these on each environment's GCP project before first deploy:

```bash
gcloud services enable \
  run.googleapis.com \
  apigateway.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  redis.googleapis.com \
  vpcaccess.googleapis.com \
  compute.googleapis.com \
  storage.googleapis.com \
  --project PROJECT_ID
```

## Stack promotion

- `dev` → `staging` → `prod`
- Each stack has its own GCP project with its own `env-vars` and `stack-config` secrets.
- Promote by running the deploy workflow against the target stack.

## DNS

After first deployment, point DNS records to the load balancer IPs output by `pulumi up`:

```
app.classifier.tabiya.tech  → A     <appForwardingRuleIP>
docs.classifier.tabiya.tech → A     <docsForwardingRuleIP>
api.classifier.tabiya.tech  → CNAME <apiGatewayUrl>
```
