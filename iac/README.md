# Infrastructure as Code

Pulumi TypeScript — GCP.

**Never run `pulumi up` from this repo directly. Deployments are triggered by GitHub Actions (deploy.yml).**

## Prerequisites

- [Pulumi CLI](https://www.pulumi.com/docs/install/)
- [Node.js](https://nodejs.org/) ≥ 20
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud`)
- A Pulumi account and access token

## Structure

```
iac/pulumi/
├── index.ts              — entry point (wires all modules)
├── registry-and-iam.ts   — Artifact Registry + service accounts
├── secrets.ts            — Secret Manager secrets
├── memorystore.ts        — Redis + VPC connector
├── cloud-run.ts          — NER, NEL, Classify Cloud Run services
├── api-gateway.ts        — GCP API Gateway with API key auth
├── storage.ts            — GCS buckets + CDN for app and docs
├── Pulumi.yaml           — project config and schema
├── Pulumi.dev.yaml       — dev stack config
├── Pulumi.staging.yaml   — staging stack config
└── Pulumi.prod.yaml      — prod stack config
```

## One-time setup (per stack)

```bash
cd iac/pulumi
npm install
pulumi login            # authenticate with Pulumi Cloud
pulumi stack select dev # or staging / prod

# Required secrets (store in Pulumi config, not git):
pulumi config set --secret tabiya-classifier:mongodbUri "mongodb+srv://..."
pulumi config set --secret tabiya-classifier:hfToken "hf_..."

# Image URIs (set after first Docker build):
pulumi config set tabiya-classifier:nerImage "us-central1-docker.pkg.dev/PROJECT/tabiya-classifier/ner:SHA"
pulumi config set tabiya-classifier:nelImage "us-central1-docker.pkg.dev/PROJECT/tabiya-classifier/nel:SHA"
pulumi config set tabiya-classifier:classifyImage "us-central1-docker.pkg.dev/PROJECT/tabiya-classifier/classify:SHA"
```

## Preview and deploy (manual)

```bash
pulumi preview   # dry-run — always do this first
pulumi up        # apply changes
```

## Stack promotion

- `dev` → `staging` → `prod`
- Each stack has its own GCP project and config.
- Promote by selecting the target stack and updating image URIs.

## Required GCP APIs

Enable these on the GCP project before running `pulumi up`:

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

## MongoDB Atlas

MongoDB Atlas is **not** managed by Pulumi. Provision the cluster manually or via the Atlas UI, then provide the connection URI as a Pulumi secret (see setup above).

Required collections (created on first use):
- `users`
- `api_keys`
- `user_configs`

## DNS

After deployment, point DNS records to the load balancer IPs:

```
app.classifier.tabiya.tech  → A  <appForwardingRuleIP>
docs.classifier.tabiya.tech → A  <docsForwardingRuleIP>
api.classifier.tabiya.tech  → CNAME <gatewayDefaultHostname>
```
