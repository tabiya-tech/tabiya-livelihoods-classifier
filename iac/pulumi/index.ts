/**
 * Tabiya Livelihoods Classifier — GCP Infrastructure
 *
 * Stacks: dev, staging, prod
 *
 * Required Pulumi config (set via `pulumi config set`):
 *   tabiya-classifier:project       — GCP project ID
 *   tabiya-classifier:region        — GCP region (default: us-central1)
 *   tabiya-classifier:nerImage      — NER Docker image URI
 *   tabiya-classifier:nelImage      — NEL Docker image URI
 *   tabiya-classifier:classifyImage — Classify Docker image URI
 *   tabiya-classifier:mongodbUri    — MongoDB Atlas URI (secret)
 *   tabiya-classifier:mongodbDbName — MongoDB database name
 *   tabiya-classifier:hfToken       — HuggingFace token (secret)
 *
 * Required environment variables (for CI/CD):
 *   PULUMI_ACCESS_TOKEN             — Pulumi Cloud token
 *   GOOGLE_CREDENTIALS              — GCP service account JSON key
 */

import * as pulumi from "@pulumi/pulumi";

import { createArtifactRegistry, createServiceAccounts } from "./registry-and-iam";
import { createCloudRunServices } from "./cloud-run";
import { createApiGateway } from "./api-gateway";
import { createRedis } from "./memorystore";
import { createFrontendBuckets } from "./storage";
import { createSecrets } from "./secrets";

const config = new pulumi.Config();
const project = config.require("project");
const region = config.get("region") ?? "us-central1";
const mongodbUri = config.requireSecret("mongodbUri");
const mongodbDbName = config.get("mongodbDbName") ?? "tabiya-classifier";
const hfToken = config.requireSecret("hfToken");
const nerImage = config.require("nerImage");
const nelImage = config.require("nelImage");
const classifyImage = config.require("classifyImage");

// ── Artifact Registry ──────────────────────────────────────────────────────
const { registry, serviceAccounts } = createArtifactRegistry({ project, region });
export const artifactRegistryUrl = registry.location.apply(
  (loc) => `${loc}-docker.pkg.dev/${project}/tabiya-classifier`
);

// ── Secret Manager ─────────────────────────────────────────────────────────
const secrets = createSecrets({ project, mongodbUri, hfToken });

// ── Memorystore Redis ──────────────────────────────────────────────────────
const { redis, vpcConnector } = createRedis({ project, region });
export const redisHost = redis.host;

// ── Cloud Run Services ─────────────────────────────────────────────────────
const services = createCloudRunServices({
  project,
  region,
  serviceAccounts,
  nerImage,
  nelImage,
  classifyImage,
  hfTokenSecret: secrets.hfToken,
  mongodbUriSecret: secrets.mongodbUri,
  mongodbDbName,
  redisHost: redis.host,
  vpcConnector,
});
export const nerUrl = services.ner.statuses.apply((s) => s[0].url);
export const nelUrl = services.nel.statuses.apply((s) => s[0].url);
export const classifyUrl = services.classify.statuses.apply((s) => s[0].url);

// ── API Gateway ────────────────────────────────────────────────────────────
const { gateway } = createApiGateway({
  project,
  region,
  nerUrl,
  nelUrl,
  classifyUrl,
});
export const apiGatewayUrl = gateway.defaultHostname;

// ── Frontend Buckets (app + docs) ──────────────────────────────────────────
const { appBucket, docsBucket } = createFrontendBuckets({ project, region });
export const appBucketUrl = appBucket.url;
export const docsBucketUrl = docsBucket.url;
