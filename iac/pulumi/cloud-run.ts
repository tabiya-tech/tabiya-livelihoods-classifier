/**
 * Cloud Run services: NER, NEL, Classify.
 *
 * NER can optionally run on GPU (L4).  NEL and Classify use CPU.
 * All services are internal-only except via the API Gateway.
 */

import * as gcp from "@pulumi/gcp";
import * as pulumi from "@pulumi/pulumi";

interface ServiceAccounts {
  nerSa: gcp.serviceaccount.Account;
  nelSa: gcp.serviceaccount.Account;
  classifySa: gcp.serviceaccount.Account;
}

interface Args {
  project: string;
  region: string;
  serviceAccounts: ServiceAccounts;
  nerImage: string;
  nelImage: string;
  classifyImage: string;
  hfTokenSecret: gcp.secretmanager.Secret;
  mongodbUriSecret: gcp.secretmanager.Secret;
  mongodbDbName: string;
  redisHost: pulumi.Output<string>;
  vpcConnector: gcp.vpcaccess.Connector;
}

export function createCloudRunServices({
  project,
  region,
  serviceAccounts,
  nerImage,
  nelImage,
  classifyImage,
  hfTokenSecret,
  mongodbUriSecret,
  mongodbDbName,
  redisHost,
  vpcConnector,
}: Args) {
  const { nerSa, nelSa, classifySa } = serviceAccounts;

  // ── NER service ───────────────────────────────────────────────────────────
  const ner = new gcp.cloudrunv2.Service("ner-service", {
    project,
    location: region,
    name: "ner-service",
    ingress: "INGRESS_TRAFFIC_INTERNAL_ONLY",
    template: {
      serviceAccount: nerSa.email,
      scaling: { minInstanceCount: 0, maxInstanceCount: 3 },
      containers: [
        {
          image: nerImage,
          ports: [{ containerPort: 5002 }],
          resources: {
            limits: { cpu: "2", memory: "4Gi" },
            // Uncomment to enable GPU (requires GPU quota):
            // limits: { cpu: "8", memory: "32Gi", "nvidia.com/gpu": "1" },
            startupCpuBoost: true,
          },
          envs: [
            {
              name: "HF_TOKEN",
              valueSource: {
                secretKeyRef: {
                  secret: hfTokenSecret.secretId,
                  version: "latest",
                },
              },
            },
            { name: "NER_MODEL", value: "tabiya/roberta-base-job-ner" },
            { name: "PORT", value: "5002" },
          ],
          livenessProbe: {
            httpGet: { path: "/v1/health", port: 5002 },
            initialDelaySeconds: 30,
            periodSeconds: 30,
          },
          startupProbe: {
            httpGet: { path: "/v1/health", port: 5002 },
            initialDelaySeconds: 60,
            periodSeconds: 10,
            failureThreshold: 30,
          },
        },
      ],
    },
  });

  // ── NEL service ───────────────────────────────────────────────────────────
  const nel = new gcp.cloudrunv2.Service("nel-service", {
    project,
    location: region,
    name: "nel-service",
    ingress: "INGRESS_TRAFFIC_INTERNAL_ONLY",
    template: {
      serviceAccount: nelSa.email,
      scaling: { minInstanceCount: 0, maxInstanceCount: 5 },
      containers: [
        {
          image: nelImage,
          ports: [{ containerPort: 5003 }],
          resources: {
            limits: { cpu: "2", memory: "4Gi" },
            startupCpuBoost: true,
          },
          envs: [
            { name: "LINKER_MODEL", value: "all-MiniLM-L6-v2" },
            { name: "NEL_FILES_PATH", value: "/app/nel/nel/files" },
            { name: "PORT", value: "5003" },
          ],
          livenessProbe: {
            httpGet: { path: "/v1/health", port: 5003 },
            initialDelaySeconds: 30,
            periodSeconds: 30,
          },
          startupProbe: {
            httpGet: { path: "/v1/health", port: 5003 },
            initialDelaySeconds: 60,
            periodSeconds: 10,
            failureThreshold: 30,
          },
        },
      ],
    },
  });

  // ── Classify service ──────────────────────────────────────────────────────
  const classify = new gcp.cloudrunv2.Service("classify-service", {
    project,
    location: region,
    name: "classify-service",
    ingress: "INGRESS_TRAFFIC_ALL",  // public — fronted by API Gateway
    template: {
      serviceAccount: classifySa.email,
      scaling: { minInstanceCount: 0, maxInstanceCount: 10 },
      vpcAccess: {
        connector: vpcConnector.id,
        egress: "PRIVATE_RANGES_ONLY",
      },
      containers: [
        {
          image: classifyImage,
          ports: [{ containerPort: 5001 }],
          resources: {
            limits: { cpu: "2", memory: "2Gi" },
            startupCpuBoost: true,
          },
          envs: [
            {
              name: "NER_API_URL",
              value: ner.uri,
            },
            {
              name: "NEL_API_URL",
              value: nel.uri,
            },
            {
              name: "REDIS_URL",
              value: redisHost.apply((h) => `redis://${h}:6379`),
            },
            { name: "REDIS_BATCH_TTL", value: "3600" },
            {
              name: "APPLICATION_MONGODB_URI",
              valueSource: {
                secretKeyRef: {
                  secret: mongodbUriSecret.secretId,
                  version: "latest",
                },
              },
            },
            { name: "MONGODB_DB_NAME", value: mongodbDbName },
            { name: "MAX_TEXT_LENGTH", value: "50000" },
            { name: "MAX_BATCH_SIZE", value: "500" },
            { name: "PORT", value: "5001" },
          ],
          livenessProbe: {
            httpGet: { path: "/v1/health", port: 5001 },
            initialDelaySeconds: 10,
            periodSeconds: 30,
          },
        },
      ],
    },
  });

  // Allow API Gateway (any caller with api-gateway SA) to invoke Classify
  new gcp.cloudrunv2.ServiceIamMember("classify-invoker", {
    project,
    location: region,
    name: classify.name,
    role: "roles/run.invoker",
    member: "allUsers",  // API Gateway enforces auth; Cloud Run is the backend
  });

  // Allow Classify SA to invoke NER and NEL internally
  new gcp.cloudrunv2.ServiceIamMember("ner-classify-invoker", {
    project,
    location: region,
    name: ner.name,
    role: "roles/run.invoker",
    member: classifySa.email.apply((e) => `serviceAccount:${e}`),
  });

  new gcp.cloudrunv2.ServiceIamMember("nel-classify-invoker", {
    project,
    location: region,
    name: nel.name,
    role: "roles/run.invoker",
    member: classifySa.email.apply((e) => `serviceAccount:${e}`),
  });

  return { ner, nel, classify };
}
