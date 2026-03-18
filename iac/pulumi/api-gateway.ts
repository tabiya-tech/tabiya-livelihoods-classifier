/**
 * GCP API Gateway with API key authentication.
 *
 * Routes:
 *   /v1/ner/**      → ner-service (Cloud Run)
 *   /v1/nel/**      → nel-service (Cloud Run)
 *   /v1/classify/** → classify-service (Cloud Run)
 *
 * Auth: x-api-key header required on all paths except GET /v1/health.
 *
 * The gateway sits in front of Classify (public ingress).  NER and NEL
 * remain internal-only; Classify calls them server-side.
 *
 * Note: GCP API Gateway requires an OpenAPI 2.0 spec (Swagger) as its config.
 * We generate it inline from the Cloud Run URLs.
 */

import * as gcp from "@pulumi/gcp";
import * as pulumi from "@pulumi/pulumi";

interface Args {
  project: string;
  region: string;
  nerUrl: pulumi.Output<string | undefined>;
  nelUrl: pulumi.Output<string | undefined>;
  classifyUrl: pulumi.Output<string | undefined>;
}

/** Build the OpenAPI 2.0 spec that API Gateway uses. */
function buildSpec(
  project: string,
  classifyUrl: string,
): string {
  return JSON.stringify({
    swagger: "2.0",
    info: {
      title: "Tabiya Livelihoods Classifier API",
      version: "1.0.0",
    },
    host: "api.classifier.tabiya.tech",
    schemes: ["https"],
    produces: ["application/json"],
    "x-google-backend": {
      address: classifyUrl,
      protocol: "h2",
    },
    securityDefinitions: {
      api_key: {
        type: "apiKey",
        name: "x-api-key",
        in: "header",
        "x-google-management": {
          metrics: [
            {
              name: "requests",
              displayName: "Requests",
              valueType: "INT64",
              metricKind: "DELTA",
            },
          ],
          quota: {
            limits: [
              {
                name: "read-requests-per-day",
                metric: "requests",
                unit: "1/d/{project}",
                values: { STANDARD: 1000 },
              },
            ],
          },
        },
      },
    },
    paths: {
      "/v1/health": {
        get: {
          summary: "Health check (unauthenticated)",
          operationId: "healthCheck",
          "x-google-backend": { address: `${classifyUrl}/v1/health` },
          responses: { "200": { description: "OK" } },
        },
      },
      "/v1/classify": {
        post: {
          summary: "Classify a single job description",
          operationId: "classify",
          security: [{ api_key: [] }],
          parameters: [
            {
              in: "body",
              name: "body",
              schema: { type: "object" },
            },
          ],
          "x-google-backend": { address: `${classifyUrl}/v1/classify` },
          responses: { "200": { description: "Classification result" } },
        },
      },
      "/v1/classify/batch": {
        post: {
          summary: "Submit a batch classification job",
          operationId: "classifyBatch",
          security: [{ api_key: [] }],
          parameters: [
            {
              in: "body",
              name: "body",
              schema: { type: "object" },
            },
          ],
          "x-google-backend": { address: `${classifyUrl}/v1/classify/batch` },
          responses: { "202": { description: "Batch accepted" } },
        },
      },
      "/v1/batch/{batch_id}/status": {
        get: {
          summary: "Poll batch status",
          operationId: "batchStatus",
          security: [{ api_key: [] }],
          parameters: [
            { in: "path", name: "batch_id", type: "string", required: true },
          ],
          "x-google-backend": {
            address: `${classifyUrl}/v1/batch/{batch_id}/status`,
            pathTranslation: "APPEND_PATH_TO_ADDRESS",
          },
          responses: { "200": { description: "Batch status" } },
        },
      },
      "/v1/batch/{batch_id}/results": {
        get: {
          summary: "Retrieve batch results",
          operationId: "batchResults",
          security: [{ api_key: [] }],
          parameters: [
            { in: "path", name: "batch_id", type: "string", required: true },
          ],
          "x-google-backend": {
            address: `${classifyUrl}/v1/batch/{batch_id}/results`,
            pathTranslation: "APPEND_PATH_TO_ADDRESS",
          },
          responses: { "200": { description: "Batch results" } },
        },
      },
    },
  });
}

export function createApiGateway({ project, region, classifyUrl }: Omit<Args, "nerUrl" | "nelUrl"> & { classifyUrl: pulumi.Output<string | undefined> }) {
  // GCP API Gateway is a global resource; region is used for the API config
  const api = new gcp.apigateway.Api("classifier-api", {
    project,
    apiId: "tabiya-classifier-api",
  });

  const apiConfig = new gcp.apigateway.ApiConfig("classifier-api-config", {
    project,
    api: api.apiId,
    displayName: "Tabiya Classifier API Config",
    openapiDocuments: [
      {
        document: {
          path: "openapi.json",
          contents: classifyUrl.apply((url) =>
            Buffer.from(buildSpec(project, url ?? "http://localhost:5001")).toString("base64")
          ),
        },
      },
    ],
  });

  const gateway = new gcp.apigateway.Gateway("classifier-gateway", {
    project,
    region,
    gatewayId: "tabiya-classifier-gateway",
    apiConfig: apiConfig.id,
    displayName: "Tabiya Classifier Gateway",
  });

  return { api, apiConfig, gateway };
}
