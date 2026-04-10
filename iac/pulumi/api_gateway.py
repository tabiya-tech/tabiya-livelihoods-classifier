"""GCP API Gateway with API key authentication.

Routes:
  /v1/classify/** → classify-service (Cloud Run)

Auth: x-api-key header required on all paths except GET /v1/health.

Note: GCP API Gateway requires an OpenAPI 2.0 spec (Swagger) as its config.
We generate it inline from the Cloud Run URLs.
"""

import base64
import json

import pulumi
import pulumi_gcp as gcp


def _build_spec(project: str, classify_url: str) -> str:
    spec = {
        "swagger": "2.0",
        "info": {
            "title": "Tabiya Livelihoods Classifier API",
            "version": "1.0.0",
        },
        "host": "api.classifier.tabiya.tech",
        "schemes": ["https"],
        "produces": ["application/json"],
        "x-google-backend": {
            "address": classify_url,
            "protocol": "h2",
        },
        "securityDefinitions": {
            "api_key": {
                "type": "apiKey",
                "name": "x-api-key",
                "in": "header",
                "x-google-management": {
                    "metrics": [
                        {
                            "name": "requests",
                            "displayName": "Requests",
                            "valueType": "INT64",
                            "metricKind": "DELTA",
                        }
                    ],
                    "quota": {
                        "limits": [
                            {
                                "name": "read-requests-per-day",
                                "metric": "requests",
                                "unit": "1/d/{project}",
                                "values": {"STANDARD": 1000},
                            }
                        ]
                    },
                },
            }
        },
        "paths": {
            "/v1/health": {
                "get": {
                    "summary": "Health check (unauthenticated)",
                    "operationId": "healthCheck",
                    "x-google-backend": {"address": f"{classify_url}/v1/health"},
                    "responses": {"200": {"description": "OK"}},
                }
            },
            "/v1/classify": {
                "post": {
                    "summary": "Classify a single job description",
                    "operationId": "classify",
                    "security": [{"api_key": []}],
                    "parameters": [{"in": "body", "name": "body", "schema": {"type": "object"}}],
                    "x-google-backend": {"address": f"{classify_url}/v1/classify"},
                    "responses": {"200": {"description": "Classification result"}},
                }
            },
            "/v1/classify/batch": {
                "post": {
                    "summary": "Submit a batch classification job",
                    "operationId": "classifyBatch",
                    "security": [{"api_key": []}],
                    "parameters": [{"in": "body", "name": "body", "schema": {"type": "object"}}],
                    "x-google-backend": {"address": f"{classify_url}/v1/classify/batch"},
                    "responses": {"202": {"description": "Batch accepted"}},
                }
            },
            "/v1/batch/{batch_id}/status": {
                "get": {
                    "summary": "Poll batch status",
                    "operationId": "batchStatus",
                    "security": [{"api_key": []}],
                    "parameters": [
                        {"in": "path", "name": "batch_id", "type": "string", "required": True}
                    ],
                    "x-google-backend": {
                        "address": f"{classify_url}/v1/batch/{{batch_id}}/status",
                        "pathTranslation": "APPEND_PATH_TO_ADDRESS",
                    },
                    "responses": {"200": {"description": "Batch status"}},
                }
            },
            "/v1/batch/{batch_id}/results": {
                "get": {
                    "summary": "Retrieve batch results",
                    "operationId": "batchResults",
                    "security": [{"api_key": []}],
                    "parameters": [
                        {"in": "path", "name": "batch_id", "type": "string", "required": True}
                    ],
                    "x-google-backend": {
                        "address": f"{classify_url}/v1/batch/{{batch_id}}/results",
                        "pathTranslation": "APPEND_PATH_TO_ADDRESS",
                    },
                    "responses": {"200": {"description": "Batch results"}},
                }
            },
        },
    }
    return json.dumps(spec)


def create_api_gateway(
    project: str,
    region: str,
    classify_url: pulumi.Output,
):
    # GCP API Gateway is a global resource; region is used for the API config
    api = gcp.apigateway.Api(
        "classifier-api",
        project=project,
        api_id="tabiya-classifier-api",
    )

    api_config = gcp.apigateway.ApiConfig(
        "classifier-api-config",
        project=project,
        api=api.api_id,
        display_name="Tabiya Classifier API Config",
        openapi_documents=[
            gcp.apigateway.ApiConfigOpenapiDocumentArgs(
                document=gcp.apigateway.ApiConfigOpenapiDocumentDocumentArgs(
                    path="openapi.json",
                    contents=classify_url.apply(
                        lambda url: base64.b64encode(
                            _build_spec(project, url or "http://localhost:5001").encode()
                        ).decode()
                    ),
                )
            )
        ],
    )

    gateway = gcp.apigateway.Gateway(
        "classifier-gateway",
        project=project,
        region=region,
        gateway_id="tabiya-classifier-gateway",
        api_config=api_config.id,
        display_name="Tabiya Classifier Gateway",
    )

    return api, api_config, gateway
