"""GCP API Gateway with API key and Firebase authentication.

Routes:
  /v1/health              → unauthenticated
  /v1/classify/**         → API key required (x-api-key header)
  /v1/batch/**            → API key required
  /v1/user/**             → Firebase Bearer token required

The API Gateway verifies both the API key and the Firebase JWT.
For Firebase routes it decodes the token claims and forwards them to the
backend as the x-apigateway-api-userinfo header (base64 JSON).

Note: GCP API Gateway requires an OpenAPI 2.0 spec (Swagger) as its config.
"""

import base64
import json

import pulumi
import pulumi_gcp as gcp


def _build_spec(project: str, classify_url: str, firebase_project_id: str, env_subdomain: str) -> str:
    spec = {
        "swagger": "2.0",
        "info": {
            "title": "Tabiya Livelihoods Classifier API",
            "version": "1.0.0",
        },
        "host": env_subdomain,
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
            },
            "firebase": {
                # https://cloud.google.com/endpoints/docs/openapi/authenticating-users-firebase
                "authorizationUrl": "",
                "flow": "implicit",
                "type": "oauth2",
                "x-google-issuer": f"https://securetoken.google.com/{firebase_project_id}",
                "x-google-jwks_uri": "https://www.googleapis.com/service_accounts/v1/metadata/x509/securetoken@system.gserviceaccount.com",
                "x-google-audiences": firebase_project_id,
            },
        },
        "paths": {
            "/docs": {
                "get": {
                    "summary": "FastAPI Swagger UI",
                    "operationId": "swaggerUI",
                    "x-google-backend": {"address": f"{classify_url}/docs"},
                    "responses": {"200": {"description": "OK"}},
                }
            },
            "/openapi.json": {
                "get": {
                    "summary": "OpenAPI schema",
                    "operationId": "openapiSchema",
                    "x-google-backend": {"address": f"{classify_url}/openapi.json"},
                    "responses": {"200": {"description": "OK"}},
                }
            },
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
            "/v1/user/config": {
                "get": {
                    "summary": "Get user model configuration",
                    "operationId": "getUserConfig",
                    "security": [{"firebase": []}],
                    "x-google-backend": {"address": f"{classify_url}/v1/user/config"},
                    "responses": {"200": {"description": "User config"}},
                },
                "put": {
                    "summary": "Update user model configuration",
                    "operationId": "updateUserConfig",
                    "security": [{"firebase": []}],
                    "parameters": [{"in": "body", "name": "body", "schema": {"type": "object"}}],
                    "x-google-backend": {"address": f"{classify_url}/v1/user/config"},
                    "responses": {"204": {"description": "Updated"}},
                },
            },
            "/v1/user/api-keys": {
                "get": {
                    "summary": "List API keys",
                    "operationId": "listApiKeys",
                    "security": [{"firebase": []}],
                    "x-google-backend": {"address": f"{classify_url}/v1/user/api-keys"},
                    "responses": {"200": {"description": "API key list"}},
                },
                "post": {
                    "summary": "Create an API key",
                    "operationId": "createApiKey",
                    "security": [{"firebase": []}],
                    "parameters": [{"in": "body", "name": "body", "schema": {"type": "object"}}],
                    "x-google-backend": {"address": f"{classify_url}/v1/user/api-keys"},
                    "responses": {"201": {"description": "Created"}},
                },
            },
            "/v1/user/api-keys/{key_id}": {
                "delete": {
                    "summary": "Revoke an API key",
                    "operationId": "deleteApiKey",
                    "security": [{"firebase": []}],
                    "parameters": [
                        {"in": "path", "name": "key_id", "type": "string", "required": True}
                    ],
                    "x-google-backend": {
                        "address": f"{classify_url}/v1/user/api-keys/{{key_id}}",
                        "pathTranslation": "APPEND_PATH_TO_ADDRESS",
                    },
                    "responses": {"204": {"description": "Revoked"}},
                }
            },
            "/v1/user/usage": {
                "get": {
                    "summary": "Get 30-day usage stats",
                    "operationId": "getUsage",
                    "security": [{"firebase": []}],
                    "x-google-backend": {"address": f"{classify_url}/v1/user/usage"},
                    "responses": {"200": {"description": "Usage data"}},
                }
            },
        },
    }
    return json.dumps(spec)


def create_api_gateway(
    project: str,
    region: str,
    classify_url: pulumi.Output,
    firebase_project_id: str,
    env_subdomain: str,
):
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
                            _build_spec(
                                project,
                                url or "http://localhost:5001",
                                firebase_project_id,
                                env_subdomain,
                            ).encode()
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
