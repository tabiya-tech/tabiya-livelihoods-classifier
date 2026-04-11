"""Cloud Run services: NER, NEL, Classify.

NER can optionally run on GPU (L4). NEL and Classify use CPU.
All services are internal-only except via the API Gateway.
"""

import pulumi_gcp as gcp


def create_cloud_run_services(
    project: str,
    region: str,
    service_accounts: dict,
    ner_image: str,
    nel_image: str,
    classify_image: str,
    hf_token_secret: gcp.secretmanager.Secret,
    mongodb_uri_secret: gcp.secretmanager.Secret,
    mongodb_db_name: str,
    firebase_project_id: str,
):
    ner_sa = service_accounts["ner_sa"]
    nel_sa = service_accounts["nel_sa"]
    classify_sa = service_accounts["classify_sa"]

    # ── NER service ───────────────────────────────────────────────────────────
    ner = gcp.cloudrunv2.Service(
        "ner-service",
        project=project,
        location=region,
        name="ner-service",
        ingress="INGRESS_TRAFFIC_INTERNAL_ONLY",
        template=gcp.cloudrunv2.ServiceTemplateArgs(
            service_account=ner_sa.email,
            scaling=gcp.cloudrunv2.ServiceTemplateScalingArgs(
                min_instance_count=0,
                max_instance_count=3,
            ),
            containers=[
                gcp.cloudrunv2.ServiceTemplateContainerArgs(
                    image=ner_image,
                    ports=[gcp.cloudrunv2.ServiceTemplateContainerPortArgs(container_port=5002)],
                    resources=gcp.cloudrunv2.ServiceTemplateContainerResourcesArgs(
                        limits={"cpu": "2", "memory": "4Gi"},
                        # Uncomment to enable GPU (requires GPU quota):
                        # limits={"cpu": "8", "memory": "32Gi", "nvidia.com/gpu": "1"},
                        startup_cpu_boost=True,
                    ),
                    envs=[
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="HF_TOKEN",
                            value_source=gcp.cloudrunv2.ServiceTemplateContainerEnvValueSourceArgs(
                                secret_key_ref=gcp.cloudrunv2.ServiceTemplateContainerEnvValueSourceSecretKeyRefArgs(
                                    secret=hf_token_secret.secret_id,
                                    version="latest",
                                ),
                            ),
                        ),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="NER_MODEL", value="tabiya/roberta-base-job-ner"
                        ),
                    ],
                    liveness_probe=gcp.cloudrunv2.ServiceTemplateContainerLivenessProbeArgs(
                        http_get=gcp.cloudrunv2.ServiceTemplateContainerLivenessProbeHttpGetArgs(
                            path="/v1/health", port=5002
                        ),
                        initial_delay_seconds=30,
                        period_seconds=30,
                    ),
                    startup_probe=gcp.cloudrunv2.ServiceTemplateContainerStartupProbeArgs(
                        http_get=gcp.cloudrunv2.ServiceTemplateContainerStartupProbeHttpGetArgs(
                            path="/v1/health", port=5002
                        ),
                        initial_delay_seconds=60,
                        period_seconds=10,
                        failure_threshold=30,
                    ),
                )
            ],
        ),
    )

    # ── NEL service ───────────────────────────────────────────────────────────
    nel = gcp.cloudrunv2.Service(
        "nel-service",
        project=project,
        location=region,
        name="nel-service",
        ingress="INGRESS_TRAFFIC_INTERNAL_ONLY",
        template=gcp.cloudrunv2.ServiceTemplateArgs(
            service_account=nel_sa.email,
            scaling=gcp.cloudrunv2.ServiceTemplateScalingArgs(
                min_instance_count=0,
                max_instance_count=5,
            ),
            containers=[
                gcp.cloudrunv2.ServiceTemplateContainerArgs(
                    image=nel_image,
                    ports=[gcp.cloudrunv2.ServiceTemplateContainerPortArgs(container_port=5003)],
                    resources=gcp.cloudrunv2.ServiceTemplateContainerResourcesArgs(
                        limits={"cpu": "2", "memory": "4Gi"},
                        startup_cpu_boost=True,
                    ),
                    envs=[
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="HF_TOKEN",
                            value_source=gcp.cloudrunv2.ServiceTemplateContainerEnvValueSourceArgs(
                                secret_key_ref=gcp.cloudrunv2.ServiceTemplateContainerEnvValueSourceSecretKeyRefArgs(
                                    secret=hf_token_secret.secret_id,
                                    version="latest",
                                ),
                            ),
                        ),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="LINKER_MODEL", value="all-MiniLM-L6-v2"
                        ),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="NEL_FILES_PATH", value="/app/nel/nel/files"
                        ),
                    ],
                    liveness_probe=gcp.cloudrunv2.ServiceTemplateContainerLivenessProbeArgs(
                        http_get=gcp.cloudrunv2.ServiceTemplateContainerLivenessProbeHttpGetArgs(
                            path="/v1/health", port=5003
                        ),
                        initial_delay_seconds=30,
                        period_seconds=30,
                    ),
                    startup_probe=gcp.cloudrunv2.ServiceTemplateContainerStartupProbeArgs(
                        http_get=gcp.cloudrunv2.ServiceTemplateContainerStartupProbeHttpGetArgs(
                            path="/v1/health", port=5003
                        ),
                        initial_delay_seconds=30,
                        period_seconds=10,
                        failure_threshold=12,  # 2 min total: model loads from image cache
                    ),
                )
            ],
        ),
    )

    # ── Classify service ──────────────────────────────────────────────────────
    classify = gcp.cloudrunv2.Service(
        "classify-service",
        project=project,
        location=region,
        name="classify-service",
        ingress="INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER",  # GCP API Gateway counts as LB; blocks direct .run.app access
        template=gcp.cloudrunv2.ServiceTemplateArgs(
            service_account=classify_sa.email,
            scaling=gcp.cloudrunv2.ServiceTemplateScalingArgs(
                min_instance_count=1,
                max_instance_count=10,
            ),
            containers=[
                gcp.cloudrunv2.ServiceTemplateContainerArgs(
                    image=classify_image,
                    ports=[gcp.cloudrunv2.ServiceTemplateContainerPortArgs(container_port=5001)],
                    resources=gcp.cloudrunv2.ServiceTemplateContainerResourcesArgs(
                        limits={"cpu": "2", "memory": "2Gi"},
                        startup_cpu_boost=True,
                    ),
                    envs=[
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="NER_API_URL", value=ner.uri
                        ),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="NEL_API_URL", value=nel.uri
                        ),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="APPLICATION_MONGODB_URI",
                            value_source=gcp.cloudrunv2.ServiceTemplateContainerEnvValueSourceArgs(
                                secret_key_ref=gcp.cloudrunv2.ServiceTemplateContainerEnvValueSourceSecretKeyRefArgs(
                                    secret=mongodb_uri_secret.secret_id,
                                    version="latest",
                                ),
                            ),
                        ),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="MONGODB_DB_NAME", value=mongodb_db_name
                        ),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="MAX_TEXT_LENGTH", value="50000"
                        ),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="MAX_BATCH_SIZE", value="500"
                        ),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="FIREBASE_PROJECT_ID", value=firebase_project_id
                        ),
                    ],
                    liveness_probe=gcp.cloudrunv2.ServiceTemplateContainerLivenessProbeArgs(
                        http_get=gcp.cloudrunv2.ServiceTemplateContainerLivenessProbeHttpGetArgs(
                            path="/v1/health", port=5001
                        ),
                        initial_delay_seconds=10,
                        period_seconds=30,
                    ),
                )
            ],
        ),
    )

    # Allow the GCP API Gateway service agent to invoke Classify.
    # The service agent identity follows the pattern below; it is project-number-based
    # and is created automatically when the API Gateway API is enabled.
    api_gateway_sa = f"serviceAccount:service-{project}@gcp-sa-apigateway.iam.gserviceaccount.com"
    gcp.cloudrunv2.ServiceIamMember(
        "classify-invoker",
        project=project,
        location=region,
        name=classify.name,
        role="roles/run.invoker",
        member=api_gateway_sa,
    )

    # Allow Classify SA to invoke NER and NEL internally
    gcp.cloudrunv2.ServiceIamMember(
        "ner-classify-invoker",
        project=project,
        location=region,
        name=ner.name,
        role="roles/run.invoker",
        member=classify_sa.email.apply(lambda e: f"serviceAccount:{e}"),
    )

    gcp.cloudrunv2.ServiceIamMember(
        "nel-classify-invoker",
        project=project,
        location=region,
        name=nel.name,
        role="roles/run.invoker",
        member=classify_sa.email.apply(lambda e: f"serviceAccount:{e}"),
    )

    return ner, nel, classify
