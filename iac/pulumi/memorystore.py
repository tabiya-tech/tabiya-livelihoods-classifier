"""GCP Memorystore Redis instance + Serverless VPC connector.

Cloud Run services use the VPC connector to reach the private Redis endpoint.
The connector must be in the same region as the Cloud Run services.
"""

import pulumi_gcp as gcp


def create_redis(project: str, region: str):
    # Serverless VPC Access connector (required for Cloud Run → Memorystore)
    vpc_connector = gcp.vpcaccess.Connector(
        "classifier-vpc-connector",
        project=project,
        region=region,
        name="classifier-connector",
        ip_cidr_range="10.8.0.0/28",
        network="default",
        min_instances=2,
        max_instances=3,
        machine_type="e2-micro",
    )

    # Redis instance (STANDARD_HA for prod; BASIC is cheaper for dev)
    redis = gcp.redis.Instance(
        "classifier-redis",
        project=project,
        region=region,
        name="classifier-redis",
        tier="BASIC",
        memory_size_gb=1,
        redis_version="REDIS_7_0",
        connect_mode="PRIVATE_SERVICE_ACCESS",
        authorized_network=f"projects/{project}/global/networks/default",
        labels={"app": "tabiya-classifier"},
    )

    return redis, vpc_connector
