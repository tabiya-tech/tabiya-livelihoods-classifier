/**
 * GCP Memorystore Redis instance + Serverless VPC connector.
 *
 * Cloud Run services use the VPC connector to reach the private Redis endpoint.
 * The connector must be in the same region as the Cloud Run services.
 */

import * as gcp from "@pulumi/gcp";

interface Args {
  project: string;
  region: string;
}

export function createRedis({ project, region }: Args) {
  // Serverless VPC Access connector (required for Cloud Run → Memorystore)
  const vpcConnector = new gcp.vpcaccess.Connector("classifier-vpc-connector", {
    project,
    region,
    name: "classifier-connector",
    ipCidrRange: "10.8.0.0/28",
    network: "default",
    minInstances: 2,
    maxInstances: 3,
    machineType: "e2-micro",
  });

  // Redis instance (STANDARD_HA for prod; BASIC is cheaper for dev)
  const redis = new gcp.redis.Instance("classifier-redis", {
    project,
    region,
    name: "classifier-redis",
    tier: "BASIC",
    memorySizeGb: 1,
    redisVersion: "REDIS_7_0",
    connectMode: "PRIVATE_SERVICE_ACCESS",
    authorizedNetwork: "projects/" + project + "/global/networks/default",
    labels: {
      app: "tabiya-classifier",
    },
  });

  return { redis, vpcConnector };
}
