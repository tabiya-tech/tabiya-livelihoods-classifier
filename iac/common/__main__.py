"""Tabiya Livelihoods Classifier — Common Stack

Deploys the global HTTPS load balancer, SSL certificate, URL routing,
and the Cloud DNS A record that ties the frontend domain to the LB IP.

Reads outputs from:
  - tabiya-classifier-dns   → dns_zone_name
  - tabiya-classifier-backend → api_gateway_id, app_bucket_name, docs_bucket_name

Required Pulumi config:
  tabiya-classifier-common:project         — GCP project ID
  tabiya-classifier-common:region          — GCP region (for Serverless NEG)
  tabiya-classifier-common:env             — Stack name (dev / staging / prod)
  tabiya-classifier-common:frontendDomain  — e.g. "dev.classifier.tabiya.tech"
"""

import pulumi
from load_balancer import create_load_balancer
from dns_record import create_a_record

config = pulumi.Config()
project = config.require("project")
region = config.get("region") or "us-central1"
env = config.require("env")
frontend_domain = config.require("frontendDomain")

PULUMI_ORG = "tabiya-tech"

dns_stack = pulumi.StackReference(f"{PULUMI_ORG}/tabiya-classifier-dns/{env}")
backend_stack = pulumi.StackReference(f"{PULUMI_ORG}/tabiya-classifier-backend/{env}")

dns_zone_name = dns_stack.require_output("dns_zone_name")
api_gateway_id = backend_stack.require_output("apiGatewayId")
app_bucket_name = backend_stack.require_output("appBucketName")
docs_bucket_name = backend_stack.require_output("docsBucketName")

ip = create_load_balancer(
    project=project,
    region=region,
    frontend_domain=frontend_domain,
    api_gateway_id=api_gateway_id,
    app_bucket_name=app_bucket_name,
    docs_bucket_name=docs_bucket_name,
)

create_a_record(
    project=project,
    dns_zone_name=dns_zone_name,
    ip_address=ip.address,
    frontend_domain=frontend_domain,
)

pulumi.export("ipAddress", ip.address)
pulumi.export("frontendUrl", f"https://{frontend_domain}")
