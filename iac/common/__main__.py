"""Tabiya Livelihoods Classifier — Common Stack

Deploys the global HTTPS load balancer, SSL certificate, URL routing,
and Cloud DNS A records for all subdomains.

Reads outputs from:
  - tabiya-classifier-dns      → dns_zone_name
  - tabiya-classifier-backend  → apiGatewayId
  - tabiya-classifier-frontend → appBucketName, docsBucketName

Required Pulumi config:
  tabiya-classifier-common:project     — GCP project ID
  tabiya-classifier-common:region      — GCP region (for Serverless NEG)
  tabiya-classifier-common:env         — Stack name (dev / staging / prod)
  tabiya-classifier-common:apiDomain   — e.g. "dev.classifier.tabiya.tech"
  tabiya-classifier-common:appDomain   — e.g. "app.dev.classifier.tabiya.tech"
  tabiya-classifier-common:docsDomain  — e.g. "docs.dev.classifier.tabiya.tech"
"""

import pulumi
from load_balancer import create_load_balancer

config = pulumi.Config()
project = config.require("project")
region = config.get("region") or "us-central1"
env = config.require("env")
api_domain = config.require("apiDomain")
app_domain = config.require("appDomain")
docs_domain = config.require("docsDomain")

PULUMI_ORG = "tabiya-tech"

backend_stack = pulumi.StackReference(f"{PULUMI_ORG}/tabiya-classifier-backend/{env}")
frontend_stack = pulumi.StackReference(f"{PULUMI_ORG}/tabiya-classifier-frontend/{env}")

api_gateway_id = backend_stack.require_output("apiGatewayId")
app_bucket_name = frontend_stack.require_output("appBucketName")
docs_bucket_name = frontend_stack.require_output("docsBucketName")

ip = create_load_balancer(
    project=project,
    region=region,
    api_domain=api_domain,
    app_domain=app_domain,
    docs_domain=docs_domain,
    api_gateway_id=api_gateway_id,
    app_bucket_name=app_bucket_name,
    docs_bucket_name=docs_bucket_name,
)

pulumi.export("ipAddress", ip.address)
pulumi.export("appUrl", f"https://{app_domain}")
pulumi.export("docsUrl", f"https://{docs_domain}")
pulumi.export("apiUrl", f"https://{api_domain}")
