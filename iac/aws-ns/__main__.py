"""Tabiya Livelihoods Classifier — AWS NS Stack

Manages all DNS A records for the classifier in AWS Route 53.
No GCP Cloud DNS zone is used — all records live directly in Route 53.

Reads outputs from:
  - tabiya-classifier-common → ipAddress

Required Pulumi config:
  tabiya-classifier-aws-ns:env        — Stack name (dev / staging / prod)
  tabiya-classifier-aws-ns:subdomain  — e.g. "dev.classifier" (prefix before .tabiya.tech)

Required environment variables (injected by CI from GitHub Actions secrets):
  AWS_ACCESS_KEY_ID
  AWS_SECRET_ACCESS_KEY
  AWS_REGION (or set aws:region in config)
"""

import pulumi
from route53 import create_dns_records

config = pulumi.Config()
env = config.require("env")
subdomain = config.require("subdomain")

PULUMI_ORG = "tabiya-tech"

common_stack = pulumi.StackReference(f"{PULUMI_ORG}/tabiya-classifier-common/{env}")
ip_address = common_stack.require_output("ipAddress")

create_dns_records(
    subdomain=subdomain,
    ip_address=ip_address,
)
