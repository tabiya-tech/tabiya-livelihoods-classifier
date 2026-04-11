"""Tabiya Livelihoods Classifier — AWS NS Stack

Adds an NS record in the tabiya.tech Route 53 hosted zone that delegates
the classifier subdomain (e.g. dev.classifier.tabiya.tech) to GCP Cloud DNS.
Also adds an A record for the apex domain pointing to the GCP load balancer IP.

Reads outputs from:
  - tabiya-classifier-dns    → name_servers
  - tabiya-classifier-common → ipAddress

Required Pulumi config:
  tabiya-classifier-aws-ns:env             — Stack name (dev / staging / prod)
  tabiya-classifier-aws-ns:subdomain      — e.g. "dev.classifier" (prefix before .tabiya.tech)

Required environment variables (injected by CI from GitHub Actions secrets):
  AWS_ACCESS_KEY_ID
  AWS_SECRET_ACCESS_KEY
  AWS_REGION (or set aws:region in config)
"""

import pulumi
from route53 import create_ns_record

config = pulumi.Config()
env = config.require("env")
subdomain = config.require("subdomain")

PULUMI_ORG = "tabiya-tech"

dns_stack = pulumi.StackReference(f"{PULUMI_ORG}/tabiya-classifier-dns/{env}")
common_stack = pulumi.StackReference(f"{PULUMI_ORG}/tabiya-classifier-common/{env}")

name_servers = dns_stack.require_output("name_servers")
ip_address = common_stack.require_output("ipAddress")

create_ns_record(
    subdomain=subdomain,
    name_servers=name_servers,
    ip_address=ip_address,
)
