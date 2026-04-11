"""Tabiya Livelihoods Classifier — Enable Services Stack

Enables all GCP APIs required across all stacks in this project.
Must be deployed before any other stack.

Required Pulumi config:
  tabiya-classifier-enable-services:project — GCP project ID
"""

import pulumi
import pulumi_gcp as gcp

config = pulumi.Config()
project = config.require("project")

# All APIs required across dns, auth, backend, and common stacks.
# disable_on_destroy=False: never accidentally disable APIs when tearing down,
# as doing so could break other stacks or require manual recovery.
REQUIRED_SERVICES = [
    # ── Bootstrap (must exist before Pulumi can manage any other service) ──
    "cloudresourcemanager.googleapis.com",

    # ── DNS ────────────────────────────────────────────────────────────────
    "dns.googleapis.com",

    # ── Auth ───────────────────────────────────────────────────────────────
    "firebase.googleapis.com",
    "identitytoolkit.googleapis.com",

    # ── Backend ────────────────────────────────────────────────────────────
    "artifactregistry.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "apigateway.googleapis.com",
    "servicemanagement.googleapis.com",  # required by API Gateway
    "servicecontrol.googleapis.com",     # required by API Gateway
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",     # required for Workload Identity Federation (CI login)

    # ── Common (load balancer) ─────────────────────────────────────────────
    "compute.googleapis.com",
]

for service in REQUIRED_SERVICES:
    resource_name = service.split(".")[0]
    gcp.projects.Service(
        f"enable-{resource_name}",
        project=project,
        service=service,
        disable_dependent_services=False,
        disable_on_destroy=False,
    )
