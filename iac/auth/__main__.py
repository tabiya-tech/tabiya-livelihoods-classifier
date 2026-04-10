"""Tabiya Livelihoods Classifier — Auth Stack

Configures Firebase Identity Platform on the GCP project.
The frontend domain is added to the authorized domains list so Firebase
auth tokens issued for it will be accepted by the backend.

Required Pulumi config:
  tabiya-classifier-auth:project          — GCP project ID
  tabiya-classifier-auth:frontendDomain  — e.g. "dev.classifier.tabiya.tech"
"""

import pulumi
from identity_platform import configure_identity_platform

config = pulumi.Config()
project = config.require("project")
frontend_domain = config.require("frontendDomain")

# Always allow localhost for local development
authorized_domains = [frontend_domain, "localhost"]

configure_identity_platform(
    project=project,
    authorized_domains=authorized_domains,
)
