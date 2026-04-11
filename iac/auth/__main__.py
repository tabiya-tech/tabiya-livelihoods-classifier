"""Tabiya Livelihoods Classifier — Auth Stack

Configures Firebase Identity Platform on the GCP project.
Both frontend domains are added to the authorized domains list so Firebase
auth tokens issued for them will be accepted by the backend.

Required Pulumi config:
  tabiya-classifier-auth:project     — GCP project ID
  tabiya-classifier-auth:appDomain   — e.g. "app.dev.classifier.tabiya.tech"
  tabiya-classifier-auth:docsDomain  — e.g. "docs.dev.classifier.tabiya.tech"
"""

import pulumi
from identity_platform import configure_identity_platform

config = pulumi.Config()
project = config.require("project")
app_domain = config.require("appDomain")
docs_domain = config.require("docsDomain")

# Always allow localhost for local development
authorized_domains = [app_domain, docs_domain, "localhost"]

api_key = configure_identity_platform(
    project=project,
    authorized_domains=authorized_domains,
)

# Exported so the frontend stack can read them via StackReference
pulumi.export("firebase_api_key", api_key)
pulumi.export("firebase_project_id", project)
# Auth domain for Firebase SDK: <project>.firebaseapp.com
pulumi.export("firebase_auth_domain", f"{project}.firebaseapp.com")
