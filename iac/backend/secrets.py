"""Secret Manager secrets for sensitive configuration.

The Secret resources are managed by Pulumi (creation, IAM).
The secret *values* are set here on first deploy from environment variables
that prepare.py loaded from GCP Secret Manager before pulumi up ran.
Subsequent rotations are done outside Pulumi via `gcloud secrets versions add`
so that secret values never enter the Pulumi state file.
"""

import pulumi_gcp as gcp


def create_secrets(project: str, mongodb_uri: str, hf_token: str, taxonomy_mongodb_uri: str) -> dict:
    mongodb_uri_secret = gcp.secretmanager.Secret(
        "mongodb-uri",
        project=project,
        secret_id="tabiya-classifier-mongodb-uri",
        replication=gcp.secretmanager.SecretReplicationArgs(auto={}),
    )

    gcp.secretmanager.SecretVersion(
        "mongodb-uri-version",
        secret=mongodb_uri_secret.id,
        secret_data=mongodb_uri,
    )

    hf_token_secret = gcp.secretmanager.Secret(
        "hf-token",
        project=project,
        secret_id="tabiya-classifier-hf-token",
        replication=gcp.secretmanager.SecretReplicationArgs(auto={}),
    )

    gcp.secretmanager.SecretVersion(
        "hf-token-version",
        secret=hf_token_secret.id,
        secret_data=hf_token,
    )

    taxonomy_mongodb_uri_secret = gcp.secretmanager.Secret(
        "taxonomy-mongodb-uri",
        project=project,
        secret_id="tabiya-classifier-taxonomy-mongodb-uri",
        replication=gcp.secretmanager.SecretReplicationArgs(auto={}),
    )

    gcp.secretmanager.SecretVersion(
        "taxonomy-mongodb-uri-version",
        secret=taxonomy_mongodb_uri_secret.id,
        secret_data=taxonomy_mongodb_uri,
    )

    return {
        "mongodb_uri": mongodb_uri_secret,
        "hf_token": hf_token_secret,
        "taxonomy_mongodb_uri": taxonomy_mongodb_uri_secret,
    }
