"""Secret Manager secrets for sensitive configuration.

Secrets are created here; their values are pushed externally (manually or via CI/CD).
Cloud Run services reference secrets by name — see cloud_run.py.
"""

import pulumi
import pulumi_gcp as gcp


def create_secrets(project: str, mongodb_uri: pulumi.Output, hf_token: pulumi.Output):
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

    return {"mongodb_uri": mongodb_uri_secret, "hf_token": hf_token_secret}
