"""Artifact Registry repository and service accounts for Cloud Run services."""

import pulumi_gcp as gcp


def create_artifact_registry(project: str, region: str):
    # Docker repository in Artifact Registry
    registry = gcp.artifactregistry.Repository(
        "tabiya-classifier",
        project=project,
        location=region,
        repository_id="tabiya-classifier",
        format="DOCKER",
        description="Tabiya Livelihoods Classifier Docker images",
    )

    # One service account per Cloud Run service (least-privilege)
    ner_sa = gcp.serviceaccount.Account(
        "ner-sa",
        project=project,
        account_id="ner-service",
        display_name="NER Cloud Run Service Account",
    )

    nel_sa = gcp.serviceaccount.Account(
        "nel-sa",
        project=project,
        account_id="nel-service",
        display_name="NEL Cloud Run Service Account",
    )

    classify_sa = gcp.serviceaccount.Account(
        "classify-sa",
        project=project,
        account_id="classify-service",
        display_name="Classify Cloud Run Service Account",
    )

    nel_v2_sa = gcp.serviceaccount.Account(
        "nel-v2-sa",
        project=project,
        account_id="nel-v2-service",
        display_name="NEL v2 Cloud Run Service Account",
    )

    classify_v2_sa = gcp.serviceaccount.Account(
        "classify-v2-sa",
        project=project,
        account_id="classify-v2-service",
        display_name="Classify v2 Cloud Run Service Account",
    )

    # Grant each SA read access to the Artifact Registry
    for i, sa in enumerate([ner_sa, nel_sa, classify_sa, nel_v2_sa, classify_v2_sa]):
        gcp.artifactregistry.RepositoryIamMember(
            f"registry-reader-{i}",
            project=project,
            location=region,
            repository=registry.repository_id,
            role="roles/artifactregistry.reader",
            member=sa.email.apply(lambda e: f"serviceAccount:{e}"),
        )

    # Grant each SA access only to the specific secrets it needs:
    #   NER      → hf-token only
    #   NEL      → no secrets
    #   Classify → mongodb-uri only (hf-token is not needed by the orchestrator)
    gcp.secretmanager.SecretIamMember(
        "ner-sa-hf-token-accessor",
        project=project,
        secret_id="tabiya-classifier-hf-token",
        role="roles/secretmanager.secretAccessor",
        member=ner_sa.email.apply(lambda e: f"serviceAccount:{e}"),
    )

    gcp.secretmanager.SecretIamMember(
        "nel-sa-hf-token-accessor",
        project=project,
        secret_id="tabiya-classifier-hf-token",
        role="roles/secretmanager.secretAccessor",
        member=nel_sa.email.apply(lambda e: f"serviceAccount:{e}"),
    )

    gcp.secretmanager.SecretIamMember(
        "classify-sa-mongodb-uri-accessor",
        project=project,
        secret_id="tabiya-classifier-mongodb-uri",
        role="roles/secretmanager.secretAccessor",
        member=classify_sa.email.apply(lambda e: f"serviceAccount:{e}"),
    )

    # NEL v2 needs both the app MongoDB URI and the taxonomy MongoDB URI
    gcp.secretmanager.SecretIamMember(
        "nel-v2-sa-mongodb-uri-accessor",
        project=project,
        secret_id="tabiya-classifier-mongodb-uri",
        role="roles/secretmanager.secretAccessor",
        member=nel_v2_sa.email.apply(lambda e: f"serviceAccount:{e}"),
    )

    gcp.secretmanager.SecretIamMember(
        "nel-v2-sa-taxonomy-mongodb-uri-accessor",
        project=project,
        secret_id="tabiya-classifier-taxonomy-mongodb-uri",
        role="roles/secretmanager.secretAccessor",
        member=nel_v2_sa.email.apply(lambda e: f"serviceAccount:{e}"),
    )

    # Per-service Cloud Run invoker bindings are set in cloud_run.py after the
    # services exist. No project-level run.invoker grant is needed.

    return registry, {
        "ner_sa": ner_sa,
        "nel_sa": nel_sa,
        "classify_sa": classify_sa,
        "nel_v2_sa": nel_v2_sa,
        "classify_v2_sa": classify_v2_sa,
    }
