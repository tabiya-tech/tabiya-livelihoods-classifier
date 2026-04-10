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

    # Grant each SA read access to the Artifact Registry
    for i, sa in enumerate([ner_sa, nel_sa, classify_sa]):
        gcp.artifactregistry.RepositoryIamMember(
            f"registry-reader-{i}",
            project=project,
            location=region,
            repository=registry.repository_id,
            role="roles/artifactregistry.reader",
            member=sa.email.apply(lambda e: f"serviceAccount:{e}"),
        )

    # Grant all service accounts access to Secret Manager secrets
    for i, sa in enumerate([ner_sa, nel_sa, classify_sa]):
        gcp.projects.IAMMember(
            f"secret-accessor-{i}",
            project=project,
            role="roles/secretmanager.secretAccessor",
            member=sa.email.apply(lambda e: f"serviceAccount:{e}"),
        )

    # Grant classify SA Cloud Run invoker on NER and NEL (internal calls)
    for i in range(2):
        gcp.projects.IAMMember(
            f"run-invoker-{i}",
            project=project,
            role="roles/run.invoker",
            member=classify_sa.email.apply(lambda e: f"serviceAccount:{e}"),
        )

    return registry, {"ner_sa": ner_sa, "nel_sa": nel_sa, "classify_sa": classify_sa}
