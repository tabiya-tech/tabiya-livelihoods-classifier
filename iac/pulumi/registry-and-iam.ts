/**
 * Artifact Registry repository and service accounts for Cloud Run services.
 */

import * as gcp from "@pulumi/gcp";

interface Args {
  project: string;
  region: string;
}

export function createArtifactRegistry({ project, region }: Args) {
  // Docker repository in Artifact Registry
  const registry = new gcp.artifactregistry.Repository("tabiya-classifier", {
    project,
    location: region,
    repositoryId: "tabiya-classifier",
    format: "DOCKER",
    description: "Tabiya Livelihoods Classifier Docker images",
  });

  // One service account per Cloud Run service (least-privilege)
  const nerSa = new gcp.serviceaccount.Account("ner-sa", {
    project,
    accountId: "ner-service",
    displayName: "NER Cloud Run Service Account",
  });

  const nelSa = new gcp.serviceaccount.Account("nel-sa", {
    project,
    accountId: "nel-service",
    displayName: "NEL Cloud Run Service Account",
  });

  const classifySa = new gcp.serviceaccount.Account("classify-sa", {
    project,
    accountId: "classify-service",
    displayName: "Classify Cloud Run Service Account",
  });

  // Grant each SA read access to the Artifact Registry
  const registryReaders = [nerSa, nelSa, classifySa].map((sa, i) =>
    new gcp.artifactregistry.RepositoryIamMember(`registry-reader-${i}`, {
      project,
      location: region,
      repository: registry.repositoryId,
      role: "roles/artifactregistry.reader",
      member: sa.email.apply((e) => `serviceAccount:${e}`),
    })
  );

  // Grant classify SA access to Secret Manager secrets
  const secretAccessors = [nerSa, nelSa, classifySa].map((sa, i) =>
    new gcp.projects.IAMMember(`secret-accessor-${i}`, {
      project,
      role: "roles/secretmanager.secretAccessor",
      member: sa.email.apply((e) => `serviceAccount:${e}`),
    })
  );

  // Grant classify SA Cloud Run invoker on NER and NEL (internal calls)
  const invokerBindings = [nerSa, nelSa].map((sa, i) =>
    new gcp.projects.IAMMember(`run-invoker-${i}`, {
      project,
      role: "roles/run.invoker",
      member: classifySa.email.apply((e) => `serviceAccount:${e}`),
    })
  );

  void registryReaders;
  void secretAccessors;
  void invokerBindings;

  return {
    registry,
    serviceAccounts: { nerSa, nelSa, classifySa },
  };
}
