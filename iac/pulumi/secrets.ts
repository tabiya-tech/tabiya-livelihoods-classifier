/**
 * Secret Manager secrets for sensitive configuration.
 *
 * Secrets are created here; their values are pushed externally (manually or via CI/CD).
 * Cloud Run services reference secrets by name — see cloud-run.ts.
 */

import * as gcp from "@pulumi/gcp";
import * as pulumi from "@pulumi/pulumi";

interface Args {
  project: string;
  mongodbUri: pulumi.Output<string>;
  hfToken: pulumi.Output<string>;
}

export function createSecrets({ project, mongodbUri, hfToken }: Args) {
  const mongodbUriSecret = new gcp.secretmanager.Secret("mongodb-uri", {
    project,
    secretId: "tabiya-classifier-mongodb-uri",
    replication: { auto: {} },
  });

  new gcp.secretmanager.SecretVersion("mongodb-uri-version", {
    secret: mongodbUriSecret.id,
    secretData: mongodbUri,
  });

  const hfTokenSecret = new gcp.secretmanager.Secret("hf-token", {
    project,
    secretId: "tabiya-classifier-hf-token",
    replication: { auto: {} },
  });

  new gcp.secretmanager.SecretVersion("hf-token-version", {
    secret: hfTokenSecret.id,
    secretData: hfToken,
  });

  return { mongodbUri: mongodbUriSecret, hfToken: hfTokenSecret };
}
