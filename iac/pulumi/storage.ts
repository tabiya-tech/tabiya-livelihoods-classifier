/**
 * GCS buckets for static frontend hosting (app + docs).
 *
 * Deployment:
 *   yarn --cwd app build
 *   gsutil -m rsync -r -d app/dist gs://app-classifier-tabiya-tech
 *
 *   yarn --cwd docs build
 *   gsutil -m rsync -r -d docs/dist gs://docs-classifier-tabiya-tech
 *
 * CDN invalidation (if needed):
 *   gcloud compute url-maps invalidate-cdn-cache LOAD_BALANCER_NAME --path "/*"
 */

import * as gcp from "@pulumi/gcp";

interface Args {
  project: string;
  region: string;
}

function createStaticBucket(name: string, project: string) {
  const bucket = new gcp.storage.Bucket(name, {
    project,
    name,
    location: "US",
    uniformBucketLevelAccess: true,
    website: {
      mainPageSuffix: "index.html",
      notFoundPage: "index.html",  // SPA fallback
    },
    cors: [
      {
        origins: ["*"],
        methods: ["GET", "HEAD", "OPTIONS"],
        responseHeaders: ["Content-Type"],
        maxAgeSeconds: 3600,
      },
    ],
  });

  // Allow public read
  new gcp.storage.BucketIAMMember(`${name}-public`, {
    bucket: bucket.name,
    role: "roles/storage.objectViewer",
    member: "allUsers",
  });

  // Backend bucket for Cloud CDN
  const backendBucket = new gcp.compute.BackendBucket(`${name}-backend`, {
    project,
    name: name.replace(/\./g, "-"),
    bucketName: bucket.name,
    enableCdn: true,
    cdnPolicy: {
      cacheMode: "CACHE_ALL_STATIC",
      defaultTtl: 3600,
      maxTtl: 86400,
    },
  });

  return { bucket, backendBucket };
}

export function createFrontendBuckets({ project, region }: Args) {
  const app = createStaticBucket("app-classifier-tabiya-tech", project);
  const docs = createStaticBucket("docs-classifier-tabiya-tech", project);

  // ── URL maps and HTTPS load balancers ────────────────────────────────────

  const appUrlMap = new gcp.compute.URLMap("app-url-map", {
    project,
    name: "app-classifier-tabiya-tech-lb",
    defaultService: app.backendBucket.selfLink,
  });

  const docsUrlMap = new gcp.compute.URLMap("docs-url-map", {
    project,
    name: "docs-classifier-tabiya-tech-lb",
    defaultService: docs.backendBucket.selfLink,
  });

  // GCP-managed SSL certificates
  const appCert = new gcp.compute.ManagedSslCertificate("app-cert", {
    project,
    name: "app-classifier-tabiya-tech-cert",
    managed: { domains: ["app.classifier.tabiya.tech"] },
  });

  const docsCert = new gcp.compute.ManagedSslCertificate("docs-cert", {
    project,
    name: "docs-classifier-tabiya-tech-cert",
    managed: { domains: ["docs.classifier.tabiya.tech"] },
  });

  // HTTPS proxies
  const appProxy = new gcp.compute.TargetHttpsProxy("app-proxy", {
    project,
    name: "app-classifier-tabiya-tech-proxy",
    urlMap: appUrlMap.selfLink,
    sslCertificates: [appCert.selfLink],
  });

  const docsProxy = new gcp.compute.TargetHttpsProxy("docs-proxy", {
    project,
    name: "docs-classifier-tabiya-tech-proxy",
    urlMap: docsUrlMap.selfLink,
    sslCertificates: [docsCert.selfLink],
  });

  // Global forwarding rules (port 443)
  new gcp.compute.GlobalForwardingRule("app-forwarding-rule", {
    project,
    name: "app-classifier-tabiya-tech-fw",
    target: appProxy.selfLink,
    portRange: "443",
    loadBalancingScheme: "EXTERNAL_MANAGED",
  });

  new gcp.compute.GlobalForwardingRule("docs-forwarding-rule", {
    project,
    name: "docs-classifier-tabiya-tech-fw",
    target: docsProxy.selfLink,
    portRange: "443",
    loadBalancingScheme: "EXTERNAL_MANAGED",
  });

  return { appBucket: app.bucket, docsBucket: docs.bucket };
}
