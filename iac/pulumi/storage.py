"""GCS buckets for static frontend hosting (app + docs).

Deployment:
  yarn --cwd app build
  gsutil -m rsync -r -d app/dist gs://app-classifier-tabiya-tech

  yarn --cwd docs build
  gsutil -m rsync -r -d docs/dist gs://docs-classifier-tabiya-tech

CDN invalidation (if needed):
  gcloud compute url-maps invalidate-cdn-cache LOAD_BALANCER_NAME --path "/*"
"""

import pulumi_gcp as gcp


def _create_static_bucket(name: str, project: str):
    bucket = gcp.storage.Bucket(
        name,
        project=project,
        name=name,
        location="US",
        uniform_bucket_level_access=True,
        website=gcp.storage.BucketWebsiteArgs(
            main_page_suffix="index.html",
            not_found_page="index.html",  # SPA fallback
        ),
        cors=[
            gcp.storage.BucketCorArgs(
                origins=["*"],
                methods=["GET", "HEAD", "OPTIONS"],
                response_headers=["Content-Type"],
                max_age_seconds=3600,
            )
        ],
    )

    # Allow public read
    gcp.storage.BucketIAMMember(
        f"{name}-public",
        bucket=bucket.name,
        role="roles/storage.objectViewer",
        member="allUsers",
    )

    # Backend bucket for Cloud CDN
    backend_bucket = gcp.compute.BackendBucket(
        f"{name}-backend",
        project=project,
        name=name.replace(".", "-"),
        bucket_name=bucket.name,
        enable_cdn=True,
        cdn_policy=gcp.compute.BackendBucketCdnPolicyArgs(
            cache_mode="CACHE_ALL_STATIC",
            default_ttl=3600,
            max_ttl=86400,
        ),
    )

    return bucket, backend_bucket


def create_frontend_buckets(project: str, region: str):
    app_bucket, app_backend = _create_static_bucket("app-classifier-tabiya-tech", project)
    docs_bucket, docs_backend = _create_static_bucket("docs-classifier-tabiya-tech", project)

    # ── URL maps and HTTPS load balancers ─────────────────────────────────────

    app_url_map = gcp.compute.URLMap(
        "app-url-map",
        project=project,
        name="app-classifier-tabiya-tech-lb",
        default_service=app_backend.self_link,
    )

    docs_url_map = gcp.compute.URLMap(
        "docs-url-map",
        project=project,
        name="docs-classifier-tabiya-tech-lb",
        default_service=docs_backend.self_link,
    )

    # GCP-managed SSL certificates
    app_cert = gcp.compute.ManagedSslCertificate(
        "app-cert",
        project=project,
        name="app-classifier-tabiya-tech-cert",
        managed=gcp.compute.ManagedSslCertificateManagedArgs(
            domains=["app.classifier.tabiya.tech"]
        ),
    )

    docs_cert = gcp.compute.ManagedSslCertificate(
        "docs-cert",
        project=project,
        name="docs-classifier-tabiya-tech-cert",
        managed=gcp.compute.ManagedSslCertificateManagedArgs(
            domains=["docs.classifier.tabiya.tech"]
        ),
    )

    # HTTPS proxies
    app_proxy = gcp.compute.TargetHttpsProxy(
        "app-proxy",
        project=project,
        name="app-classifier-tabiya-tech-proxy",
        url_map=app_url_map.self_link,
        ssl_certificates=[app_cert.self_link],
    )

    docs_proxy = gcp.compute.TargetHttpsProxy(
        "docs-proxy",
        project=project,
        name="docs-classifier-tabiya-tech-proxy",
        url_map=docs_url_map.self_link,
        ssl_certificates=[docs_cert.self_link],
    )

    # Global forwarding rules (port 443)
    gcp.compute.GlobalForwardingRule(
        "app-forwarding-rule",
        project=project,
        name="app-classifier-tabiya-tech-fw",
        target=app_proxy.self_link,
        port_range="443",
        load_balancing_scheme="EXTERNAL_MANAGED",
    )

    gcp.compute.GlobalForwardingRule(
        "docs-forwarding-rule",
        project=project,
        name="docs-classifier-tabiya-tech-fw",
        target=docs_proxy.self_link,
        port_range="443",
        load_balancing_scheme="EXTERNAL_MANAGED",
    )

    return app_bucket, docs_bucket
