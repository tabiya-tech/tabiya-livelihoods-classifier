"""GCS buckets for static frontend hosting (app + docs).

Only bucket creation here — load balancer wiring is in the common stack.
"""

import pulumi_gcp as gcp


def create_static_bucket(name: str, project: str):
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

    gcp.storage.BucketIAMMember(
        f"{name}-public",
        bucket=bucket.name,
        role="roles/storage.objectViewer",
        member="allUsers",
    )

    return bucket


def create_frontend_buckets(project: str, env: str):
    app_bucket = create_static_bucket(f"app-{env}-classifier-tabiya-tech", project)
    docs_bucket = create_static_bucket(f"docs-{env}-classifier-tabiya-tech", project)
    return app_bucket, docs_bucket
