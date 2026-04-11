"""Global HTTPS load balancer with host-based routing.

Hosts and their backends:
  dev.classifier.tabiya.tech        /api/* → API Gateway (Serverless NEG)
  app.dev.classifier.tabiya.tech    /*     → app GCS bucket
  docs.dev.classifier.tabiya.tech   /*     → docs GCS bucket

SSL: GCP-managed certificate covering all three domains.
HTTP → HTTPS redirect on port 80.
"""

import hashlib

import pulumi
import pulumi_gcp as gcp


def create_load_balancer(
    project: str,
    region: str,
    api_domain: str,
    app_domain: str,
    docs_domain: str,
    api_gateway_id: pulumi.Output,
    app_bucket_name: pulumi.Output,
    docs_bucket_name: pulumi.Output,
):
    # ── Static IP ──────────────────────────────────────────────────────────
    ip = gcp.compute.GlobalAddress(
        "classifier-ip",
        project=project,
        name="classifier-lb-ip",
        address_type="EXTERNAL",
    )

    # ── Backend buckets (CDN-enabled) ──────────────────────────────────────
    app_backend = gcp.compute.BackendBucket(
        "app-backend-bucket",
        project=project,
        name="classifier-app-backend",
        bucket_name=app_bucket_name,
        enable_cdn=True,
        cdn_policy=gcp.compute.BackendBucketCdnPolicyArgs(
            cache_mode="USE_ORIGIN_HEADERS",
            default_ttl=3600,
            max_ttl=86400,
        ),
    )

    docs_backend = gcp.compute.BackendBucket(
        "docs-backend-bucket",
        project=project,
        name="classifier-docs-backend",
        bucket_name=docs_bucket_name,
        enable_cdn=True,
        cdn_policy=gcp.compute.BackendBucketCdnPolicyArgs(
            cache_mode="USE_ORIGIN_HEADERS",
            default_ttl=3600,
            max_ttl=86400,
        ),
    )

    # ── API Gateway backend (Serverless NEG) ───────────────────────────────
    api_neg = gcp.compute.RegionNetworkEndpointGroup(
        "api-gateway-neg",
        project=project,
        region=region,
        name="classifier-api-gateway-neg",
        network_endpoint_type="SERVERLESS",
        app_engine=None,
        cloud_run=None,
        serverless_deployment=gcp.compute.RegionNetworkEndpointGroupServerlessDeploymentArgs(
            platform="apigateway.googleapis.com",
            resource=api_gateway_id,
        ),
    )

    api_backend = gcp.compute.BackendService(
        "api-gateway-backend",
        project=project,
        name="classifier-api-gateway-backend",
        protocol="HTTP",
        load_balancing_scheme="EXTERNAL_MANAGED",
        enable_cdn=False,
        backends=[gcp.compute.BackendServiceBackendArgs(group=api_neg.self_link)],
        log_config=gcp.compute.BackendServiceLogConfigArgs(enable=True),
    )

    # ── URL map (HTTPS) ────────────────────────────────────────────────────
    # Three host rules — each domain routes to its own backend.
    # The API domain strips the /api prefix before forwarding to the gateway.
    url_map = gcp.compute.URLMap(
        "classifier-url-map",
        project=project,
        name="classifier-lb-url-map",
        default_service=app_backend.self_link,
        host_rules=[
            gcp.compute.URLMapHostRuleArgs(
                hosts=[api_domain],
                path_matcher="api-paths",
            ),
            gcp.compute.URLMapHostRuleArgs(
                hosts=[app_domain],
                path_matcher="app-paths",
            ),
            gcp.compute.URLMapHostRuleArgs(
                hosts=[docs_domain],
                path_matcher="docs-paths",
            ),
        ],
        path_matchers=[
            gcp.compute.URLMapPathMatcherArgs(
                name="api-paths",
                default_service=app_backend.self_link,
                path_rules=[
                    gcp.compute.URLMapPathMatcherPathRuleArgs(
                        paths=["/api", "/api/*"],
                        service=api_backend.self_link,
                        route_action=gcp.compute.URLMapPathMatcherPathRuleRouteActionArgs(
                            url_rewrite=gcp.compute.URLMapPathMatcherPathRuleRouteActionUrlRewriteArgs(
                                path_prefix_rewrite="/",
                            )
                        ),
                    ),
                ],
            ),
            gcp.compute.URLMapPathMatcherArgs(
                name="app-paths",
                default_service=app_backend.self_link,
            ),
            gcp.compute.URLMapPathMatcherArgs(
                name="docs-paths",
                default_service=docs_backend.self_link,
            ),
        ],
    )

    # ── SSL certificate (covers all three domains) ─────────────────────────
    # Name includes a hash of the domains so that any change in domains causes
    # Pulumi to create the new cert before deleting the old one (create-before-
    # delete), avoiding the GCP error where a cert in use cannot be deleted.
    domains = [f"{api_domain}.", f"{app_domain}.", f"{docs_domain}."]
    cert_hash = hashlib.sha1(",".join(sorted(domains)).encode()).hexdigest()[:8]
    cert_name = f"classifier-ssl-{cert_hash}"

    ssl_cert = gcp.compute.ManagedSslCertificate(
        "classifier-ssl-cert",
        project=project,
        name=cert_name,
        managed=gcp.compute.ManagedSslCertificateManagedArgs(
            domains=domains,
        ),
    )

    # ── HTTPS proxy + forwarding rule ──────────────────────────────────────
    https_proxy = gcp.compute.TargetHttpsProxy(
        "classifier-https-proxy",
        project=project,
        name="classifier-https-proxy",
        url_map=url_map.self_link,
        ssl_certificates=[ssl_cert.self_link],
    )

    gcp.compute.GlobalForwardingRule(
        "classifier-https-rule",
        project=project,
        name="classifier-https-fw",
        target=https_proxy.self_link,
        ip_address=ip.address,
        port_range="443",
        load_balancing_scheme="EXTERNAL_MANAGED",
    )

    # ── HTTP → HTTPS redirect ──────────────────────────────────────────────
    redirect_map = gcp.compute.URLMap(
        "classifier-redirect-map",
        project=project,
        name="classifier-http-redirect",
        default_url_redirect=gcp.compute.URLMapDefaultUrlRedirectArgs(
            https_redirect=True,
            strip_query=False,
        ),
    )

    http_proxy = gcp.compute.TargetHttpProxy(
        "classifier-http-proxy",
        project=project,
        name="classifier-http-proxy",
        url_map=redirect_map.self_link,
    )

    gcp.compute.GlobalForwardingRule(
        "classifier-http-rule",
        project=project,
        name="classifier-http-fw",
        target=http_proxy.self_link,
        ip_address=ip.address,
        port_range="80",
        load_balancing_scheme="EXTERNAL_MANAGED",
    )

    return ip
