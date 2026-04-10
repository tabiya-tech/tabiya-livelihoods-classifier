"""Create an A record in the GCP Cloud DNS managed zone pointing to the load balancer IP."""

import pulumi
import pulumi_gcp as gcp


def create_a_record(
    project: str,
    dns_zone_name: pulumi.Output,
    ip_address: pulumi.Output,
    frontend_domain: str,
):
    gcp.dns.RecordSet(
        "classifier-a-record",
        project=project,
        managed_zone=dns_zone_name,
        name=f"{frontend_domain}.",
        type="A",
        ttl=300,
        rrdatas=[ip_address],
    )
