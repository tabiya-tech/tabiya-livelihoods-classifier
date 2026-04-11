"""Create A records in the GCP Cloud DNS managed zone pointing to the load balancer IP."""

import pulumi
import pulumi_gcp as gcp


def create_a_record(
    project: str,
    dns_zone_name: pulumi.Output,
    ip_address: pulumi.Output,
    domain: str,
):
    resource_name = f"a-record-{domain.replace('.', '-')}"
    gcp.dns.RecordSet(
        resource_name,
        project=project,
        managed_zone=dns_zone_name,
        name=f"{domain}.",
        type="A",
        ttl=300,
        rrdatas=[ip_address],
    )
