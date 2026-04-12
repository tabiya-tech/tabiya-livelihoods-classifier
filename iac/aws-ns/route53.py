"""AWS Route 53 records for the classifier subdomain.

All A records are managed here — no GCP Cloud DNS zone is used.
"""

import pulumi
import pulumi_aws as aws


def create_dns_records(
    subdomain: str,
    ip_address: pulumi.Output,
):
    """Create A records for the apex and frontend subdomains in Route 53.

    subdomain: the part before .tabiya.tech, e.g. "dev.classifier"
    ip_address: load balancer IP from the common stack
    """
    zone = aws.route53.get_zone(name="tabiya.tech", private_zone=False)

    for name in [
        f"{subdomain}.tabiya.tech",
        f"app.{subdomain}.tabiya.tech",
        f"docs.{subdomain}.tabiya.tech",
    ]:
        label = name.replace(".", "-")
        aws.route53.Record(
            f"a-record-{label}",
            zone_id=zone.zone_id,
            name=name,
            type="A",
            ttl=300,
            records=[ip_address],
        )
