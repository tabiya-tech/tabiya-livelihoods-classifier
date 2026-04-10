"""AWS Route 53 NS record that delegates the classifier subdomain to GCP Cloud DNS."""

import pulumi
import pulumi_aws as aws


def create_ns_record(
    subdomain: str,
    name_servers: pulumi.Output,
):
    """Create an NS record in the tabiya.tech Route 53 zone.

    The zone is looked up by name at deploy time — no hardcoded zone ID needed.
    subdomain: the part before .tabiya.tech, e.g. "dev.classifier"
    name_servers: list of GCP Cloud DNS nameservers from the dns stack
    """
    zone = aws.route53.get_zone(name="tabiya.tech", private_zone=False)

    aws.route53.Record(
        "classifier-ns-record",
        zone_id=zone.zone_id,
        name=f"{subdomain}.tabiya.tech",
        type="NS",
        ttl=300,
        records=name_servers,
    )
