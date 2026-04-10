"""GCP Cloud DNS managed zone for a classifier environment subdomain."""

import pulumi_gcp as gcp


def create_dns_zone(project: str, dns_name: str):
    """Create a Cloud DNS managed zone for {env}.classifier.tabiya.tech.

    dns_name must include the trailing dot, e.g. "dev.classifier.tabiya.tech."
    GCP will assign four nameservers that must be delegated from Route 53.
    """
    # Derive a valid zone name from the dns_name (letters, digits, dashes; max 63 chars)
    zone_label = dns_name.rstrip(".").replace(".", "-")[:63]

    zone = gcp.dns.ManagedZone(
        "classifier-dns-zone",
        project=project,
        name=zone_label,
        dns_name=dns_name,
        description=f"DNS zone for {dns_name.rstrip('.')}",
        visibility="public",
    )

    return zone, zone.name_servers
