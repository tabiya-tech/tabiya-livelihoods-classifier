"""Tabiya Livelihoods Classifier — DNS Stack

Creates a GCP Cloud DNS managed zone for the environment subdomain.
The nameservers it returns must be added to AWS Route 53 (handled by the aws-ns stack).

Required Pulumi config:
  tabiya-classifier-dns:project    — GCP project ID
  tabiya-classifier-dns:dnsName   — e.g. "dev.classifier.tabiya.tech." (trailing dot required)
"""

import pulumi
from dns_zone import create_dns_zone

config = pulumi.Config()
project = config.require("project")
dns_name = config.require("dnsName")  # must end with "."

zone, name_servers = create_dns_zone(project=project, dns_name=dns_name)

pulumi.export("dns_zone_id", zone.id)
pulumi.export("dns_zone_name", zone.name)
pulumi.export("name_servers", name_servers)
