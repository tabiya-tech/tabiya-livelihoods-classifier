"""Contract version. Bump on any breaking change to Manifest, InvokeRequest,
InvokeResponse, ErrorEnvelope, or any slot payload model.

Semver. The major component is what the orchestrator compares against
plugins' declared `x-tabiya-contract-version` at registration time.
"""

CONTRACT_VERSION = "1.0.0"
