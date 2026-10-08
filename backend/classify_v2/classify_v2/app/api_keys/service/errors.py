class ApiKeysError(Exception):
    """Base for api-keys feature errors."""


class ApiKeyNotFoundError(ApiKeysError):
    """The requested key id does not belong to this user, or never existed."""


class ApiKeysQuotaExceededError(ApiKeysError):
    """User already holds the maximum number of active keys."""


class GcpApiKeysError(ApiKeysError):
    """Underlying GCP API Keys operation failed; the request cannot be fulfilled."""
