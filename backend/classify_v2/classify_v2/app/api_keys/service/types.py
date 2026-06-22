"""Domain types for the api-keys feature."""

from pydantic import BaseModel, Field


class ApiKeyMetadata(BaseModel):
    """Persistent metadata for an issued key. Never exposes the plaintext key."""

    key_id: str
    user_id: str
    label: str
    created_at: float
    last_used_at: float | None = None
    revoked: bool = False


class CreatedApiKey(BaseModel):
    """Returned exactly once from POST — the plaintext key plus its metadata."""

    key: str = Field(..., description="The plaintext API key. Shown to the user once.")
    meta: ApiKeyMetadata
