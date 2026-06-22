"""Request / response shapes for the api-keys routes."""

from pydantic import BaseModel, Field

from classify_v2.app.api_keys.service.types import ApiKeyMetadata


class CreateApiKeyRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=100)


class CreateApiKeyResponse(BaseModel):
    """Issued once — `key` is the plaintext value the user must copy now."""

    key: str
    meta: ApiKeyMetadata


class ListApiKeysResponse(BaseModel):
    keys: list[ApiKeyMetadata]
