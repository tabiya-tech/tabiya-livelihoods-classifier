from pydantic import BaseModel, Field

from nel.app.linking.service.types import EntityType, NELResponse


class EntityInput(BaseModel):
    text: str
    entity_type: EntityType


class NELRequest(BaseModel):
    entities: list[EntityInput] = Field(..., min_length=1, max_length=200)
    top_k: int = Field(default=5, ge=1, le=50)
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
