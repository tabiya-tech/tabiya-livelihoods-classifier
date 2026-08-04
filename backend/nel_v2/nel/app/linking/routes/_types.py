from pydantic import BaseModel, Field

from nel.app.linking.service.types import EntityType, NELResponse


class EntityInput(BaseModel):
    text: str
    entity_type: EntityType


class NELRequest(BaseModel):
    entities: list[EntityInput] = Field(..., min_length=1, max_length=200)
    top_k: int = Field(default=5, ge=1, le=50)
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    language: str | None = Field(
        default=None,
        description=(
            "Language of the entity text ('en', 'es', or a locale like 'AR-es'). When set, "
            "linking uses the taxonomy model configured for that language "
            "(TAXONOMY_MODEL_ID_<LANG>), overriding the caller's user config. Omit to use "
            "the user config, then DEFAULT_TAXONOMY_MODEL_ID."
        ),
        examples=["es"],
    )
