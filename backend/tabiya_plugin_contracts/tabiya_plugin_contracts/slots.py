"""Slot payload models — the typed values that flow between pipeline stages.

Each `SlotType` enum value has exactly one matching Pydantic model here.
Adapters use `SLOT_MODEL_BY_TYPE` to parse/validate the `input` and `output`
fields of `InvokeRequest`/`InvokeResponse` against the manifest's declared
input_slot / output_slot.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional, Type

from pydantic import BaseModel, Field


class SlotType(str, Enum):
    NONE = "None"
    RAW_TEXT = "RawText"
    RAW_TEXT_STREAM = "RawTextStream"
    ENTITIES = "Entities"
    LINKED_ENTITIES = "LinkedEntities"


class NoneSlot(BaseModel):
    """Sentinel payload for Source inputs and Sink outputs."""

    kind: Literal["None"] = "None"


class RawText(BaseModel):
    text: str


class RawTextStreamItem(BaseModel):
    job_id: str
    text: str


class RawTextStream(BaseModel):
    items: list[RawTextStreamItem]


class EntitySpan(BaseModel):
    start: int
    end: int


class Entity(BaseModel):
    surface_form: str
    entity_type: str
    span: EntitySpan


class Entities(BaseModel):
    entities: list[Entity]
    source_text: str


class Match(BaseModel):
    """A single ESCO/taxonomy match for a linked entity."""

    id: str
    preferred_label: str
    score: float
    uri: Optional[str] = None


class LinkedEntity(Entity):
    matches: list[Match] = Field(default_factory=list)


class LinkedEntities(BaseModel):
    entities: list[LinkedEntity]
    source_text: str


SLOT_MODEL_BY_TYPE: dict[SlotType, Type[BaseModel]] = {
    SlotType.NONE: NoneSlot,
    SlotType.RAW_TEXT: RawText,
    SlotType.RAW_TEXT_STREAM: RawTextStream,
    SlotType.ENTITIES: Entities,
    SlotType.LINKED_ENTITIES: LinkedEntities,
}


class Slot(BaseModel):
    """Manifest input_slot / output_slot descriptor."""

    type: SlotType
    cardinality: Literal["single", "none"] = "single"
