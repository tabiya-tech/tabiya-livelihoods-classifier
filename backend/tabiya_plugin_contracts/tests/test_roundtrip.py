"""Roundtrip tests. For every top-level contract model, serialise a
representative instance to JSON, parse it back through the same Pydantic
model, and assert the resulting dict is byte-equal to the source dict.

This guards against silent regressions where a field changes shape between
Pydantic version bumps (default handling, alias behaviour, etc.).
"""

from __future__ import annotations

import json

from tabiya_plugin_contracts import (
    CONTRACT_VERSION,
    Context,
    Entities,
    Entity,
    EntitySpan,
    ErrorCode,
    ErrorEnvelope,
    Health,
    HealthStatus,
    InvokeRequest,
    InvokeResponse,
    LinkedEntities,
    LinkedEntity,
    Manifest,
    Match,
    NoneSlot,
    PluginCategory,
    RawText,
    RawTextStream,
    RawTextStreamItem,
    SLOT_MODEL_BY_TYPE,
    Slot,
    SlotType,
)


def _roundtrip(instance):
    """Return (dumped_dict, reparsed_dict) using by-alias to preserve x-tabiya-* keys."""

    dumped_json = instance.model_dump_json(by_alias=True)
    dumped_dict = json.loads(dumped_json)
    reparsed = type(instance).model_validate(dumped_dict)
    reparsed_dict = json.loads(reparsed.model_dump_json(by_alias=True))
    return dumped_dict, reparsed_dict


def test_contract_version_is_semver_triple() -> None:
    # GIVEN the exported CONTRACT_VERSION constant
    givenVersion = CONTRACT_VERSION

    # WHEN we split on "."
    parts = givenVersion.split(".")

    # THEN there are exactly three numeric components (semver)
    expectedPartCount = 3
    assert len(parts) == expectedPartCount
    for part in parts:
        assert part.isdigit()


def test_manifest_roundtrips_with_x_tabiya_extensions_preserved() -> None:
    # GIVEN a manifest that exercises every slot type and every declared capability flag
    givenManifest = Manifest(
        plugin_id="tabiya.ner.v1",
        name="Tabiya NER",
        version="0.1.0",
        category=PluginCategory.CORE,
        summary="Named entity recognition over job ad prose.",
        detail="roberta-base-job-ner",
        icon="ner",
        input_slot=Slot(type=SlotType.RAW_TEXT, cardinality="single"),
        output_slot=Slot(type=SlotType.ENTITIES, cardinality="single"),
        config_schema={
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "string",
                    "x-source": "/v2/plugins/tabiya.ner.v1/options/model_id",
                }
            },
        },
        timeout_ms=15_000,
        **{
            "x-tabiya-contract-version": CONTRACT_VERSION,
            "x-tabiya-streams": False,
            "x-tabiya-idempotent": True,
            "x-tabiya-cancellable": True,
            "x-tabiya-batch-max": 32,
        },
    )

    # WHEN we roundtrip through JSON
    dumped, reparsed = _roundtrip(givenManifest)

    # THEN the dict is preserved byte-for-byte, including the x-tabiya-* aliases
    assert dumped == reparsed
    assert dumped["x-tabiya-contract-version"] == CONTRACT_VERSION
    assert dumped["x-tabiya-idempotent"] is True


def test_invoke_request_and_response_roundtrip() -> None:
    # GIVEN an invoke request whose input matches the RawText slot payload
    givenRequest = InvokeRequest(
        context=Context(
            request_id="req-123",
            user_id="user-1",
            pipeline_id="pipe-1",
            stage_index=0,
            deadline_ms=10_000,
        ),
        config={"model_id": "roberta-base-job-ner"},
        input=RawText(text="Statistician wanted").model_dump(),
    )
    givenResponse = InvokeResponse(
        output=Entities(
            entities=[
                Entity(
                    surface_form="Statistician",
                    entity_type="occupation",
                    span=EntitySpan(start=0, end=12),
                )
            ],
            source_text="Statistician wanted",
        ).model_dump(),
        metadata={"processing_time_ms": 42},
    )

    # WHEN we roundtrip both
    request_dumped, request_reparsed = _roundtrip(givenRequest)
    response_dumped, response_reparsed = _roundtrip(givenResponse)

    # THEN both survive intact
    assert request_dumped == request_reparsed
    assert response_dumped == response_reparsed


def test_error_envelope_covers_every_error_code() -> None:
    # GIVEN every ErrorCode enum value
    givenCodes = list(ErrorCode)

    # WHEN we build an envelope for each and roundtrip it
    for code in givenCodes:
        envelope = ErrorEnvelope(code=code, message=f"test {code.value}", detail={"stage": 1})
        dumped, reparsed = _roundtrip(envelope)

        # THEN the code survives as its string value
        assert dumped == reparsed
        assert dumped["code"] == code.value

    # AND we have exactly the six codes locked in design §1
    expectedCodeCount = 6
    assert len(givenCodes) == expectedCodeCount


def test_health_roundtrips_for_every_status() -> None:
    # GIVEN each HealthStatus value
    for status in HealthStatus:
        givenHealth = Health(status=status, detail=None if status is HealthStatus.OK else "warming up")

        # WHEN we roundtrip
        dumped, reparsed = _roundtrip(givenHealth)

        # THEN it survives
        assert dumped == reparsed


def test_slot_model_by_type_covers_every_slot_type() -> None:
    # GIVEN every SlotType enum value
    givenSlotTypes = set(SlotType)

    # WHEN we resolve each via the lookup table
    resolvedTypes = {slot_type for slot_type in givenSlotTypes if slot_type in SLOT_MODEL_BY_TYPE}

    # THEN every slot type has exactly one model
    assert resolvedTypes == givenSlotTypes


def test_linked_entities_roundtrip_preserves_matches() -> None:
    # GIVEN a linked entity with two matches
    givenLinked = LinkedEntities(
        entities=[
            LinkedEntity(
                surface_form="Statistician",
                entity_type="occupation",
                span=EntitySpan(start=0, end=12),
                matches=[
                    Match(id="OCC-1", preferred_label="statistician", score=0.91, uri="http://x/1"),
                    Match(id="OCC-2", preferred_label="data analyst", score=0.72, uri="http://x/2"),
                ],
            )
        ],
        source_text="Statistician wanted",
    )

    # WHEN we roundtrip
    dumped, reparsed = _roundtrip(givenLinked)

    # THEN scores + URIs survive
    assert dumped == reparsed
    expectedMatchCount = 2
    assert len(dumped["entities"][0]["matches"]) == expectedMatchCount


def test_raw_text_stream_roundtrip() -> None:
    # GIVEN a two-item stream
    givenStream = RawTextStream(
        items=[
            RawTextStreamItem(job_id="job-a", text="ad a"),
            RawTextStreamItem(job_id="job-b", text="ad b"),
        ]
    )

    # WHEN we roundtrip
    dumped, reparsed = _roundtrip(givenStream)

    # THEN both items survive
    assert dumped == reparsed
    expectedItemCount = 2
    assert len(dumped["items"]) == expectedItemCount


def test_none_slot_roundtrip() -> None:
    # GIVEN the sentinel None-slot payload
    givenNone = NoneSlot()

    # WHEN we roundtrip
    dumped, reparsed = _roundtrip(givenNone)

    # THEN the kind discriminator is preserved
    assert dumped == reparsed
    assert dumped["kind"] == "None"
