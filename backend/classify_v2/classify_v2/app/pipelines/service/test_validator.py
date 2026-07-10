"""Pipeline validator tests.

Every rule from design §7 gets its own test. Uses a fake registry so no
Mongo / network / actual manifest fetch is involved.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pytest
from tabiya_plugin_contracts import (
    CONTRACT_VERSION,
    Manifest,
    PluginCategory,
    Slot,
    SlotType,
)

from classify_v2.app.pipelines.registry import (
    PluginRegistry,
    PluginStatus,
    ResolvedPlugin,
)
from classify_v2.app.pipelines.repository import StageDocument
from classify_v2.app.pipelines.service.errors import IssueCode
from classify_v2.app.pipelines.service.validator import PipelineValidator


class _StubRegistry:
    """Minimal drop-in for PluginRegistry.get() the validator needs."""

    def __init__(self, entries: dict[str, ResolvedPlugin]) -> None:
        self._entries = entries

    def get(self, plugin_id: str) -> Optional[ResolvedPlugin]:
        return self._entries.get(plugin_id)


def _manifest(
    plugin_id: str,
    *,
    category: PluginCategory,
    input_slot: SlotType,
    output_slot: SlotType,
    config_schema: dict | None = None,
) -> Manifest:
    return Manifest(
        plugin_id=plugin_id,
        name=plugin_id,
        version="0.1.0",
        category=category,
        summary="test",
        icon="ner",
        input_slot=Slot(type=input_slot, cardinality="none" if input_slot == SlotType.NONE else "single"),
        output_slot=Slot(type=output_slot, cardinality="none" if output_slot == SlotType.NONE else "single"),
        config_schema=config_schema or {},
        timeout_ms=5_000,
        **{"x-tabiya-contract-version": CONTRACT_VERSION},
    )


def _resolved(manifest: Manifest, *, status: PluginStatus = PluginStatus.ENABLED,
              coming_soon: bool = False, last_error: str | None = None) -> ResolvedPlugin:
    return ResolvedPlugin(
        plugin_id=manifest.plugin_id,
        resolved_url=f"http://plugin.local/{manifest.plugin_id}",
        manifest=manifest if status == PluginStatus.ENABLED else None,
        status=status,
        coming_soon=coming_soon,
        last_error=last_error,
        last_refreshed_at=datetime.now(timezone.utc),
    )


def _canonical_registry(**overrides: ResolvedPlugin) -> _StubRegistry:
    """A four-plugin registry matching the shipped catalog."""

    entries: dict[str, ResolvedPlugin] = {
        "tabiya.source.text.v1": _resolved(
            _manifest(
                "tabiya.source.text.v1",
                category=PluginCategory.SOURCE,
                input_slot=SlotType.NONE,
                output_slot=SlotType.RAW_TEXT,
                config_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "additionalProperties": False,
                },
            )
        ),
        "tabiya.ner.v1": _resolved(
            _manifest(
                "tabiya.ner.v1",
                category=PluginCategory.CORE,
                input_slot=SlotType.RAW_TEXT,
                output_slot=SlotType.ENTITIES,
                config_schema={
                    "type": "object",
                    "properties": {
                        "model_id": {"type": "string"},
                        "entity_types": {"type": "array", "items": {"type": "string"}},
                    },
                    "additionalProperties": False,
                },
            )
        ),
        "tabiya.nel.v1": _resolved(
            _manifest(
                "tabiya.nel.v1",
                category=PluginCategory.CORE,
                input_slot=SlotType.ENTITIES,
                output_slot=SlotType.LINKED_ENTITIES,
                config_schema={
                    "type": "object",
                    "properties": {
                        "nel_model_id": {"type": "string"},
                        "taxonomy_model_id": {"type": "string"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
                        "min_similarity": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["nel_model_id", "taxonomy_model_id"],
                    "additionalProperties": False,
                },
            )
        ),
        "tabiya.sink.results.v1": _resolved(
            _manifest(
                "tabiya.sink.results.v1",
                category=PluginCategory.SINK,
                input_slot=SlotType.LINKED_ENTITIES,
                output_slot=SlotType.NONE,
                config_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            )
        ),
    }
    entries.update(overrides)
    return _StubRegistry(entries)


def _canonical_stages() -> list[StageDocument]:
    return [
        StageDocument(plugin_id="tabiya.source.text.v1", config={"text": ""}),
        StageDocument(plugin_id="tabiya.ner.v1", config={}),
        StageDocument(
            plugin_id="tabiya.nel.v1",
            config={
                "nel_model_id": "all-MiniLM-L6-v2",
                "taxonomy_model_id": "model-abc",
                "top_k": 5,
                "min_similarity": 0.0,
            },
        ),
        StageDocument(plugin_id="tabiya.sink.results.v1", config={}),
    ]


def _issue_codes(issues) -> list[IssueCode]:
    return [issue.code for issue in issues]


def test_canonical_four_stage_pipeline_validates_clean() -> None:
    # GIVEN the canonical Default Tabiya shape
    givenRegistry = _canonical_registry()
    givenStages = _canonical_stages()
    validator = PipelineValidator(givenRegistry)  # type: ignore[arg-type]

    # WHEN we validate
    issues = validator.validate(givenStages)

    # THEN no issues
    assert issues == []


def test_ner_only_pipeline_ending_on_results_validates_clean() -> None:
    # GIVEN a text → ner → results pipeline (no NEL): NER emits Entities and
    # the Results sink declares LinkedEntities, which slot_accepts permits.
    givenRegistry = _canonical_registry()
    givenStages = [
        StageDocument(plugin_id="tabiya.source.text.v1", config={"text": ""}),
        StageDocument(plugin_id="tabiya.ner.v1", config={}),
        StageDocument(plugin_id="tabiya.sink.results.v1", config={}),
    ]
    validator = PipelineValidator(givenRegistry)  # type: ignore[arg-type]

    # WHEN we validate
    issues = validator.validate(givenStages)

    # THEN there is no slot-mismatch — ending on NER output is allowed
    assert IssueCode.SLOT_MISMATCH not in _issue_codes(issues)


def test_too_few_stages_is_reported() -> None:
    # GIVEN a single-stage pipeline
    givenRegistry = _canonical_registry()
    givenStages = [StageDocument(plugin_id="tabiya.source.text.v1", config={"text": ""})]
    validator = PipelineValidator(givenRegistry)  # type: ignore[arg-type]

    # WHEN we validate
    issues = validator.validate(givenStages)

    # THEN TOO_FEW_STAGES surfaces
    expectedCode = IssueCode.TOO_FEW_STAGES
    assert expectedCode in _issue_codes(issues)


def test_unknown_plugin_id_is_reported_with_stage_index() -> None:
    # GIVEN a stage referencing an id that isn't in the catalog
    givenRegistry = _canonical_registry()
    givenStages = _canonical_stages()
    givenStages[2] = StageDocument(plugin_id="tabiya.ghost.v1", config={})
    validator = PipelineValidator(givenRegistry)  # type: ignore[arg-type]

    # WHEN we validate
    issues = validator.validate(givenStages)

    # THEN UNKNOWN_PLUGIN with the right stage index
    unknown = [issue for issue in issues if issue.code == IssueCode.UNKNOWN_PLUGIN]
    expectedIssueCount = 1
    expectedStageIndex = 2
    assert len(unknown) == expectedIssueCount
    assert unknown[0].stage_index == expectedStageIndex
    assert unknown[0].plugin_id == "tabiya.ghost.v1"


def test_coming_soon_plugin_is_rejected() -> None:
    # GIVEN a stage referencing a coming_soon plugin
    givenRegistry = _canonical_registry(
        **{
            "tabiya.source.scraper.v1": ResolvedPlugin(
                plugin_id="tabiya.source.scraper.v1",
                coming_soon=True,
                status=PluginStatus.UNAVAILABLE,
                last_error="coming_soon",
            )
        }
    )
    givenStages = [
        StageDocument(plugin_id="tabiya.source.scraper.v1", config={}),
        StageDocument(plugin_id="tabiya.sink.results.v1", config={}),
    ]
    validator = PipelineValidator(givenRegistry)  # type: ignore[arg-type]

    # WHEN we validate
    issues = validator.validate(givenStages)

    # THEN COMING_SOON_PLUGIN
    assert IssueCode.COMING_SOON_PLUGIN in _issue_codes(issues)


def test_unavailable_plugin_is_rejected() -> None:
    # GIVEN a stage whose plugin is UNAVAILABLE (bundle unreachable)
    givenRegistry = _canonical_registry(
        **{
            "tabiya.ner.v1": ResolvedPlugin(
                plugin_id="tabiya.ner.v1",
                status=PluginStatus.UNAVAILABLE,
                last_error="unreachable: connection refused",
            )
        }
    )
    givenStages = _canonical_stages()
    validator = PipelineValidator(givenRegistry)  # type: ignore[arg-type]

    # WHEN we validate
    issues = validator.validate(givenStages)

    # THEN UNAVAILABLE_PLUGIN
    unavailable = [issue for issue in issues if issue.code == IssueCode.UNAVAILABLE_PLUGIN]
    assert len(unavailable) == 1
    assert unavailable[0].plugin_id == "tabiya.ner.v1"


def test_first_stage_must_be_a_source() -> None:
    # GIVEN a pipeline starting with a Core plugin instead of a Source
    givenRegistry = _canonical_registry()
    givenStages = [
        StageDocument(plugin_id="tabiya.ner.v1", config={}),
        StageDocument(plugin_id="tabiya.sink.results.v1", config={}),
    ]
    validator = PipelineValidator(givenRegistry)  # type: ignore[arg-type]

    # WHEN we validate
    issues = validator.validate(givenStages)

    # THEN NOT_A_SOURCE at stage index 0
    not_a_source = [issue for issue in issues if issue.code == IssueCode.NOT_A_SOURCE]
    assert len(not_a_source) == 1
    assert not_a_source[0].stage_index == 0


def test_last_stage_must_be_a_sink() -> None:
    # GIVEN a pipeline that ends before a Sink stage
    givenRegistry = _canonical_registry()
    givenStages = [
        StageDocument(plugin_id="tabiya.source.text.v1", config={"text": ""}),
        StageDocument(plugin_id="tabiya.ner.v1", config={}),
    ]
    validator = PipelineValidator(givenRegistry)  # type: ignore[arg-type]

    # WHEN we validate
    issues = validator.validate(givenStages)

    # THEN NOT_A_SINK on the last stage
    not_a_sink = [issue for issue in issues if issue.code == IssueCode.NOT_A_SINK]
    assert len(not_a_sink) == 1
    assert not_a_sink[0].stage_index == 1


def test_multiple_sources_reported_on_extras() -> None:
    # GIVEN two Source stages
    givenRegistry = _canonical_registry(
        **{
            "tabiya.source.text2.v1": _resolved(
                _manifest(
                    "tabiya.source.text2.v1",
                    category=PluginCategory.SOURCE,
                    input_slot=SlotType.NONE,
                    output_slot=SlotType.RAW_TEXT,
                )
            )
        }
    )
    givenStages = [
        StageDocument(plugin_id="tabiya.source.text.v1", config={"text": ""}),
        StageDocument(plugin_id="tabiya.source.text2.v1", config={}),
        StageDocument(plugin_id="tabiya.sink.results.v1", config={}),
    ]
    validator = PipelineValidator(givenRegistry)  # type: ignore[arg-type]

    # WHEN we validate
    issues = validator.validate(givenStages)

    # THEN MULTIPLE_SOURCES on the extra
    multiple = [issue for issue in issues if issue.code == IssueCode.MULTIPLE_SOURCES]
    assert len(multiple) == 1
    assert multiple[0].stage_index == 1


def test_slot_mismatch_between_adjacent_stages_reported() -> None:
    # GIVEN a pipeline that skips NER (so NEL sees RawText instead of Entities)
    givenRegistry = _canonical_registry()
    givenStages = [
        StageDocument(plugin_id="tabiya.source.text.v1", config={"text": ""}),
        StageDocument(
            plugin_id="tabiya.nel.v1",
            config={"nel_model_id": "m", "taxonomy_model_id": "t"},
        ),
        StageDocument(plugin_id="tabiya.sink.results.v1", config={}),
    ]
    validator = PipelineValidator(givenRegistry)  # type: ignore[arg-type]

    # WHEN we validate
    issues = validator.validate(givenStages)

    # THEN SLOT_MISMATCH on the NEL stage with expected slot detail
    mismatches = [issue for issue in issues if issue.code == IssueCode.SLOT_MISMATCH]
    assert len(mismatches) >= 1
    expectedDetail = {
        "previous_output": SlotType.RAW_TEXT.value,
        "current_input": SlotType.ENTITIES.value,
    }
    assert mismatches[0].detail == expectedDetail


def test_missing_required_config_is_NOT_a_validation_error() -> None:
    # GIVEN a NEL stage missing its required nel_model_id. Stage config (model
    # selection) is set independently of pipeline structure, so a partial or
    # empty config must NOT block saving/validating the pipeline.
    givenRegistry = _canonical_registry()
    givenStages = _canonical_stages()
    givenStages[2] = StageDocument(
        plugin_id="tabiya.nel.v1",
        config={"taxonomy_model_id": "model-abc"},
    )
    validator = PipelineValidator(givenRegistry)  # type: ignore[arg-type]

    # WHEN we validate
    issues = validator.validate(givenStages)

    # THEN no STAGE_CONFIG_INVALID is raised at pipeline-validate time
    assert IssueCode.STAGE_CONFIG_INVALID not in _issue_codes(issues)


def test_out_of_range_config_is_NOT_a_validation_error() -> None:
    # GIVEN a NEL top_k over the schema max — still not a pipeline-level error;
    # per-stage config is validated at invoke time by the plugin, not here.
    givenRegistry = _canonical_registry()
    givenStages = _canonical_stages()
    givenStages[2] = StageDocument(
        plugin_id="tabiya.nel.v1",
        config={
            "nel_model_id": "m",
            "taxonomy_model_id": "t",
            "top_k": 9_999,
        },
    )
    validator = PipelineValidator(givenRegistry)  # type: ignore[arg-type]

    # WHEN we validate
    issues = validator.validate(givenStages)

    # THEN no STAGE_CONFIG_INVALID
    assert IssueCode.STAGE_CONFIG_INVALID not in _issue_codes(issues)


def test_multiple_ner_stages_reported() -> None:
    # GIVEN a pipeline with two NER stages
    givenRegistry = _canonical_registry()
    givenStages = [
        StageDocument(plugin_id="tabiya.source.text.v1", config={"text": ""}),
        StageDocument(plugin_id="tabiya.ner.v1", config={}),
        StageDocument(plugin_id="tabiya.ner.v1", config={}),
        StageDocument(
            plugin_id="tabiya.nel.v1",
            config={"nel_model_id": "m", "taxonomy_model_id": "t"},
        ),
        StageDocument(plugin_id="tabiya.sink.results.v1", config={}),
    ]
    validator = PipelineValidator(givenRegistry)  # type: ignore[arg-type]

    # WHEN we validate
    issues = validator.validate(givenStages)

    # THEN NER_LIMIT_EXCEEDED
    assert IssueCode.NER_LIMIT_EXCEEDED in _issue_codes(issues)


def test_multiple_nel_stages_reported() -> None:
    # GIVEN two NEL stages
    givenRegistry = _canonical_registry()
    givenStages = [
        StageDocument(plugin_id="tabiya.source.text.v1", config={"text": ""}),
        StageDocument(plugin_id="tabiya.ner.v1", config={}),
        StageDocument(
            plugin_id="tabiya.nel.v1",
            config={"nel_model_id": "m", "taxonomy_model_id": "t"},
        ),
        StageDocument(
            plugin_id="tabiya.nel.v1",
            config={"nel_model_id": "m", "taxonomy_model_id": "t"},
        ),
        StageDocument(plugin_id="tabiya.sink.results.v1", config={}),
    ]
    validator = PipelineValidator(givenRegistry)  # type: ignore[arg-type]

    # WHEN we validate
    issues = validator.validate(givenStages)

    # THEN NEL_LIMIT_EXCEEDED
    assert IssueCode.NEL_LIMIT_EXCEEDED in _issue_codes(issues)


def test_validator_reports_every_issue_in_a_single_pass() -> None:
    # GIVEN a pipeline with two independent unknown-plugin stages — the
    # validator should surface both in one pass, not stop at the first.
    givenRegistry = _canonical_registry()
    givenStages = _canonical_stages()
    givenStages[1] = StageDocument(plugin_id="tabiya.ghost.v1", config={})
    givenStages[2] = StageDocument(plugin_id="tabiya.phantom.v1", config={})
    validator = PipelineValidator(givenRegistry)  # type: ignore[arg-type]

    # WHEN we validate
    issues = validator.validate(givenStages)

    # THEN both unknown plugins are reported, at their respective stage indexes
    unknown = [
        issue for issue in issues if issue.code == IssueCode.UNKNOWN_PLUGIN
    ]
    reportedIndexes = {issue.stage_index for issue in unknown}
    assert reportedIndexes == {1, 2}
