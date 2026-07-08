"""Pipeline validator.

Enforces every rule from design doc §7. Runs at save time (POST/PUT and
the standalone /validate endpoint) and at invoke time (11.6). Kept as a
plain synchronous class over already-fetched registry state so a single
invoke call doesn't fan out network reads for every stage — the caller
passes in the registry and the validator reads from the in-memory cache.

The validator collects every issue in one pass rather than short-circuiting
on the first failure. That lets the editor surface all problems at once
so the user isn't playing whack-a-mole.
"""

from __future__ import annotations

from typing import Iterable, Optional

import jsonschema
from jsonschema import Draft202012Validator
from tabiya_plugin_contracts import Manifest, PluginCategory, SlotType

from classify_v2.app.pipelines.registry import (
    PluginRegistry,
    PluginStatus,
)
from classify_v2.app.pipelines.repository import StageDocument
from classify_v2.app.pipelines.service.errors import IssueCode, ValidationIssue

_NER_PLUGIN_ID = "tabiya.ner.v1"
_NEL_PLUGIN_ID = "tabiya.nel.v1"


class PipelineValidator:
    """All-in-one validator over a stages array.

    Instantiate with the current `PluginRegistry`; call `validate(stages)`
    to get a (possibly empty) list of issues. The registry is queried by
    plugin_id — a plugin absent from the catalog is a distinct issue
    (`UNKNOWN_PLUGIN`) from one that's catalogued but currently
    unavailable (`UNAVAILABLE_PLUGIN`).
    """

    def __init__(self, registry: PluginRegistry) -> None:
        self._registry = registry

    def validate(self, stages: list[StageDocument]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        issues.extend(self._check_min_stages(stages))

        # Resolve every plugin_id up front. Everything downstream operates
        # on the resolved manifests + records the issues found here.
        manifests, resolution_issues = self._resolve_manifests(stages)
        issues.extend(resolution_issues)

        # Rules that need a *complete* set of manifests only run when every
        # plugin resolved. If any plugin is unknown/unavailable, the user
        # needs to fix that first — we still surface those errors below,
        # but skip slot / source / sink / limit checks to avoid follow-on
        # noise from a stage that isn't even the right plugin type.
        if all(manifest is not None for manifest in manifests) and len(stages) >= 2:
            resolved_manifests = [manifest for manifest in manifests if manifest is not None]
            issues.extend(self._check_source_and_sink_positions(stages, resolved_manifests))
            issues.extend(self._check_slot_compatibility(stages, resolved_manifests))
            issues.extend(self._check_ner_and_nel_limits(stages))

        issues.extend(self._check_stage_configs(stages, manifests))
        return issues

    # ── individual rules ────────────────────────────────────────────────

    def _check_min_stages(self, stages: list[StageDocument]) -> Iterable[ValidationIssue]:
        if len(stages) < 2:
            yield ValidationIssue(
                code=IssueCode.TOO_FEW_STAGES,
                message="A pipeline needs at least a Source and a Sink stage.",
                detail={"stage_count": len(stages)},
            )

    def _resolve_manifests(
        self, stages: list[StageDocument]
    ) -> tuple[list[Optional[Manifest]], list[ValidationIssue]]:
        manifests: list[Optional[Manifest]] = []
        issues: list[ValidationIssue] = []
        for stage_index, stage in enumerate(stages):
            resolved = self._registry.get(stage.plugin_id)
            if resolved is None:
                manifests.append(None)
                issues.append(
                    ValidationIssue(
                        code=IssueCode.UNKNOWN_PLUGIN,
                        message=f"Plugin '{stage.plugin_id}' is not in the catalog.",
                        stage_index=stage_index,
                        plugin_id=stage.plugin_id,
                    )
                )
                continue
            if resolved.coming_soon:
                manifests.append(None)
                issues.append(
                    ValidationIssue(
                        code=IssueCode.COMING_SOON_PLUGIN,
                        message=(
                            f"Plugin '{stage.plugin_id}' is marked Coming Soon "
                            "and cannot be used in a pipeline yet."
                        ),
                        stage_index=stage_index,
                        plugin_id=stage.plugin_id,
                    )
                )
                continue
            if resolved.status != PluginStatus.ENABLED or resolved.manifest is None:
                manifests.append(None)
                issues.append(
                    ValidationIssue(
                        code=IssueCode.UNAVAILABLE_PLUGIN,
                        message=(
                            f"Plugin '{stage.plugin_id}' is currently unavailable"
                            f" ({resolved.status.value})."
                        ),
                        stage_index=stage_index,
                        plugin_id=stage.plugin_id,
                        detail={"last_error": resolved.last_error} if resolved.last_error else None,
                    )
                )
                continue
            manifests.append(resolved.manifest)
        return manifests, issues

    def _check_source_and_sink_positions(
        self, stages: list[StageDocument], manifests: list[Manifest]
    ) -> Iterable[ValidationIssue]:
        first_manifest = manifests[0]
        last_manifest = manifests[-1]
        if first_manifest.input_slot.type != SlotType.NONE:
            yield ValidationIssue(
                code=IssueCode.NOT_A_SOURCE,
                message=(
                    f"First stage must be a Source (input_slot=None). "
                    f"'{first_manifest.plugin_id}' expects "
                    f"{first_manifest.input_slot.type.value}."
                ),
                stage_index=0,
                plugin_id=first_manifest.plugin_id,
            )
        if last_manifest.output_slot.type != SlotType.NONE:
            yield ValidationIssue(
                code=IssueCode.NOT_A_SINK,
                message=(
                    f"Last stage must be a Sink (output_slot=None). "
                    f"'{last_manifest.plugin_id}' emits "
                    f"{last_manifest.output_slot.type.value}."
                ),
                stage_index=len(stages) - 1,
                plugin_id=last_manifest.plugin_id,
            )

        source_positions = [
            index
            for index, manifest in enumerate(manifests)
            if manifest.category == PluginCategory.SOURCE
        ]
        if len(source_positions) > 1:
            for index in source_positions[1:]:
                yield ValidationIssue(
                    code=IssueCode.MULTIPLE_SOURCES,
                    message=(
                        "A pipeline can only contain one Source stage. "
                        f"'{manifests[index].plugin_id}' is an additional Source."
                    ),
                    stage_index=index,
                    plugin_id=manifests[index].plugin_id,
                )

        sink_positions = [
            index
            for index, manifest in enumerate(manifests)
            if manifest.category == PluginCategory.SINK
        ]
        if len(sink_positions) > 1:
            for index in sink_positions[:-1]:
                yield ValidationIssue(
                    code=IssueCode.MULTIPLE_SINKS,
                    message=(
                        "A pipeline can only contain one Sink stage. "
                        f"'{manifests[index].plugin_id}' is an additional Sink."
                    ),
                    stage_index=index,
                    plugin_id=manifests[index].plugin_id,
                )

    def _check_slot_compatibility(
        self, stages: list[StageDocument], manifests: list[Manifest]
    ) -> Iterable[ValidationIssue]:
        for stage_index in range(1, len(manifests)):
            previous_output = manifests[stage_index - 1].output_slot.type
            current_input = manifests[stage_index].input_slot.type
            if previous_output != current_input:
                yield ValidationIssue(
                    code=IssueCode.SLOT_MISMATCH,
                    message=(
                        f"Stage '{manifests[stage_index].plugin_id}' expects "
                        f"{current_input.value} but the previous stage emits "
                        f"{previous_output.value}."
                    ),
                    stage_index=stage_index,
                    plugin_id=manifests[stage_index].plugin_id,
                    detail={
                        "previous_output": previous_output.value,
                        "current_input": current_input.value,
                    },
                )

    def _check_ner_and_nel_limits(
        self, stages: list[StageDocument]
    ) -> Iterable[ValidationIssue]:
        ner_positions = [
            index for index, stage in enumerate(stages)
            if stage.plugin_id == _NER_PLUGIN_ID
        ]
        nel_positions = [
            index for index, stage in enumerate(stages)
            if stage.plugin_id == _NEL_PLUGIN_ID
        ]
        for index in ner_positions[1:]:
            yield ValidationIssue(
                code=IssueCode.NER_LIMIT_EXCEEDED,
                message="A pipeline can only contain one NER stage.",
                stage_index=index,
                plugin_id=_NER_PLUGIN_ID,
            )
        for index in nel_positions[1:]:
            yield ValidationIssue(
                code=IssueCode.NEL_LIMIT_EXCEEDED,
                message="A pipeline can only contain one NEL stage.",
                stage_index=index,
                plugin_id=_NEL_PLUGIN_ID,
            )

    def _check_stage_configs(
        self,
        stages: list[StageDocument],
        manifests: list[Optional[Manifest]],
    ) -> Iterable[ValidationIssue]:
        for stage_index, (stage, manifest) in enumerate(zip(stages, manifests)):
            if manifest is None:
                # Already reported as UNKNOWN / COMING_SOON / UNAVAILABLE.
                continue
            schema = manifest.config_schema
            if not schema:
                continue
            validator_impl = Draft202012Validator(schema)
            errors = sorted(
                validator_impl.iter_errors(stage.config or {}),
                key=lambda err: err.absolute_path,
            )
            if not errors:
                continue
            yield ValidationIssue(
                code=IssueCode.STAGE_CONFIG_INVALID,
                message=(
                    f"Config for stage '{stage.plugin_id}' failed schema validation."
                ),
                stage_index=stage_index,
                plugin_id=stage.plugin_id,
                detail={
                    "errors": [
                        {
                            "path": [str(part) for part in err.absolute_path],
                            "message": err.message,
                        }
                        for err in errors
                    ]
                },
            )
