"""Pipeline service tests.

In-memory fake repository + stub registry — no Mongo touched here.
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

from classify_v2.app.pipelines.registry import PluginStatus, ResolvedPlugin
from classify_v2.app.pipelines.repository import (
    IPipelineRepository,
    PipelineDocument,
    PipelineNotFoundError,
    ReadonlyPipelineError,
    StageDocument,
)
from classify_v2.app.pipelines.service import (
    CreatePipelineInput,
    DEFAULT_TABIYA_NAME,
    PipelineService,
    PipelineValidationError,
    PipelineValidator,
    UpdatePipelineInput,
)
from classify_v2.app.pipelines.service.service import DefaultTabiyaConfig


class _FakeRepository(IPipelineRepository):
    """Ordered, tenant-partitioned in-memory pipelines store."""

    def __init__(self) -> None:
        self._docs: list[PipelineDocument] = []

    async def ensure_indexes(self) -> None:
        return None

    async def list_for_user(self, user_id: str) -> list[PipelineDocument]:
        return [doc.model_copy(deep=True) for doc in self._docs if doc.user_id == user_id]

    async def get(self, *, user_id: str, pipeline_id: str) -> Optional[PipelineDocument]:
        for doc in self._docs:
            if doc.user_id == user_id and doc.pipeline_id == pipeline_id:
                return doc.model_copy(deep=True)
        return None

    async def insert(self, doc: PipelineDocument) -> None:
        self._docs.append(doc.model_copy(deep=True))

    async def update(self, doc: PipelineDocument) -> PipelineDocument:
        for index, existing in enumerate(self._docs):
            if existing.user_id == doc.user_id and existing.pipeline_id == doc.pipeline_id:
                if existing.is_readonly:
                    raise ReadonlyPipelineError(doc.pipeline_id)
                merged = doc.model_copy(
                    update={
                        "is_readonly": existing.is_readonly,
                        "is_default": existing.is_default,
                        "is_active": existing.is_active,
                        "created_at": existing.created_at,
                        "updated_at": datetime.now(timezone.utc),
                    }
                )
                self._docs[index] = merged
                return merged.model_copy(deep=True)
        raise PipelineNotFoundError(doc.pipeline_id, doc.user_id)

    async def delete(self, *, user_id: str, pipeline_id: str) -> None:
        for index, existing in enumerate(self._docs):
            if existing.user_id == user_id and existing.pipeline_id == pipeline_id:
                if existing.is_readonly:
                    raise ReadonlyPipelineError(pipeline_id)
                del self._docs[index]
                return
        raise PipelineNotFoundError(pipeline_id, user_id)

    async def set_active(self, *, user_id: str, pipeline_id: str) -> PipelineDocument:
        target: PipelineDocument | None = None
        for index, existing in enumerate(self._docs):
            if existing.user_id != user_id:
                continue
            if existing.pipeline_id == pipeline_id:
                target_index = index
                target = existing.model_copy(update={"is_active": True})
            elif existing.is_active:
                self._docs[index] = existing.model_copy(update={"is_active": False})
        if target is None:
            raise PipelineNotFoundError(pipeline_id, user_id)
        self._docs[target_index] = target
        return target.model_copy(deep=True)


class _StubRegistry:
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


def _resolved(manifest: Manifest) -> ResolvedPlugin:
    return ResolvedPlugin(
        plugin_id=manifest.plugin_id,
        resolved_url=f"http://plugin.local/{manifest.plugin_id}",
        manifest=manifest,
        status=PluginStatus.ENABLED,
    )


def _canonical_registry() -> _StubRegistry:
    return _StubRegistry(
        {
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
                    config_schema={"type": "object", "additionalProperties": False},
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
                        "additionalProperties": False,
                    },
                )
            ),
        }
    )


def _canonical_stages() -> list[StageDocument]:
    return [
        StageDocument(plugin_id="tabiya.source.text.v1", config={"text": ""}),
        StageDocument(plugin_id="tabiya.ner.v1", config={}),
        StageDocument(
            plugin_id="tabiya.nel.v1",
            config={
                "nel_model_id": "m",
                "taxonomy_model_id": "t",
                "top_k": 5,
                "min_similarity": 0.0,
            },
        ),
        StageDocument(plugin_id="tabiya.sink.results.v1", config={}),
    ]


def _build_service() -> tuple[PipelineService, _FakeRepository]:
    registry = _canonical_registry()
    repo = _FakeRepository()
    service = PipelineService(
        repository=repo,
        validator=PipelineValidator(registry),  # type: ignore[arg-type]
        registry=registry,  # type: ignore[arg-type]
    )
    return service, repo


def _default_config() -> DefaultTabiyaConfig:
    return DefaultTabiyaConfig(
        nel_model_id="all-MiniLM-L6-v2",
        taxonomy_model_id="model-abc",
    )


async def test_create_persists_a_valid_pipeline_and_returns_generated_id() -> None:
    # GIVEN a valid CreatePipelineInput
    service, repo = _build_service()
    givenInput = CreatePipelineInput(name="My Pipeline", stages=_canonical_stages())

    # WHEN we create it
    result = await service.create(user_id="uid-1", request=givenInput)

    # THEN it's persisted with a generated pipeline_id, is_active=False
    assert result.pipeline_id
    assert result.name == "My Pipeline"
    assert result.is_active is False
    assert result.is_default is False
    assert result.is_readonly is False
    listed = await repo.list_for_user("uid-1")
    assert len(listed) == 1


async def test_create_rejects_invalid_stages_with_validation_error() -> None:
    # GIVEN a create request with an unknown plugin
    service, _ = _build_service()
    givenStages = [
        StageDocument(plugin_id="tabiya.ghost.v1", config={}),
        StageDocument(plugin_id="tabiya.sink.results.v1", config={}),
    ]
    givenInput = CreatePipelineInput(name="Bad", stages=givenStages)

    # WHEN we create it, THEN we get PipelineValidationError with issues
    with pytest.raises(PipelineValidationError) as exc_info:
        await service.create(user_id="uid-1", request=givenInput)
    assert exc_info.value.issues


async def test_update_replaces_name_and_stages_and_preserves_flags() -> None:
    # GIVEN a persisted pipeline
    service, repo = _build_service()
    created = await service.create(
        user_id="uid-1",
        request=CreatePipelineInput(name="Original", stages=_canonical_stages()),
    )

    # WHEN we update it
    givenStages = list(_canonical_stages())
    givenStages[2] = StageDocument(
        plugin_id="tabiya.nel.v1",
        config={
            "nel_model_id": "m",
            "taxonomy_model_id": "t",
            "top_k": 3,
            "min_similarity": 0.1,
        },
    )
    updated = await service.update(
        user_id="uid-1",
        pipeline_id=created.pipeline_id,
        request=UpdatePipelineInput(name="Renamed", stages=givenStages),
    )

    # THEN name + stages changed, flags preserved
    assert updated.name == "Renamed"
    assert updated.stages[2].config["top_k"] == 3
    assert updated.is_readonly is False
    assert updated.is_default is False


async def test_update_rejects_readonly_pipeline() -> None:
    # GIVEN a seeded default (which is readonly)
    service, _ = _build_service()
    seeded = await service.ensure_default(
        user_id="uid-1", default_config=_default_config()
    )
    givenStages = _canonical_stages()

    # WHEN we try to update it
    # THEN ReadonlyPipelineError
    with pytest.raises(ReadonlyPipelineError):
        await service.update(
            user_id="uid-1",
            pipeline_id=seeded.pipeline_id,
            request=UpdatePipelineInput(name="Hijacked", stages=givenStages),
        )


async def test_update_rejects_invalid_stages() -> None:
    # GIVEN a persisted pipeline
    service, _ = _build_service()
    created = await service.create(
        user_id="uid-1",
        request=CreatePipelineInput(name="A", stages=_canonical_stages()),
    )
    givenBadStages = [
        StageDocument(plugin_id="tabiya.ghost.v1", config={}),
        StageDocument(plugin_id="tabiya.sink.results.v1", config={}),
    ]

    # WHEN we try to update with an invalid pipeline
    # THEN PipelineValidationError
    with pytest.raises(PipelineValidationError):
        await service.update(
            user_id="uid-1",
            pipeline_id=created.pipeline_id,
            request=UpdatePipelineInput(name="Bad", stages=givenBadStages),
        )


async def test_activate_flips_the_designated_pipeline() -> None:
    # GIVEN two persisted pipelines
    service, _ = _build_service()
    first = await service.create(
        user_id="uid-1",
        request=CreatePipelineInput(name="A", stages=_canonical_stages()),
    )
    second = await service.create(
        user_id="uid-1",
        request=CreatePipelineInput(name="B", stages=_canonical_stages()),
    )

    # WHEN we activate the second
    activated = await service.activate(
        user_id="uid-1", pipeline_id=second.pipeline_id
    )

    # THEN it's active; the first isn't
    assert activated.is_active is True
    others = [
        doc for doc in await service.list_for_user("uid-1") if doc.pipeline_id != second.pipeline_id
    ]
    assert all(doc.is_active is False for doc in others)
    assert first.pipeline_id != activated.pipeline_id


async def test_activate_re_validates_and_refuses_stale_reference() -> None:
    # GIVEN a persisted pipeline saved against the current registry
    service, repo = _build_service()
    created = await service.create(
        user_id="uid-1",
        request=CreatePipelineInput(name="A", stages=_canonical_stages()),
    )
    # AND the registry loses one of its plugins between save and activate
    #     (simulate manifest drift by rebuilding the service with a smaller registry)
    stripped_registry = _StubRegistry({
        plugin_id: entry
        for plugin_id, entry in _canonical_registry()._entries.items()
        if plugin_id != "tabiya.ner.v1"
    })
    service_after_drift = PipelineService(
        repository=repo,
        validator=PipelineValidator(stripped_registry),  # type: ignore[arg-type]
        registry=stripped_registry,  # type: ignore[arg-type]
    )

    # WHEN we try to activate the previously-valid pipeline
    # THEN PipelineValidationError with the missing plugin surfaced
    with pytest.raises(PipelineValidationError) as exc_info:
        await service_after_drift.activate(
            user_id="uid-1", pipeline_id=created.pipeline_id
        )
    assert any(
        issue.plugin_id == "tabiya.ner.v1" for issue in exc_info.value.issues
    )


async def test_clone_produces_a_new_pipeline_with_suffixed_name() -> None:
    # GIVEN a persisted pipeline
    service, _ = _build_service()
    created = await service.create(
        user_id="uid-1",
        request=CreatePipelineInput(name="Alpha", stages=_canonical_stages()),
    )

    # WHEN we clone it
    cloned = await service.clone(user_id="uid-1", pipeline_id=created.pipeline_id)

    # THEN the clone has a new id, is_readonly=False, is_default=False, is_active=False,
    # and name suffixed with " (copy)"
    expectedName = "Alpha (copy)"
    assert cloned.pipeline_id != created.pipeline_id
    assert cloned.name == expectedName
    assert cloned.is_readonly is False
    assert cloned.is_default is False
    assert cloned.is_active is False


async def test_clone_does_not_double_suffix_a_previous_copy() -> None:
    # GIVEN a pipeline whose name already ends with " (copy)"
    service, _ = _build_service()
    original = await service.create(
        user_id="uid-1",
        request=CreatePipelineInput(name="Alpha (copy)", stages=_canonical_stages()),
    )

    # WHEN we clone it
    cloned = await service.clone(user_id="uid-1", pipeline_id=original.pipeline_id)

    # THEN the name is left alone
    expectedName = "Alpha (copy)"
    assert cloned.name == expectedName


async def test_delete_removes_the_pipeline() -> None:
    # GIVEN a persisted pipeline
    service, _ = _build_service()
    created = await service.create(
        user_id="uid-1",
        request=CreatePipelineInput(name="Alpha", stages=_canonical_stages()),
    )

    # WHEN we delete it
    await service.delete(user_id="uid-1", pipeline_id=created.pipeline_id)

    # THEN it's gone
    assert await service.list_for_user("uid-1") == []


async def test_delete_of_readonly_pipeline_raises() -> None:
    # GIVEN a seeded default
    service, _ = _build_service()
    seeded = await service.ensure_default(
        user_id="uid-1", default_config=_default_config()
    )

    # WHEN we try to delete it
    # THEN ReadonlyPipelineError
    with pytest.raises(ReadonlyPipelineError):
        await service.delete(user_id="uid-1", pipeline_id=seeded.pipeline_id)


async def test_ensure_default_creates_when_missing_and_activates_it() -> None:
    # GIVEN a user with no pipelines
    service, _ = _build_service()

    # WHEN we ensure the default
    seeded = await service.ensure_default(
        user_id="uid-1", default_config=_default_config()
    )

    # THEN one Default Tabiya row appears, active + readonly + default
    assert seeded.name == DEFAULT_TABIYA_NAME
    assert seeded.is_default is True
    assert seeded.is_readonly is True
    assert seeded.is_active is True


async def test_ensure_default_is_idempotent() -> None:
    # GIVEN a user who already has the default
    service, _ = _build_service()
    first_seed = await service.ensure_default(
        user_id="uid-1", default_config=_default_config()
    )

    # WHEN we call ensure_default again
    second_seed = await service.ensure_default(
        user_id="uid-1", default_config=_default_config()
    )

    # THEN the same doc comes back, no new row is inserted
    assert second_seed.pipeline_id == first_seed.pipeline_id
    listed = await service.list_for_user("uid-1")
    assert len(listed) == 1


async def test_ensure_default_does_not_deactivate_an_existing_active_pipeline() -> None:
    # GIVEN a user with an already-active custom pipeline
    service, _ = _build_service()
    created = await service.create(
        user_id="uid-1",
        request=CreatePipelineInput(name="Custom", stages=_canonical_stages()),
    )
    await service.activate(user_id="uid-1", pipeline_id=created.pipeline_id)

    # WHEN we ensure the default
    seeded = await service.ensure_default(
        user_id="uid-1", default_config=_default_config()
    )

    # THEN the seed is inserted but NOT active
    assert seeded.is_active is False
    # AND the previous active is still active
    listed_by_id = {doc.pipeline_id: doc for doc in await service.list_for_user("uid-1")}
    assert listed_by_id[created.pipeline_id].is_active is True


async def test_get_of_missing_id_raises_not_found() -> None:
    # GIVEN no pipelines
    service, _ = _build_service()

    # WHEN we get a missing id
    # THEN PipelineNotFoundError
    with pytest.raises(PipelineNotFoundError):
        await service.get(user_id="uid-1", pipeline_id="ghost")


def test_validate_returns_issues_without_persisting() -> None:
    # GIVEN a bad set of stages
    service, _ = _build_service()
    givenStages = [
        StageDocument(plugin_id="tabiya.source.text.v1", config={"text": ""}),
    ]

    # WHEN we validate
    issues = service.validate(givenStages)

    # THEN issues are returned (nothing persisted — validate is sync/read-only)
    assert issues
