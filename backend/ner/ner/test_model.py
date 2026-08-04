"""Tests for the checkpoint-label -> pipeline-entity-type resolution.

``tabiya/roberta-base-job-ner`` tags five label families (Skill, Occupation,
Qualification, Experience, Domain) but the pipeline only speaks the first three. A
label with no pipeline equivalent has to be dropped: returning it made
``NERResponse`` raise a pydantic ``ValidationError``, which ``main.py`` catches as a
``ValueError`` and turns into a 400 for the whole request — so one Domain span cost
every usable entity in that job.
"""

from ner.model import PIPELINE_ENTITY_TYPES, resolve_entity_type


class TestResolveEntityType:
    def test_pipeline_types_are_the_response_models_enum(self):
        # GIVEN the entity types the NER response model accepts
        # THEN they are exactly the ones the resolver keeps
        assert PIPELINE_ENTITY_TYPES == {"occupation", "skill", "qualification"}

    def test_pipeline_labels_pass_through_lowercased(self):
        # GIVEN a checkpoint label that already names a pipeline type
        # WHEN resolving it without a label map
        # THEN it comes back lowercased
        assert resolve_entity_type("Skill") == "skill"
        assert resolve_entity_type("Occupation") == "occupation"
        assert resolve_entity_type("Qualification") == "qualification"

    def test_labels_outside_the_pipeline_vocabulary_are_dropped(self):
        # GIVEN the extra labels tabiya/roberta-base-job-ner emits
        # WHEN resolving them
        # THEN there is no pipeline type to report them as
        assert resolve_entity_type("Experience") is None
        assert resolve_entity_type("Domain") is None

    def test_label_map_renames_a_checkpoints_own_vocabulary(self):
        # GIVEN a checkpoint whose labels need renaming onto the pipeline's types
        label_map = {"knowledge": "skill", "profession": "occupation"}

        # WHEN resolving a mapped label
        # THEN the mapped pipeline type is returned
        assert resolve_entity_type("Knowledge", label_map) == "skill"
        assert resolve_entity_type("profession", label_map) == "occupation"

    def test_label_map_can_route_an_extra_label_into_the_pipeline(self):
        # GIVEN a deployment that wants Domain spans linked as skills
        label_map = {"domain": "skill"}

        # WHEN resolving Domain
        # THEN it is kept as a skill rather than dropped
        assert resolve_entity_type("Domain", label_map) == "skill"

    def test_label_map_to_a_non_pipeline_type_still_drops(self):
        # GIVEN a label map with a target that is not a pipeline type
        label_map = {"domain": "sector"}

        # WHEN resolving it
        # THEN the span is still dropped, rather than failing response validation
        assert resolve_entity_type("Domain", label_map) is None
