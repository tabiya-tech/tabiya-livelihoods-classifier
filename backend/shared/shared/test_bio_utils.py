"""Tests for BIO post-processing used by the NER service."""

from shared.bio_utils import extract_entities


class TestExtractEntities:
    def test_adjacent_spans_of_the_same_type_stay_separate(self):
        # GIVEN two skills tagged back to back, the second starting with B-
        tokens = ["Excel", "Word"]
        tags = ["B-Skill", "B-Skill"]

        # WHEN extracting entities
        entities = extract_entities(tokens, tags)

        # THEN each B- tag starts its own entity
        assert entities == [
            {"type": "Skill", "tokens": ["Excel"]},
            {"type": "Skill", "tokens": ["Word"]},
        ]

    def test_inside_tags_continue_the_current_entity(self):
        # GIVEN a two-word occupation
        tokens = ["Head", "Chef", "needed"]
        tags = ["B-Occupation", "I-Occupation", "O"]

        # WHEN extracting entities
        # THEN the I- tag extends the entity the B- tag started
        assert extract_entities(tokens, tags) == [
            {"type": "Occupation", "tokens": ["Head", "Chef"]}
        ]

    def test_an_inside_tag_of_a_different_type_starts_a_new_entity(self):
        # GIVEN an I- tag whose type differs from the entity before it
        tokens = ["cook", "food", "safety"]
        tags = ["B-Occupation", "I-Skill", "I-Skill"]

        # WHEN extracting entities
        # THEN the type change starts a new entity
        assert extract_entities(tokens, tags) == [
            {"type": "Occupation", "tokens": ["cook"]},
            {"type": "Skill", "tokens": ["food", "safety"]},
        ]

    def test_outside_tags_end_the_current_entity(self):
        # GIVEN two skills separated by an untagged word
        tokens = ["Excel", "and", "Word"]
        tags = ["B-Skill", "O", "B-Skill"]

        # WHEN extracting entities
        # THEN they come back as two entities
        assert extract_entities(tokens, tags) == [
            {"type": "Skill", "tokens": ["Excel"]},
            {"type": "Skill", "tokens": ["Word"]},
        ]
