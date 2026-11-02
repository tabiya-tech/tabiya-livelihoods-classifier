"""Tests for BIO post-processing used by the NER service."""

from shared.bio_utils import extract_entities


class TestExtractEntities:
    def test_adjacent_spans_of_the_same_type_stay_separate(self):
        # GIVEN two skills tagged back to back, the second starting with B-
        given_tokens = ["Excel", "Word"]
        given_tags = ["B-Skill", "B-Skill"]

        # WHEN extracting entities
        actual_entities = extract_entities(given_tokens, given_tags)

        # THEN each B- tag starts its own entity
        expected_entities = [
            {"type": "Skill", "tokens": ["Excel"]},
            {"type": "Skill", "tokens": ["Word"]},
        ]
        assert actual_entities == expected_entities

    def test_inside_tags_continue_the_current_entity(self):
        # GIVEN a two-word occupation followed by an untagged word
        given_tokens = ["Head", "Chef", "needed"]
        given_tags = ["B-Occupation", "I-Occupation", "O"]

        # WHEN extracting entities
        actual_entities = extract_entities(given_tokens, given_tags)

        # THEN the I- tag extends the entity the B- tag started
        expected_entities = [{"type": "Occupation", "tokens": ["Head", "Chef"]}]
        assert actual_entities == expected_entities

    def test_an_inside_tag_of_a_different_type_starts_a_new_entity(self):
        # GIVEN an I- tag whose type differs from the entity before it
        given_tokens = ["cook", "food", "safety"]
        given_tags = ["B-Occupation", "I-Skill", "I-Skill"]

        # WHEN extracting entities
        actual_entities = extract_entities(given_tokens, given_tags)

        # THEN the type change starts a new entity
        expected_entities = [
            {"type": "Occupation", "tokens": ["cook"]},
            {"type": "Skill", "tokens": ["food", "safety"]},
        ]
        assert actual_entities == expected_entities

    def test_outside_tags_end_the_current_entity(self):
        # GIVEN two skills separated by an untagged word
        given_tokens = ["Excel", "and", "Word"]
        given_tags = ["B-Skill", "O", "B-Skill"]

        # WHEN extracting entities
        actual_entities = extract_entities(given_tokens, given_tags)

        # THEN they come back as two entities
        expected_entities = [
            {"type": "Skill", "tokens": ["Excel"]},
            {"type": "Skill", "tokens": ["Word"]},
        ]
        assert actual_entities == expected_entities

    def test_labels_containing_hyphens_keep_their_full_type(self):
        # GIVEN a checkpoint whose label itself contains a hyphen
        given_tokens = ["Head", "Chef"]
        given_tags = ["B-Job-Title", "I-Job-Title"]

        # WHEN extracting entities
        actual_entities = extract_entities(given_tokens, given_tags)

        # THEN only the B-/I- prefix is split off, so the label stays whole
        expected_entities = [{"type": "Job-Title", "tokens": ["Head", "Chef"]}]
        assert actual_entities == expected_entities
