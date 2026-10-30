"""Tests for BIO post-processing used by the NER service."""

from shared.bio_utils import extract_entities, spread_first_piece_tags


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


class TestSpreadFirstPieceTags:
    def test_continuation_pieces_continue_the_first_pieces_entity(self):
        # GIVEN "Mantenimiento" split into three pieces, only the first trained as B-Skill
        word_ids = [None, 0, 0, 0, None]
        tags = ["O", "B-Skill", "O", "B-Occupation", "O"]

        # WHEN spreading first-piece tags
        # THEN the other pieces continue the skill and special tokens are untouched
        assert spread_first_piece_tags(tags, word_ids) == [
            "O", "B-Skill", "I-Skill", "I-Skill", "O"
        ]

    def test_words_outside_entities_stay_outside(self):
        # GIVEN a word whose first piece is O but a later piece was tagged
        word_ids = [0, 0, 1]
        tags = ["O", "I-Skill", "B-Skill"]

        # WHEN spreading first-piece tags
        # THEN the stray continuation tag is dropped
        assert spread_first_piece_tags(tags, word_ids) == ["O", "O", "B-Skill"]

    def test_single_piece_words_are_unchanged(self):
        # GIVEN every word is one piece
        word_ids = [None, 0, 1, None]
        tags = ["O", "B-Skill", "B-Skill", "O"]

        # WHEN spreading first-piece tags
        # THEN nothing changes
        assert spread_first_piece_tags(tags, word_ids) == tags
