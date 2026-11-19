"""Tests for BIO post-processing used by the NER service."""

from shared.bio_utils import best_valid_bio_tags, drop_edge_punctuation, extract_entities


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


def test_best_valid_bio_tags_never_starts_a_phrase_with_inside():
    # GIVEN a word whose single most likely tag is an orphan I-Skill
    labels = ["O", "B-Skill", "I-Skill"]
    log_probs = [[-0.2, -2.0, -3.0], [-1.0, -2.0, -0.6], [-1.0, -2.0, -0.5]]
    # WHEN the whole sentence is decoded at once
    tags = best_valid_bio_tags(log_probs, labels)
    # THEN no I- appears without a B- before it
    assert tags[0] in ("O", "B-Skill")
    for previous, tag in zip(tags, tags[1:]):
        if tag == "I-Skill":
            assert previous in ("B-Skill", "I-Skill")


def test_best_valid_bio_tags_keeps_a_confident_phrase():
    labels = ["O", "B-Skill", "I-Skill"]
    log_probs = [[-3.0, -0.1, -3.0], [-3.0, -3.0, -0.1], [-0.1, -3.0, -3.0]]
    assert best_valid_bio_tags(log_probs, labels) == ["B-Skill", "I-Skill", "O"]


def test_best_valid_bio_tags_empty_sentence():
    assert best_valid_bio_tags([], ["O"]) == []


def test_drop_edge_punctuation_trims_a_trailing_bracket():
    words = ["gestión", "del", "portal", "("]
    tags = ["B-Skill", "I-Skill", "I-Skill", "I-Skill"]
    assert drop_edge_punctuation(words, tags) == ["B-Skill", "I-Skill", "I-Skill", "O"]


def test_drop_edge_punctuation_moves_the_start_past_punctuation():
    words = ["(", "Excel", "avanzado"]
    tags = ["B-Skill", "I-Skill", "I-Skill"]
    assert drop_edge_punctuation(words, tags) == ["O", "B-Skill", "I-Skill"]


def test_drop_edge_punctuation_removes_punctuation_only_phrases():
    assert drop_edge_punctuation([",", "y"], ["B-Skill", "O"]) == ["O", "O"]


def test_drop_edge_punctuation_keeps_a_bracket_closed_inside_the_phrase():
    words = ["clases", "en", "vivo", "(", "sincrónicas", ")", "."]
    tags = ["B-Skill"] + ["I-Skill"] * 6
    assert drop_edge_punctuation(words, tags) == ["B-Skill"] + ["I-Skill"] * 5 + ["O"]


def test_drop_edge_punctuation_keeps_a_phrase_that_is_a_whole_bracket():
    words = ["(", "Excel", ")"]
    tags = ["B-Skill", "I-Skill", "I-Skill"]
    assert drop_edge_punctuation(words, tags) == tags
