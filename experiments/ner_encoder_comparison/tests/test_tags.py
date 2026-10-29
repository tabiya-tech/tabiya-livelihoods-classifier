from ner_experiments.tags import TAG_NAMES, extract_spans, repair_bio


class TestRepairBio:
    def test_a_stray_inside_tag_starts_a_new_phrase(self):
        # GIVEN an I- tag with nothing of its type before it
        # WHEN repairing the sequence
        # THEN it becomes a B- tag
        assert repair_bio(["O", "I-Skill", "I-Skill"]) == ["O", "B-Skill", "I-Skill"]

    def test_an_inside_tag_of_another_type_starts_a_new_phrase(self):
        assert repair_bio(["B-Occupation", "I-Skill"]) == ["B-Occupation", "B-Skill"]

    def test_valid_sequences_are_unchanged(self):
        tags = ["B-Skill", "I-Skill", "O", "B-Qualification"]
        assert repair_bio(tags) == tags


class TestExtractSpans:
    def test_adjacent_phrases_of_the_same_type_stay_separate(self):
        # GIVEN two skills back to back
        # WHEN extracting phrases
        # THEN each B- tag starts its own phrase
        assert extract_spans(["B-Skill", "B-Skill"]) == [("Skill", 0, 1), ("Skill", 1, 2)]

    def test_a_phrase_runs_until_the_next_outside_tag(self):
        assert extract_spans(["O", "B-Occupation", "I-Occupation", "O"]) == [("Occupation", 1, 3)]

    def test_a_phrase_can_end_the_sentence(self):
        assert extract_spans(["B-Skill", "I-Skill"]) == [("Skill", 0, 2)]


def test_tag_names_are_stable():
    # The tag order is saved inside every trained model; changing it breaks old models.
    assert TAG_NAMES == [
        "O", "B-Occupation", "I-Occupation", "B-Skill", "I-Skill", "B-Qualification", "I-Qualification",
    ]
