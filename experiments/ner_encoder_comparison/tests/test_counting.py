from ner_experiments.evaluation.counting import (
    count_partial_matches,
    count_strict_matches,
    macro_and_micro_f1,
    precision_recall_f1,
)


class TestStrictMatching:
    def test_only_exact_boundaries_count(self):
        # GIVEN a real two-word skill and a prediction covering only its first word
        gold = [["B-Skill", "I-Skill", "O"]]
        predicted = [["B-Skill", "O", "O"]]

        # WHEN counting strict matches
        counts = count_strict_matches(gold, predicted)

        # THEN it is one miss and one false alarm
        assert (counts["Skill"].correct, counts["Skill"].predicted, counts["Skill"].gold) == (0, 1, 1)

    def test_the_wrong_type_does_not_count(self):
        counts = count_strict_matches([["B-Occupation"]], [["B-Skill"]])
        assert counts["Occupation"].correct == 0 and counts["Skill"].predicted == 1


class TestPartialMatching:
    def test_overlap_of_the_same_type_counts(self):
        counts = count_partial_matches([["B-Skill", "I-Skill", "O"]], [["B-Skill", "O", "O"]])
        assert (counts["Skill"].correct, counts["Skill"].predicted, counts["Skill"].gold) == (1, 1, 1)

    def test_splitting_one_phrase_in_two_leaves_one_false_alarm(self):
        # GIVEN one real phrase predicted as two halves
        counts = count_partial_matches([["B-Skill", "I-Skill"]], [["B-Skill", "B-Skill"]])

        # THEN only one half can match it
        assert (counts["Skill"].correct, counts["Skill"].predicted) == (1, 2)


def test_f1_needs_both_precision_and_recall():
    counts = count_strict_matches([["B-Skill", "O", "B-Skill"]], [["B-Skill", "O", "O"]])
    precision, recall, f1 = precision_recall_f1(counts["Skill"])
    assert (precision, recall) == (1.0, 0.5)
    assert abs(f1 - 2 / 3) < 1e-9


def test_macro_averages_the_three_types_equally():
    gold = [["B-Skill", "B-Occupation", "B-Qualification"]]
    predicted = [["B-Skill", "O", "O"]]
    macro_f1, micro_f1 = macro_and_micro_f1(count_strict_matches(gold, predicted))
    assert abs(macro_f1 - 1 / 3) < 1e-9
    assert abs(micro_f1 - 0.5) < 1e-9
