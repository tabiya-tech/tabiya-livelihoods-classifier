from ner_experiments.config import EXPERIMENT_DIR, ExperimentConfig, corpus_name
from ner_experiments.corpus.green import GREEN_TAG_NAMES, convert_green_tag, hold_out_dev
from ner_experiments.run_files import predictions_file
from ner_experiments.schemas import TaggedSentence


class TestGreenTags:
    def test_shared_types_are_kept_and_the_others_dropped(self):
        # GIVEN every Green tag number
        converted = {GREEN_TAG_NAMES[i]: convert_green_tag(i) for i in range(len(GREEN_TAG_NAMES))}

        # THEN Skill, Occupation and Qualification survive and Experience and Domain become outside
        assert converted["B-Skill"] == "B-Skill" and converted["I-Occupation"] == "I-Occupation"
        assert converted["B-Qualification"] == "B-Qualification"
        assert converted["B-Domain"] == converted["I-Experience"] == "O"

    def test_the_recovered_tag_order(self):
        # The order the English model was trained with; changing it scrambles every label.
        assert GREEN_TAG_NAMES[1] == "B-Skill" and GREEN_TAG_NAMES[9] == "I-Skill" and GREEN_TAG_NAMES[6] == "B-Occupation"


def _sentence(n: int) -> TaggedSentence:
    return TaggedSentence(ad_id=f"green/train/{n}", sentence_index=0, tokens=["w"], tags=["O"])


class TestHoldOutDev:
    def test_dev_is_disjoint_and_repeatable(self):
        # GIVEN 100 training sentences
        train = [_sentence(n) for n in range(100)]

        # WHEN holding out 10% twice with the same seed
        first, second = hold_out_dev(train, 0.1, 13), hold_out_dev(train, 0.1, 13)

        # THEN 10 go to dev, none are in both, and the split is the same both times
        assert len(first["dev"]) == 10 and len(first["train"]) == 90
        assert not {s.ad_id for s in first["dev"]} & {s.ad_id for s in first["train"]}
        assert [s.ad_id for s in first["dev"]] == [s.ad_id for s in second["dev"]]


def test_mixed_corpora_are_named_and_found_inside_the_experiment_folder():
    settings = ExperimentConfig(mix_with=["data/processed-green"])
    assert settings.mix_with == [EXPERIMENT_DIR / "data" / "processed-green"]
    assert corpus_name(settings.mix_with[0]) == "green"
    assert predictions_file(EXPERIMENT_DIR, "test", corpus="green").name == "predictions_green_test.jsonl"
    assert predictions_file(EXPERIMENT_DIR, "test").name == "predictions_test.jsonl"
