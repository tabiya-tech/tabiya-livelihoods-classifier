from ner_experiments.evaluation.bootstrap import paired_bootstrap
from ner_experiments.schemas import PredictedSentence


def _sentence(ad_id: str, gold: list[str], predicted: list[str]) -> PredictedSentence:
    return PredictedSentence(
        ad_id=ad_id, sentence_index=0, tokens=["w"] * len(gold), tags=gold, predicted_tags=predicted
    )


def _predictions(correct_ads: int, total_ads: int) -> list[PredictedSentence]:
    gold = ["B-Skill", "B-Occupation", "B-Qualification"]
    return [
        _sentence(f"ad/{ad_number}", gold, gold if ad_number < correct_ads else ["O", "O", "O"])
        for ad_number in range(total_ads)
    ]


def test_a_clearly_better_model_is_not_tied():
    # GIVEN one model right on 90 of 100 ads and another right on 30
    result = paired_bootstrap(
        {"strong": [_predictions(90, 100)], "weak": [_predictions(30, 100)]}, split="dev", resamples=200
    )

    # THEN the gap is real and the strong model ranks first
    assert result.models[0].model_name == "strong"
    assert result.gaps[0].better_model == "strong" and not result.gaps[0].tied


def test_identical_models_are_tied():
    predictions = _predictions(60, 100)
    result = paired_bootstrap({"first": [predictions], "second": [predictions]}, split="dev", resamples=200)
    assert result.gaps[0].tied
