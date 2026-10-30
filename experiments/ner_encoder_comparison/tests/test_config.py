from pathlib import Path

import pytest
from pydantic import ValidationError

from ner_experiments.config import ExperimentConfig, PrepareConfig, load_settings


def test_environment_variables_override_the_yaml_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # GIVEN a run file that sets seed 1, and NER_EXP_SEED=4 in the environment
    config_path = tmp_path / "run.yaml"
    config_path.write_text("model_name: BSC-LT/mRoBERTa\nseed: 1\n", encoding="utf-8")
    monkeypatch.setenv("NER_EXP_SEED", "4")

    # WHEN loading the settings
    settings = load_settings(ExperimentConfig, config_path)

    # THEN the YAML value is used except where the environment overrides it
    assert settings.model_name == "BSC-LT/mRoBERTa"
    assert settings.seed == 4


def test_unknown_settings_are_rejected(tmp_path: Path):
    config_path = tmp_path / "run.yaml"
    config_path.write_text("learnig_rate: 3e-5\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_settings(ExperimentConfig, config_path)


def test_run_names_are_built_from_the_settings():
    settings = ExperimentConfig(model_name="jhu-clsp/mmBERT-base", learning_rate=2e-5, seed=3, train_fraction=0.25)
    assert settings.resolved_run_name() == "mmbert-base-lr2e-05-seed3-data25"


def test_split_shares_must_add_up_to_one():
    with pytest.raises(ValidationError):
        PrepareConfig(train_share=0.8, dev_share=0.1, test_share=0.2)
