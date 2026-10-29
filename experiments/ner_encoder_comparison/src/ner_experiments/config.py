"""Experiment settings, typed and validated.

Each run is controlled by one YAML file (see ``configs/runs/``). Any setting can also be
overridden with an environment variable, which wins over the YAML file:

    NER_EXP_SEED=3 python -m ner_experiments.cli.run_experiment --config configs/runs/xlmr.yaml

Nested settings use a double underscore, e.g. ``NER_EXP_PATHS__RUNS_DIR=D:/runs``.
Default paths are inside this experiment folder, whichever folder you run from.
Priority: environment variable > YAML file > the defaults below.
"""

from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

PERU_CORPUS_URL = (
    "https://github.com/ronaldahmed/labor-market-demand-analysis/raw/master/annotated%20data.zip"
)


class _EnvironmentOverYamlSettings(BaseSettings):
    """Base class: environment variables take priority over values passed in (from YAML)."""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (env_settings, init_settings)


# The experiment folder (the one holding pyproject.toml). Default paths hang off it, so the
# commands work the same from any folder. Requires the editable install (pip install -e .).
EXPERIMENT_DIR = Path(__file__).resolve().parents[2]


class DataPaths(BaseModel):
    """Where data, frozen splits, run outputs and reports live."""

    raw_dir: Path = EXPERIMENT_DIR / "data" / "raw"
    processed_dir: Path = EXPERIMENT_DIR / "data" / "processed"
    splits_dir: Path = EXPERIMENT_DIR / "splits"
    runs_dir: Path = EXPERIMENT_DIR / "runs"
    reports_dir: Path = EXPERIMENT_DIR / "reports"


class PrepareConfig(_EnvironmentOverYamlSettings):
    """Settings for turning the Peru corpus into train/dev/test files."""

    model_config = SettingsConfigDict(env_prefix="NER_PREP_", env_nested_delimiter="__", extra="forbid")

    paths: DataPaths = DataPaths()
    corpus_url: str = PERU_CORPUS_URL
    task_description_spans: Literal["ignore", "skill"] = Field(
        default="ignore",
        description="What to do with Peru's FUN tags (task descriptions): drop them, or tag them as Skill",
    )
    move_qualification_levels: bool = Field(
        default=True,
        description="Move REQ phrases like 'Técnico' or 'postgrado' from Skill to Qualification",
    )
    train_share: float = Field(default=0.7, gt=0, lt=1)
    dev_share: float = Field(default=0.1, gt=0, lt=1)
    test_share: float = Field(default=0.2, gt=0, lt=1)
    split_seed: int = 13

    @model_validator(mode="after")
    def _shares_add_up(self) -> "PrepareConfig":
        total = self.train_share + self.dev_share + self.test_share
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"train/dev/test shares must add up to 1, got {total}")
        return self


class ExperimentConfig(_EnvironmentOverYamlSettings):
    """Settings for one training run. Set these per run, by hand."""

    model_config = SettingsConfigDict(env_prefix="NER_EXP_", env_nested_delimiter="__", extra="forbid")

    run_name: str | None = Field(default=None, description="Folder name under runs/; built from the settings if empty")
    model_name: str = "FacebookAI/xlm-roberta-base"
    seed: int = Field(default=1, description="Random seed for this run (repeat runs use 1 to 5)")

    learning_rate: float = Field(default=3e-5, gt=0)
    max_epochs: int = Field(default=15, ge=1, description="Upper limit on passes over the training data")
    early_stopping_patience: int = Field(default=3, ge=1, description="Stop after this many passes without a dev gain")
    batch_size: int = Field(default=16, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    warmup_ratio: float = Field(default=0.1, ge=0, lt=1)
    weight_decay: float = Field(default=0.01, ge=0)
    max_pieces_per_sentence: int = Field(default=256, ge=16, description="Longer sentences are split into chunks")
    train_fraction: float = Field(default=1.0, gt=0, le=1, description="Share of training ads to use (learning curve)")

    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    precision: Literal["auto", "bf16", "fp16", "fp32"] = "auto"
    dataloader_workers: int = Field(default=0, ge=0, description="Keep 0 on Windows")

    evaluate_on_test: bool = Field(
        default=False,
        description="Score the locked test set. Leave off while tuning; turn on only for the final repeat runs",
    )
    keep_model: bool = Field(default=False, description="Keep the trained model (about 0.6 to 1.2 GB per run)")
    overwrite: bool = Field(default=False, description="Replace an existing run folder with the same name")

    paths: DataPaths = DataPaths()

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        model_short_name = self.model_name.split("/")[-1].lower()
        name = f"{model_short_name}-lr{self.learning_rate:g}-seed{self.seed}"
        if self.train_fraction < 1:
            name += f"-data{round(self.train_fraction * 100)}"
        return name

    def run_dir(self) -> Path:
        return self.paths.runs_dir / self.resolved_run_name()


SettingsType = TypeVar("SettingsType", bound=BaseSettings)


def load_settings(settings_class: type[SettingsType], yaml_path: Path | None) -> SettingsType:
    """Build settings from an optional YAML file, with environment variables on top."""
    yaml_values: dict[str, Any] = {}
    if yaml_path is not None:
        with open(yaml_path, encoding="utf-8") as yaml_file:
            yaml_values = yaml.safe_load(yaml_file) or {}
        if not isinstance(yaml_values, dict):
            raise ValueError(f"{yaml_path} must contain a mapping of setting names to values")
    return settings_class(**yaml_values)
