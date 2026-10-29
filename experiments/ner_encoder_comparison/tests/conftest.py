import os

import pytest


@pytest.fixture(autouse=True)
def _clear_experiment_environment_variables(monkeypatch: pytest.MonkeyPatch):
    """Settings read NER_EXP_* / NER_PREP_* variables; ones left set in the terminal must not leak into tests."""
    for name in list(os.environ):
        if name.startswith(("NER_EXP_", "NER_PREP_")):
            monkeypatch.delenv(name)
