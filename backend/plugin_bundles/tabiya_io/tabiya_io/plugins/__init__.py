"""Installed plugins for the Tabiya-IO bundle."""

from __future__ import annotations

from typing import Awaitable, Callable, Optional

from tabiya_plugin_contracts import Manifest, Health

from .results import MANIFEST as RESULTS_MANIFEST
from .results import invoke as results_invoke
from .text_input import MANIFEST as TEXT_INPUT_MANIFEST
from .text_input import invoke as text_input_invoke


HealthFn = Callable[[], Awaitable[Health]]


INSTALLED_PLUGINS: list[tuple[Manifest, object, Optional[HealthFn]]] = [
    (TEXT_INPUT_MANIFEST, text_input_invoke, None),
    (RESULTS_MANIFEST, results_invoke, None),
]
