"""Installed plugins for the Tabiya-IO bundle."""

from __future__ import annotations

from typing import Awaitable, Callable, Optional

from tabiya_plugin_contracts import Manifest, Health

from .database import MANIFEST as DATABASE_MANIFEST
from .database import health as database_health
from .database import invoke as database_invoke
from .json_entities import MANIFEST as JSON_ENTITIES_MANIFEST
from .json_entities import invoke as json_entities_invoke
from .json_input import MANIFEST as JSON_INPUT_MANIFEST
from .json_input import invoke as json_input_invoke
from .results import MANIFEST as RESULTS_MANIFEST
from .results import invoke as results_invoke
from .scraper import MANIFEST as SCRAPER_MANIFEST
from .scraper import health as scraper_health
from .scraper import invoke as scraper_invoke
from .stopwords import MANIFEST as STOPWORDS_MANIFEST
from .stopwords import health as stopwords_health
from .stopwords import invoke as stopwords_invoke
from .text_input import MANIFEST as TEXT_INPUT_MANIFEST
from .text_input import invoke as text_input_invoke


HealthFn = Callable[[], Awaitable[Health]]


INSTALLED_PLUGINS: list[tuple[Manifest, object, Optional[HealthFn]]] = [
    (TEXT_INPUT_MANIFEST, text_input_invoke, None),
    (JSON_INPUT_MANIFEST, json_input_invoke, None),
    (JSON_ENTITIES_MANIFEST, json_entities_invoke, None),
    (RESULTS_MANIFEST, results_invoke, None),
    # Coming-soon plugins: real manifests, no implementation yet.
    (SCRAPER_MANIFEST, scraper_invoke, scraper_health),
    (STOPWORDS_MANIFEST, stopwords_invoke, stopwords_health),
    (DATABASE_MANIFEST, database_invoke, database_health),
]
