"""Health — the response body of `GET /plugin/health`.

`degraded` means the plugin is up but a downstream dependency (embeddings
cache, remote API, model file) is impaired. `down` means the plugin cannot
serve any invoke calls.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class HealthStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"


class Health(BaseModel):
    status: HealthStatus
    detail: Optional[str] = None
