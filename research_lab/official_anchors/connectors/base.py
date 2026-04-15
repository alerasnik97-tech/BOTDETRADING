from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from research_lab.official_anchors.schema import IntermediateEvent


@dataclass
class ConnectorResult:
    """Resultado de un conector: eventos intermedios + estado operativo."""

    connector_id: str
    events: list[IntermediateEvent] = field(default_factory=list)
    status: str = "ok"  # ok | partial | blocked
    message: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
