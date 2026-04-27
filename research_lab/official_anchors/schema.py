"""Schema canónico del Official Anchor Events Pipeline (free / fuentes oficiales)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Columnas del CSV canónico local (dataset de laboratorio; no integrado al motor).
CANONICAL_ANCHOR_COLUMNS: tuple[str, ...] = (
    "event_id",
    "source",
    "source_type",
    "title",
    "country",
    "currency",
    "importance",
    "anchor_group",
    "scheduled_at_utc",
    "scheduled_at_ny",
    "timezone_source",
    "is_dst_sensitive",
    "status",
    "source_approved",
    "operational_eligible",
    "source_url",
    "notes",
)


@dataclass
class IntermediateEvent:
    """Salida normalizada de un conector antes de validación temporal final."""

    title: str
    country: str
    currency: str
    local_date_ny: str  # YYYY-MM-DD (fecha civil en calendario NY del día del release)
    local_time_ny: str  # HH:MM oficial publicado (America/New_York)
    source: str
    source_type: str
    source_url: str
    anchor_group: str
    importance: str = "HIGH"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "country": self.country,
            "currency": self.currency,
            "local_date_ny": self.local_date_ny,
            "local_time_ny": self.local_time_ny,
            "source": self.source,
            "source_type": self.source_type,
            "source_url": self.source_url,
            "anchor_group": self.anchor_group,
            "importance": self.importance,
            "notes": self.notes,
        }
