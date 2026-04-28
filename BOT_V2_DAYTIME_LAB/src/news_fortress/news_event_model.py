
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class NewsEvent:
    event_id: str
    raw_title: str
    normalized_title: str
    event_family: str
    currency: str
    impact_raw: str
    impact_normalized: str
    event_time_utc: datetime
    event_time_ny: Optional[datetime] = None
    source: str = "unknown"
    source_loaded_at: Optional[datetime] = None
    confidence_level: str = "high"
    is_critical: bool = False
    is_ambiguous: bool = False
    block_reason: Optional[str] = None

    def to_dict(self):
        return {
            "event_id": self.event_id,
            "normalized_title": self.normalized_title,
            "currency": self.currency,
            "impact_normalized": self.impact_normalized,
            "event_time_utc": self.event_time_utc.isoformat() if self.event_time_utc else None,
            "is_critical": self.is_critical,
            "block_reason": self.block_reason
        }
