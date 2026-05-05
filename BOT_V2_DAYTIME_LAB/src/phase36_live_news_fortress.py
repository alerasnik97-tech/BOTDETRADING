from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "MANIPULANTE" / "09_COMPLIANCE" / "live_news_fortress_config.json"
CACHE_DIR = ROOT / "MANIPULANTE" / "09_COMPLIANCE" / "live_news_cache"
LOG_DIR = ROOT / "MANIPULANTE" / "10_LOGS_PAPER" / "news_gate"
NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class NewsEvent:
    event_id: str
    name: str
    currency: str
    impact: str
    event_time_utc: datetime
    source: str
    actual: str | None = None
    forecast: str | None = None
    previous: str | None = None

    @property
    def event_time_ny(self) -> datetime:
        return self.event_time_utc.astimezone(NY)


class LiveNewsFortress:
    """Fail-closed live-news gate for MANIPULANTE.

    The module intentionally does not scrape web calendars. It accepts only a
    validated MT5/MQL5 calendar cache or an emergency manual file explicitly
    marked VERIFIED_BY_USER. When the source is missing, stale, ambiguous or
    malformed, the gate returns NO_TRADE/ERROR_FAIL_CLOSED.
    """

    def __init__(self, config_path: Path = CONFIG_PATH) -> None:
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.cache_dir = ROOT / self.config.get("cache_dir", "MANIPULANTE/09_COMPLIANCE/live_news_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {
                "profile": "MANIPULANTE_LIVE_NEWS_FORTRESS",
                "currencies": ["EUR", "USD"],
                "impact_filter": ["HIGH"],
                "guard_minutes_before": 30,
                "guard_minutes_after": 30,
                "fail_closed": True,
                "allow_trade_if_news_source_unavailable": False,
                "manual_override_allowed": False,
                "cache_enabled": True,
                "cache_max_age_minutes": 60,
                "today_required": True,
                "week_ahead_required": True,
                "mode": "DRY_RUN_ONLY",
            }
        return json.loads(self.config_path.read_text(encoding="utf-8"))

    def _parse_dt(self, value: str) -> datetime:
        if not value:
            raise ValueError("empty datetime")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("timezone missing")
        return parsed.astimezone(timezone.utc)

    def _cache_candidates(self, day: datetime) -> list[Path]:
        date_key = day.astimezone(NY).strftime("%Y-%m-%d")
        return [
            self.cache_dir / f"{date_key}_news_gate.json",
            self.cache_dir / f"{date_key}_news_week.json",
            self.cache_dir / f"{date_key}_news_today.json",
            self.cache_dir / "latest_news_gate.json",
            self.cache_dir / "manual_verified_news_gate.json",
        ]

    def _load_cache_payload(self, day: datetime) -> tuple[dict[str, Any] | None, str]:
        for path in self._cache_candidates(day):
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                return None, f"ERROR_FAIL_CLOSED: malformed cache {path.name}: {exc}"
            source_type = str(payload.get("source_type", "")).upper()
            verified = bool(payload.get("verified_by_mt5") or payload.get("VERIFIED_BY_USER"))
            if source_type not in {"MT5_MQL5_ECONOMIC_CALENDAR", "MANUAL_EMERGENCY_VERIFIED"}:
                return None, f"NO_TRADE_NEWS_SOURCE_UNAVAILABLE: unsupported source {source_type}"
            if source_type == "MANUAL_EMERGENCY_VERIFIED" and not payload.get("VERIFIED_BY_USER"):
                return None, "NO_TRADE_NEWS_SOURCE_UNAVAILABLE: manual cache not VERIFIED_BY_USER"
            if not verified:
                return None, "NO_TRADE_NEWS_SOURCE_UNAVAILABLE: cache not verified"
            generated_at_raw = payload.get("generated_at_utc")
            generated_at = self._parse_dt(generated_at_raw)
            age_minutes = (datetime.now(timezone.utc) - generated_at).total_seconds() / 60.0
            max_age = float(self.config.get("cache_max_age_minutes", 60))
            if age_minutes > max_age:
                return None, f"NO_TRADE_NEWS_SOURCE_UNAVAILABLE: cache stale {age_minutes:.1f}m"
            return payload, "OK"
        return None, "NO_TRADE_NEWS_SOURCE_UNAVAILABLE: no MT5/manual verified cache"

    def _events_from_payload(self, payload: dict[str, Any]) -> list[NewsEvent]:
        required_currencies = set(self.config.get("currencies", ["EUR", "USD"]))
        impact_filter = {str(x).upper() for x in self.config.get("impact_filter", ["HIGH"])}
        raw_events = payload.get("events", [])
        if not isinstance(raw_events, list):
            raise ValueError("events is not a list")
        events: list[NewsEvent] = []
        seen_currencies: set[str] = set()
        for raw in raw_events:
            currency = str(raw.get("currency", "")).upper()
            impact = self.classify_event(raw)
            if currency in required_currencies:
                seen_currencies.add(currency)
            if currency not in required_currencies or impact not in impact_filter:
                continue
            events.append(
                NewsEvent(
                    event_id=str(raw.get("event_id", raw.get("id", ""))),
                    name=str(raw.get("name", raw.get("event_name", "UNKNOWN_EVENT"))),
                    currency=currency,
                    impact=impact,
                    event_time_utc=self._parse_dt(str(raw.get("time_utc", raw.get("event_time_utc", "")))),
                    source=str(payload.get("source_type", "UNKNOWN")),
                    actual=self._optional(raw.get("actual")),
                    forecast=self._optional(raw.get("forecast")),
                    previous=self._optional(raw.get("previous")),
                )
            )
        if required_currencies - seen_currencies:
            missing = ",".join(sorted(required_currencies - seen_currencies))
            raise ValueError(f"missing required currencies in source: {missing}")
        return sorted(events, key=lambda item: item.event_time_utc)

    def _optional(self, value: Any) -> str | None:
        if value in (None, "", "LONG_MIN"):
            return None
        return str(value)

    def load_today_news(self) -> tuple[list[NewsEvent], str]:
        now = datetime.now(timezone.utc)
        payload, status = self._load_cache_payload(now)
        if payload is None:
            return [], status
        try:
            start_ny = now.astimezone(NY).replace(hour=0, minute=0, second=0, microsecond=0)
            end_ny = start_ny + timedelta(days=1)
            events = [
                event for event in self._events_from_payload(payload)
                if start_ny <= event.event_time_ny < end_ny
            ]
            return events, "OK"
        except Exception as exc:
            return [], f"ERROR_FAIL_CLOSED: {exc}"

    def load_week_news(self) -> tuple[list[NewsEvent], str]:
        now = datetime.now(timezone.utc)
        payload, status = self._load_cache_payload(now)
        if payload is None:
            return [], status
        try:
            start_ny = now.astimezone(NY).replace(hour=0, minute=0, second=0, microsecond=0)
            end_ny = start_ny + timedelta(days=7)
            events = [
                event for event in self._events_from_payload(payload)
                if start_ny <= event.event_time_ny < end_ny
            ]
            return events, "OK"
        except Exception as exc:
            return [], f"ERROR_FAIL_CLOSED: {exc}"

    def classify_event(self, event: dict[str, Any]) -> str:
        raw = str(event.get("impact", event.get("importance", ""))).upper()
        if raw in {"HIGH", "CALENDAR_IMPORTANCE_HIGH", "3"}:
            return "HIGH"
        if raw in {"MEDIUM", "CALENDAR_IMPORTANCE_MODERATE", "2"}:
            return "MEDIUM"
        if raw in {"LOW", "CALENDAR_IMPORTANCE_LOW", "1"}:
            return "LOW"
        raise ValueError(f"unknown impact {raw!r}")

    def is_blocked_now(self, now: datetime | None = None) -> tuple[bool, str, NewsEvent | None]:
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        events, status = self.load_week_news()
        if status != "OK":
            return True, status, None
        before = timedelta(minutes=int(self.config.get("guard_minutes_before", 30)))
        after = timedelta(minutes=int(self.config.get("guard_minutes_after", 30)))
        for event in events:
            if event.event_time_utc - before <= now <= event.event_time_utc + after:
                return True, "NO_TRADE_NEWS_WINDOW", event
        return False, "ALLOW", None

    def next_blocking_event(self, now: datetime | None = None) -> NewsEvent | None:
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        events, status = self.load_week_news()
        if status != "OK":
            return None
        for event in events:
            if event.event_time_utc >= now:
                return event
        return None

    def get_news_gate_status(self) -> dict[str, Any]:
        blocked, status, event = self.is_blocked_now()
        next_event = self.next_blocking_event()
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "profile": self.config.get("profile"),
            "mode": self.config.get("mode"),
            "gate": "NO_TRADE" if blocked else "ALLOW",
            "status": status,
            "blocking_event": self._event_to_dict(event),
            "next_blocking_event": self._event_to_dict(next_event),
            "fail_closed": bool(self.config.get("fail_closed", True)),
            "order_send_allowed": False,
        }

    def _event_to_dict(self, event: NewsEvent | None) -> dict[str, Any] | None:
        if event is None:
            return None
        return {
            "event_id": event.event_id,
            "name": event.name,
            "currency": event.currency,
            "impact": event.impact,
            "event_time_utc": event.event_time_utc.isoformat(),
            "event_time_ny": event.event_time_ny.isoformat(),
            "source": event.source,
            "actual": event.actual,
            "forecast": event.forecast,
            "previous": event.previous,
        }

    def write_news_gate_log(self) -> Path:
        status = self.get_news_gate_status()
        day = datetime.now(NY).strftime("%Y-%m-%d")
        path = LOG_DIR / f"{day}_news_gate_status.json"
        path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
        return path


def load_today_news() -> tuple[list[NewsEvent], str]:
    return LiveNewsFortress().load_today_news()


def load_week_news() -> tuple[list[NewsEvent], str]:
    return LiveNewsFortress().load_week_news()


def classify_event(event: dict[str, Any]) -> str:
    return LiveNewsFortress().classify_event(event)


def is_blocked_now() -> tuple[bool, str, NewsEvent | None]:
    return LiveNewsFortress().is_blocked_now()


def next_blocking_event() -> NewsEvent | None:
    return LiveNewsFortress().next_blocking_event()


def get_news_gate_status() -> dict[str, Any]:
    return LiveNewsFortress().get_news_gate_status()


def write_news_gate_log() -> Path:
    return LiveNewsFortress().write_news_gate_log()


if __name__ == "__main__":
    gate = LiveNewsFortress()
    output = gate.get_news_gate_status()
    gate.write_news_gate_log()
    print(json.dumps(output, indent=2, ensure_ascii=False))
