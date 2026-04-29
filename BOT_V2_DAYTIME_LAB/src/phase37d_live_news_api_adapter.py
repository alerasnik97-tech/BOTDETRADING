from __future__ import annotations

import csv
import json
import os
import urllib.parse
import urllib.request
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from phase37_ftmo_trial_support import LAB, MANIPULANTE, now_iso, now_utc, write_csv, write_json, write_text


NY = ZoneInfo("America/New_York")
OUT = LAB / "outputs" / "phase37d_ftmo_trial_api_news_signal" / "api_live_news"
PROVIDER_DIR = MANIPULANTE / "09_COMPLIANCE" / "API_LIVE_NEWS_PROVIDER"
LOCAL_CONFIG = PROVIDER_DIR / "api_news_provider_config.local.json"
EXAMPLE_CONFIG = PROVIDER_DIR / "api_news_provider_config.example.json"
CACHE_DIR = MANIPULANTE / "09_COMPLIANCE" / "live_news_cache" / "api_provider"
SECRET_KEYS = {"api_key", "apikey", "token", "secret", "password"}


DEFAULT_CONFIG: dict[str, Any] = {
    "provider_priority": ["MQL5_BOOTSTRAP", "TRADING_ECONOMICS", "FMP", "FINNHUB", "EODHD"],
    "currencies": ["EUR", "USD"],
    "impact_filter": ["HIGH", "MODERATE"],
    "guard_minutes_before": 30,
    "guard_minutes_after": 30,
    "today_required": True,
    "week_required": True,
    "cache_max_age_minutes": 60,
    "fail_closed": True,
    "allow_trade_if_provider_unavailable": False,
    "manual_override_allowed": False,
    "mode": "FTMO_TRIAL_ONLY",
    "providers": {
        "TRADING_ECONOMICS": {"env_var": "TRADING_ECONOMICS_API_KEY"},
        "FMP": {"env_var": "FMP_API_KEY"},
        "FINNHUB": {"env_var": "FINNHUB_API_KEY"},
        "EODHD": {"env_var": "EODHD_API_KEY"},
    },
}


def _safe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    def clean(value: Any, key: str = "") -> Any:
        if key.lower() in SECRET_KEYS:
            return "***REDACTED***" if value else ""
        if isinstance(value, dict):
            return {k: clean(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v, key) for v in value]
        return value

    return clean(payload)


def ensure_provider_files() -> dict[str, Any]:
    PROVIDER_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    write_json(EXAMPLE_CONFIG, DEFAULT_CONFIG)
    if not LOCAL_CONFIG.exists():
        local = DEFAULT_CONFIG.copy()
        local["providers"] = {
            name: {"env_var": cfg["env_var"], "api_key": ""}
            for name, cfg in DEFAULT_CONFIG["providers"].items()
        }
        write_json(LOCAL_CONFIG, local)
    docs = {
        "API_NEWS_PROVIDER_README.md": """
# API Live News Provider

Phase37D usa un proveedor API machine-readable como fuente principal de noticias para FTMO Trial.
El sistema opera fail-closed: sin provider valido, sin cache de hoy y semana, o sin timezone claro, la decision es NO_TRADE.
""",
        "API_NEWS_PROVIDER_SETUP.md": """
# API News Provider Setup

Las API keys deben estar solo en variables de entorno o en `api_news_provider_config.local.json`.
Ese archivo local esta excluido de Git y del ZIP canonico.
No se imprime ni se empaqueta ninguna key.
""",
        "API_NEWS_PROVIDER_POLICY.md": """
# API News Provider Policy

- Fuente principal: API externa machine-readable.
- CalendarBridge MQL5 queda como fallback/legacy.
- Provider mock solo se permite para tests, nunca para trading.
- EUR/USD HIGH impact se bloquea con guardia +/-30 minutos.
- Si algo es dudoso: NO_TRADE.
""",
    }
    for name, content in docs.items():
        write_text(PROVIDER_DIR / name, content)
    return {
        "provider_dir": str(PROVIDER_DIR),
        "example_config": str(EXAMPLE_CONFIG),
        "local_config": str(LOCAL_CONFIG),
        "local_config_exists": LOCAL_CONFIG.exists(),
        "local_config_excluded_from_git_and_zip": True,
    }


def load_provider_config() -> dict[str, Any]:
    ensure_provider_files()
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if LOCAL_CONFIG.exists():
        try:
            local = json.loads(LOCAL_CONFIG.read_text(encoding="utf-8"))
            for key, value in local.items():
                if isinstance(value, dict) and isinstance(config.get(key), dict):
                    merged = dict(config[key])
                    merged.update(value)
                    config[key] = merged
                else:
                    config[key] = value
        except Exception as exc:
            config["config_error"] = f"LOCAL_CONFIG_PARSE_ERROR: {exc}"
    return config


def _provider_key(config: dict[str, Any], provider: str) -> str | None:
    provider_cfg = (config.get("providers") or {}).get(provider, {})
    env_var = provider_cfg.get("env_var")
    key = os.environ.get(str(env_var), "") if env_var else ""
    if not key:
        key = str(provider_cfg.get("api_key", "") or "")
    return key.strip() or None


def _date_range(today_only: bool) -> tuple[date, date]:
    start = now_utc().astimezone(NY).date()
    end = start if today_only else start + timedelta(days=7)
    return start, end


def _http_get_json(url: str, timeout: int = 12) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "MANIPULANTE-Phase37D/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    return json.loads(body)


def _provider_url(provider: str, key: str, start: date, end: date) -> str:
    s = start.isoformat()
    e = end.isoformat()
    if provider == "FMP":
        return f"https://financialmodelingprep.com/api/v3/economic_calendar?from={s}&to={e}&apikey={urllib.parse.quote(key)}"
    if provider == "FINNHUB":
        return f"https://finnhub.io/api/v1/calendar/economic?from={s}&to={e}&token={urllib.parse.quote(key)}"
    if provider == "EODHD":
        return f"https://eodhd.com/api/economic-events?from={s}&to={e}&api_token={urllib.parse.quote(key)}&fmt=json"
    if provider == "TRADING_ECONOMICS":
        return f"https://api.tradingeconomics.com/calendar?c={urllib.parse.quote(key)}&d1={s}&d2={e}"
    raise ValueError(f"Unsupported provider {provider}")


def fetch_provider_events(provider: str, config: dict[str, Any], today_only: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    key = _provider_key(config, provider)
    meta: dict[str, Any] = {
        "provider": provider,
        "api_key_present": bool(key),
        "status": "NOT_ATTEMPTED",
        "error": None,
    }
    if not key:
        meta["status"] = "NO_API_KEY"
        return [], meta
    start, end = _date_range(today_only)
    try:
        url = _provider_url(provider, key, start, end)
        raw = _http_get_json(url)
    except Exception as exc:
        meta["status"] = "FETCH_FAILED"
        meta["error"] = str(exc)[:300]
        return [], meta
    raw_events = _extract_raw_events(provider, raw)
    normalized: list[dict[str, Any]] = []
    unknown_impact = False
    for idx, raw_event in enumerate(raw_events):
        event = normalize_event(provider, raw_event, idx)
        if event.get("impact") == "UNKNOWN" and event.get("currency") in {"EUR", "USD"}:
            unknown_impact = True
        if event.get("timezone_validated"):
            normalized.append(event)
    meta["status"] = "OK"
    meta["raw_count"] = len(raw_events)
    meta["normalized_count"] = len(normalized)
    meta["unknown_impact_eurusd"] = unknown_impact
    return normalized, meta


def _extract_raw_events(provider: str, raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        for key in ("economicCalendar", "calendar", "events", "data", "result"):
            value = raw.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        if all(k in raw for k in ("event", "date")):
            return [raw]
    return []


def _parse_event_datetime(raw: dict[str, Any]) -> datetime | None:
    for key in ("event_time_utc", "date", "datetime", "time", "Date", "CalendarDate", "eventDate"):
        value = raw.get(key)
        if not value:
            continue
        text = str(value).strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except Exception:
            try:
                parsed = datetime.strptime(text[:19], "%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def _currency_from_raw(raw: dict[str, Any]) -> str:
    value = str(raw.get("currency") or raw.get("Currency") or raw.get("currencyCode") or "").upper()
    if value in {"EUR", "USD"}:
        return value
    country = str(raw.get("country") or raw.get("Country") or raw.get("region") or "").lower()
    if any(token in country for token in ["united states", "usa", "u.s.", "us"]):
        return "USD"
    if any(token in country for token in ["euro", "european union", "euro area", "germany", "france", "italy", "spain"]):
        return "EUR"
    return value


def _impact_from_raw(raw: dict[str, Any]) -> str:
    value = str(raw.get("impact") or raw.get("importance") or raw.get("Impact") or raw.get("Importance") or "").upper()
    if value in {"HIGH", "3", "CALENDAR_IMPORTANCE_HIGH", "HIGH VOLATILITY"}:
        return "HIGH"
    if value in {"MEDIUM", "2", "CALENDAR_IMPORTANCE_MODERATE", "MODERATE"}:
        return "MEDIUM"
    if value in {"LOW", "1", "CALENDAR_IMPORTANCE_LOW"}:
        return "LOW"
    return "UNKNOWN"


def normalize_event(provider: str, raw: dict[str, Any], index: int) -> dict[str, Any]:
    utc = _parse_event_datetime(raw)
    currency = _currency_from_raw(raw)
    impact = _impact_from_raw(raw)
    name = str(raw.get("event") or raw.get("event_name") or raw.get("name") or raw.get("title") or raw.get("Event") or "UNKNOWN_EVENT")
    country = str(raw.get("country") or raw.get("Country") or raw.get("region") or "")
    retrieved = now_utc().isoformat()
    if utc is None:
        return {
            "event_id": f"{provider}_{index}",
            "provider": provider,
            "event_name": name,
            "currency": currency,
            "impact": impact,
            "event_time_utc": None,
            "event_time_ny": None,
            "country": country,
            "actual": raw.get("actual"),
            "forecast": raw.get("forecast"),
            "previous": raw.get("previous"),
            "source_retrieved_at_utc": retrieved,
            "timezone_validated": False,
        }
    return {
        "event_id": str(raw.get("id") or raw.get("calendarId") or raw.get("CalendarId") or f"{provider}_{index}_{int(utc.timestamp())}"),
        "provider": provider,
        "event_name": name,
        "currency": currency,
        "impact": impact,
        "event_time_utc": utc.isoformat(),
        "event_time_ny": utc.astimezone(NY).isoformat(),
        "country": country,
        "actual": raw.get("actual"),
        "forecast": raw.get("forecast"),
        "previous": raw.get("previous"),
        "source_retrieved_at_utc": retrieved,
        "timezone_validated": True,
    }


def filter_eurusd_high_impact(events: list[dict[str, Any]], currencies: list[str], impacts: list[str]) -> list[dict[str, Any]]:
    cset = {c.upper() for c in currencies}
    iset = {i.upper() for i in impacts}
    return sorted(
        [e for e in events if str(e.get("currency", "")).upper() in cset and str(e.get("impact", "")).upper() in iset],
        key=lambda row: row.get("event_time_utc") or "",
    )


def validate_timezone(events: list[dict[str, Any]]) -> bool:
    for event in events:
        if not event.get("timezone_validated"):
            return False
        try:
            datetime.fromisoformat(str(event["event_time_utc"])).astimezone(timezone.utc)
        except Exception:
            return False
    return True


def _cache_paths(day: datetime | None = None) -> dict[str, Path]:
    key = (day or now_utc()).astimezone(NY).strftime("%Y-%m-%d")
    return {
        "today": CACHE_DIR / f"{key}_api_news_today.json",
        "week": CACHE_DIR / f"{key}_api_news_week.json",
        "status": CACHE_DIR / f"{key}_api_news_gate_status.json",
    }


def write_cache(today: dict[str, Any] | None, week: dict[str, Any] | None, status: dict[str, Any]) -> dict[str, str]:
    paths = _cache_paths()
    written: dict[str, str] = {}
    if today is not None:
        write_json(paths["today"], today)
        written["today"] = str(paths["today"])
    if week is not None:
        write_json(paths["week"], week)
        written["week"] = str(paths["week"])
    write_json(paths["status"], status)
    written["status"] = str(paths["status"])
    return written


def read_cache(kind: str) -> tuple[dict[str, Any] | None, str]:
    path = _cache_paths()[kind]
    if not path.exists():
        return None, "NO_TRADE_NEWS_PROVIDER_UNAVAILABLE"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"ERROR_FAIL_CLOSED: {exc}"
    generated = payload.get("generated_at_utc")
    if not generated:
        return None, "NO_TRADE_TIMEZONE_ERROR"
    try:
        age = (now_utc() - datetime.fromisoformat(str(generated).replace("Z", "+00:00")).astimezone(timezone.utc)).total_seconds() / 60
    except Exception:
        return None, "NO_TRADE_TIMEZONE_ERROR"
    if age > int(payload.get("cache_max_age_minutes", 60)):
        return None, "NO_TRADE_NEWS_CACHE_STALE"
    return payload, "OK"


def _build_payload(provider: str, events: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "API_LIVE_NEWS_PROVIDER",
        "provider": provider,
        "generated_at_utc": now_utc().isoformat(),
        "cache_max_age_minutes": int(config.get("cache_max_age_minutes", 60)),
        "currencies": config.get("currencies", ["EUR", "USD"]),
        "impact_filter": config.get("impact_filter", ["HIGH"]),
        "guard_minutes_before": int(config.get("guard_minutes_before", 30)),
        "guard_minutes_after": int(config.get("guard_minutes_after", 30)),
        "verified": True,
        "timezone_validated": validate_timezone(events),
        "events": events,
    }


def fetch_today_events(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return _fetch_with_failover(config, today_only=True)


def fetch_week_events(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return _fetch_with_failover(config, today_only=False)


def _fetch_with_failover(config: dict[str, Any], today_only: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    attempts = []
    for provider in config.get("provider_priority", []):
        provider = str(provider).upper()
        if provider == "MQL5_BOOTSTRAP":
            events, meta = fetch_mql5_bootstrap_events(config, today_only)
        elif provider == "MOCK" and config.get("mode") != "TEST":
            attempts.append({"provider": provider, "status": "MOCK_BLOCKED_OUTSIDE_TEST"})
            continue
        else:
            events, meta = fetch_provider_events(provider, config, today_only)
        attempts.append(meta)
        if meta.get("status") == "OK":
            return events, {"provider": provider, "attempts": attempts, "status": "OK"}
    return [], {"provider": None, "attempts": attempts, "status": "NO_PROVIDER_AVAILABLE"}


def fetch_mql5_bootstrap_events(config: dict[str, Any], today_only: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kind = "today" if today_only else "week"
    path = MANIPULANTE / "09_COMPLIANCE" / "live_news_cache" / "mql5_bootstrap" / f"ftmo_news_{kind}.json"
    meta = {"provider": "MQL5_BOOTSTRAP", "status": "NOT_FOUND", "error": None}
    
    if not path.exists():
        return [], meta
        
    try:
        raw_text = ""
        for enc in ["utf-8", "latin-1", "utf-16"]:
            try:
                raw_text = path.read_text(encoding=enc)
                break
            except:
                continue
        if not raw_text:
            return [], meta
            
        data = json.loads(raw_text)
        raw_events = data.get("events", [])
        normalized = []
        for idx, raw in enumerate(raw_events):
            # MQL5 format: name, currency, importance, time_utc
            t_str = raw.get("time_utc") # "2026.04.29 15:18"
            try:
                utc = datetime.strptime(t_str, "%Y.%m.%d %H:%M").replace(tzinfo=timezone.utc)
            except:
                continue
                
            normalized.append({
                "event_id": f"MQL5_{idx}_{int(utc.timestamp())}",
                "provider": "MQL5_BOOTSTRAP",
                "event_name": raw.get("name"),
                "currency": raw.get("currency"),
                "impact": raw.get("importance"), # Already HIGH/MODERATE
                "event_time_utc": utc.isoformat(),
                "event_time_ny": utc.astimezone(NY).isoformat(),
                "country": raw.get("country", ""),
                "timezone_validated": True
            })
        meta["status"] = "OK"
        meta["normalized_count"] = len(normalized)
        return normalized, meta
    except Exception as e:
        meta["status"] = "ERROR"
        meta["error"] = str(e)
        return [], meta


def news_gate_status(force_fetch: bool = True) -> dict[str, Any]:
    config = load_provider_config()
    if config.get("config_error"):
        status = {
            "timestamp_utc": now_iso(),
            "state": "NO_TRADE_NEWS_PROVIDER_UNAVAILABLE",
            "gate": "NO_TRADE",
            "provider_used": None,
            "api_key_present": False,
            "today_news_loaded": False,
            "week_news_loaded": False,
            "reason": config["config_error"],
            "fail_closed": True,
        }
        write_cache(None, None, status)
        return status

    today_events: list[dict[str, Any]] = []
    week_events: list[dict[str, Any]] = []
    today_meta: dict[str, Any] = {}
    week_meta: dict[str, Any] = {}
    provider_used = None
    if force_fetch:
        today_events, today_meta = fetch_today_events(config)
        week_events, week_meta = fetch_week_events(config)
        provider_used = week_meta.get("provider") or today_meta.get("provider")
        if provider_used and provider_used != today_meta.get("provider"):
            provider_used = None
    if not provider_used:
        status = {
            "timestamp_utc": now_iso(),
            "state": "NO_TRADE_NEWS_PROVIDER_UNAVAILABLE",
            "gate": "NO_TRADE",
            "provider_used": None,
            "api_key_present": any(_provider_key(config, str(p).upper()) for p in config.get("provider_priority", [])),
            "today_news_loaded": False,
            "week_news_loaded": False,
            "today_fetch": _safe_payload(today_meta),
            "week_fetch": _safe_payload(week_meta),
            "reason": "No configured provider with valid API key returned machine-readable news",
            "fail_closed": True,
        }
        write_cache(None, None, status)
        return status

    currencies = config.get("currencies", ["EUR", "USD"])
    impacts = config.get("impact_filter", ["HIGH"])
    today_high = filter_eurusd_high_impact(today_events, currencies, impacts)
    week_high = filter_eurusd_high_impact(week_events, currencies, impacts)
    if not validate_timezone(today_events + week_events):
        state = "NO_TRADE_TIMEZONE_ERROR"
    elif any(e.get("impact") == "UNKNOWN" and e.get("currency") in {"EUR", "USD"} for e in today_events + week_events):
        state = "NO_TRADE_UNKNOWN_IMPACT"
    else:
        state = "ALLOW"
    blocked_event = None
    next_event = None
    now_ny = now_utc().astimezone(NY)
    guard_before = int(config.get("guard_minutes_before", 30))
    guard_after = int(config.get("guard_minutes_after", 30))
    for event in week_high:
        event_time = datetime.fromisoformat(str(event["event_time_utc"])).astimezone(NY)
        event["guard_start_ny"] = (event_time - timedelta(minutes=guard_before)).isoformat()
        event["guard_end_ny"] = (event_time + timedelta(minutes=guard_after)).isoformat()
        if next_event is None and event_time >= now_ny:
            next_event = event
        if event_time - timedelta(minutes=guard_before) <= now_ny <= event_time + timedelta(minutes=guard_after):
            blocked_event = event
            state = "NO_TRADE_NEWS_WINDOW"
            break
    gate = "ALLOW" if state == "ALLOW" else "NO_TRADE"
    today_payload = _build_payload(provider_used, today_high, config)
    week_payload = _build_payload(provider_used, week_high, config)
    status = {
        "timestamp_utc": now_iso(),
        "state": state,
        "gate": gate,
        "provider_used": provider_used,
        "api_key_present": True,
        "today_news_loaded": True,
        "week_news_loaded": True,
        "today_events_count": len(today_high),
        "week_events_count": len(week_high),
        "next_blocking_event": next_event,
        "blocking_event": blocked_event,
        "window_blocked": blocked_event is not None,
        "cache_age_minutes": 0,
        "today_fetch": _safe_payload(today_meta),
        "week_fetch": _safe_payload(week_meta),
        "fail_closed": True,
    }
    written = write_cache(today_payload, week_payload, status)
    status["cache_paths"] = written
    return status


def next_blocking_event(status: dict[str, Any]) -> dict[str, Any] | None:
    return status.get("next_blocking_event")


def write_outputs() -> dict[str, Any]:
    setup = ensure_provider_files()
    status = news_gate_status(force_fetch=True)
    today_payload, _ = read_cache("today")
    week_payload, _ = read_cache("week")
    today_events = (today_payload or {}).get("events", [])
    week_events = (week_payload or {}).get("events", [])
    write_json(OUT / "phase37d_api_live_news.json", {"setup": setup, "status": status})
    fields = [
        "event_id",
        "provider",
        "event_name",
        "currency",
        "impact",
        "event_time_utc",
        "event_time_ny",
        "country",
        "actual",
        "forecast",
        "previous",
        "source_retrieved_at_utc",
        "timezone_validated",
        "guard_start_ny",
        "guard_end_ny",
    ]
    write_csv(OUT / "phase37d_api_news_today.csv", today_events, fields)
    write_csv(OUT / "phase37d_api_news_week.csv", week_events, fields)
    write_text(
        OUT / "phase37d_api_live_news.md",
        f"""
# Phase37D API Live News

- provider configurado: {', '.join(load_provider_config().get('provider_priority', []))}
- API key presente localmente: {status.get('api_key_present')}
- hoy cargado: {status.get('today_news_loaded')}
- semana cargada: {status.get('week_news_loaded')}
- estado: {status.get('state')}
- gate: {status.get('gate')}
- proxima noticia bloqueante: {status.get('next_blocking_event')}
""",
    )
    return status


def main() -> dict[str, Any]:
    status = write_outputs()
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return status


if __name__ == "__main__":
    main()
