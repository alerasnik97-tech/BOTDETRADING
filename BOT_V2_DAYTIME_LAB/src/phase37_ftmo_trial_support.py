from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
import zipfile
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "BOT_V2_DAYTIME_LAB"
MANIPULANTE = ROOT / "MANIPULANTE"
OUT = LAB / "outputs" / "phase37_ftmo_swing_trial_auto"
ZIP_PATH = ROOT / "000_PARA_CHATGPT.zip"
NY = ZoneInfo("America/New_York")
AR = ZoneInfo("America/Argentina/Buenos_Aires")
SECRET_TOKENS = [
    ".env",
    "secret",
    "password",
    "passwd",
    "token",
    "credential",
    "api_key",
    "apikey",
]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_iso() -> str:
    return now_utc().isoformat()


def run_cmd(args: list[str]) -> str:
    try:
        completed = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)
        return (completed.stdout + completed.stderr).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def write_json(path: Path, payload: Any) -> None:
    def default(obj):
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        if hasattr(obj, "item"): # handles numpy types
            return obj.item()
        return str(obj)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, default=default, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zip_test(path: Path) -> str | None:
    with zipfile.ZipFile(path, "r") as zf:
        return zf.testzip()


def root_live_zips() -> list[dict[str, Any]]:
    return [
        {"path": str(path), "bytes": path.stat().st_size}
        for path in ROOT.glob("*.zip")
        if path.is_file()
    ]


def all_zip_inventory() -> list[dict[str, Any]]:
    return [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "live": path.parent == ROOT and path.name == "000_PARA_CHATGPT.zip",
        }
        for path in ROOT.rglob("*.zip")
        if path.is_file()
    ]


def mask_login(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value)
    if len(raw) <= 4:
        return "***"
    return raw[:2] + "***" + raw[-2:]


def load_mt5() -> Any:
    try:
        import MetaTrader5 as mt5  # type: ignore
        return mt5
    except Exception:
        return None


def ensure_mt5() -> tuple[Any, str | None]:
    mt5 = load_mt5()
    if mt5 is None:
        return None, "MetaTrader5 Python module unavailable"
    try:
        if not mt5.initialize():
            return None, f"mt5.initialize failed: {mt5.last_error() if hasattr(mt5, 'last_error') else 'unknown'}"
    except Exception as exc:
        return None, f"mt5.initialize exception: {exc}"
    return mt5, None


def account_gate() -> dict[str, Any]:
    status: dict[str, Any] = {
        "timestamp_utc": now_iso(),
        "state": "BLOCKED_NO_MT5_CONNECTION",
        "terminal_connected": False,
        "ftmo_demo_trial_confirmed": False,
        "company": None,
        "server": None,
        "name": None,
        "login_masked": None,
        "trade_mode": None,
        "trade_mode_label": "UNKNOWN",
        "balance": None,
        "currency": None,
        "trade_allowed": None,
        "terminal_trade_allowed": None,
        "reason": "",
    }
    mt5, error = ensure_mt5()
    if mt5 is None:
        status["reason"] = error
        return status
    status["terminal_connected"] = True
    try:
        info = mt5.account_info()
        terminal = mt5.terminal_info()
    except Exception as exc:
        status["state"] = "BLOCKED_ACCOUNT_INFO_UNAVAILABLE"
        status["reason"] = str(exc)
        return status
    if info is None:
        status["state"] = "BLOCKED_ACCOUNT_INFO_UNAVAILABLE"
        status["reason"] = "account_info unavailable"
        return status
    info_dict = info._asdict() if hasattr(info, "_asdict") else {}
    terminal_dict = terminal._asdict() if terminal is not None and hasattr(terminal, "_asdict") else {}
    company = str(info_dict.get("company", "") or "")
    server = str(info_dict.get("server", "") or "")
    name = str(info_dict.get("name", "") or "")
    trade_mode = info_dict.get("trade_mode")
    status.update(
        {
            "company": company,
            "server": server,
            "name": name,
            "login_masked": mask_login(info_dict.get("login")),
            "trade_mode": trade_mode,
            "balance": float(info_dict.get("balance", 0.0) or 0.0),
            "currency": info_dict.get("currency"),
            "trade_allowed": bool(info_dict.get("trade_allowed")),
            "terminal_trade_allowed": bool(terminal_dict.get("trade_allowed")) if terminal_dict else None,
        }
    )
    mode_map = {
        getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", 0): "DEMO",
        getattr(mt5, "ACCOUNT_TRADE_MODE_CONTEST", 1): "CONTEST",
        getattr(mt5, "ACCOUNT_TRADE_MODE_REAL", 2): "REAL",
    }
    status["trade_mode_label"] = mode_map.get(trade_mode, "UNKNOWN")
    combined = f"{company} {server} {name}".lower()
    if "ftmo" not in combined:
        status["state"] = "BLOCKED_NOT_FTMO"
        status["reason"] = "company/server/name does not identify FTMO"
        return status
    if status["trade_mode_label"] == "REAL":
        status["state"] = "BLOCKED_REAL_ACCOUNT_DETECTED"
        status["reason"] = "MT5 account trade_mode is REAL"
        return status
    if "real" in combined and "free trial" not in combined and "demo" not in combined:
        status["state"] = "BLOCKED_REAL_ACCOUNT_DETECTED"
        status["reason"] = "account text contains real without demo/trial context"
        return status
    if status["currency"] != "USD":
        status["state"] = "BLOCKED_ACCOUNT_INFO_UNAVAILABLE"
        status["reason"] = "account currency is not USD"
        return status
    if abs(float(status["balance"] or 0.0) - 10000.0) > 500.0:
        status["state"] = "WARNING_BALANCE_NOT_10K"
        status["ftmo_demo_trial_confirmed"] = True
        status["reason"] = "FTMO demo/trial detected but balance is not approximately 10k"
        return status
    status["state"] = "FTMO_DEMO_TRIAL_CONFIRMED"
    status["ftmo_demo_trial_confirmed"] = True
    status["reason"] = "FTMO demo/free-trial 10k account confirmed without storing credentials"
    return status


def parse_dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timezone missing")
    return parsed.astimezone(timezone.utc)


def ftmo_news_cache_paths(day: datetime | None = None) -> dict[str, Path]:
    day = day or now_utc()
    key = day.astimezone(NY).strftime("%Y-%m-%d")
    cache = MANIPULANTE / "09_COMPLIANCE" / "live_news_cache"
    return {
        "today": cache / f"{key}_ftmo_news_today.json",
        "week": cache / f"{key}_ftmo_news_week.json",
        "status": cache / f"{key}_ftmo_news_gate_status.json",
    }


def load_news_payload(path: Path, max_age_minutes: int = 60) -> tuple[dict[str, Any] | None, str]:
    if not path.exists():
        return None, "NO_TRADE_NEWS_CACHE_MISSING"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"ERROR_FAIL_CLOSED: malformed cache: {exc}"
    source = str(payload.get("source_type", payload.get("source", ""))).upper()
    if source not in {"MT5_MQL5_ECONOMIC_CALENDAR", "MT5_MQL5_CALENDAR", "MANUAL_EMERGENCY_VERIFIED"}:
        return None, "NO_TRADE_NEWS_CACHE_MISSING"
    if source == "MANUAL_EMERGENCY_VERIFIED" and not payload.get("VERIFIED_BY_USER"):
        return None, "NO_TRADE_NEWS_CACHE_MISSING"
    if not bool(payload.get("verified_by_mt5") or payload.get("VERIFIED_BY_USER") or payload.get("verified")):
        return None, "NO_TRADE_NEWS_CACHE_MISSING"
    raw_generated = payload.get("generated_at_utc") or payload.get("timestamp_utc")
    if not raw_generated:
        return None, "NO_TRADE_TIMEZONE_ERROR"
    try:
        generated = parse_dt(str(raw_generated))
    except Exception:
        return None, "NO_TRADE_TIMEZONE_ERROR"
    age = (now_utc() - generated).total_seconds() / 60.0
    if age > max_age_minutes:
        return None, "NO_TRADE_NEWS_CACHE_STALE"
    return payload, "OK"


def normalize_news_events(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not payload:
        return []
    events = payload.get("events", [])
    if not isinstance(events, list):
        return []
    out: list[dict[str, Any]] = []
    for idx, raw in enumerate(events):
        currency = str(raw.get("currency", "")).upper()
        impact = str(raw.get("impact", raw.get("importance", ""))).upper()
        if impact in {"CALENDAR_IMPORTANCE_HIGH", "3"}:
            impact = "HIGH"
        if currency not in {"EUR", "USD"} or impact != "HIGH":
            continue
        raw_time = raw.get("event_time_utc") or raw.get("time_utc")
        try:
            utc = parse_dt(str(raw_time))
        except Exception:
            continue
        ny_time = utc.astimezone(NY)
        out.append(
            {
                "event_id": str(raw.get("event_id", raw.get("id", f"event_{idx}"))),
                "event_name": str(raw.get("event_name", raw.get("name", "UNKNOWN_EVENT"))),
                "currency": currency,
                "impact": impact,
                "event_time_utc": utc.isoformat(),
                "event_time_ny": ny_time.isoformat(),
                "source": str(payload.get("source_type", payload.get("source", "UNKNOWN"))),
                "verified": bool(payload.get("verified_by_mt5") or payload.get("VERIFIED_BY_USER") or raw.get("verified")),
                "timezone_validated": True,
                "guard_start_ny": (ny_time - timedelta(minutes=30)).isoformat(),
                "guard_end_ny": (ny_time + timedelta(minutes=30)).isoformat(),
            }
        )
    return sorted(out, key=lambda row: row["event_time_utc"])


def live_news_gate() -> dict[str, Any]:
    paths = ftmo_news_cache_paths()
    today_payload, today_status = load_news_payload(paths["today"])
    week_payload, week_status = load_news_payload(paths["week"])
    today_events = normalize_news_events(today_payload)
    week_events = normalize_news_events(week_payload)
    now_ny = now_utc().astimezone(NY)
    blocked_event = None
    next_event = None
    for event in week_events:
        event_time = parse_dt(event["event_time_utc"]).astimezone(NY)
        if next_event is None and event_time >= now_ny:
            next_event = event
        if event_time - timedelta(minutes=30) <= now_ny <= event_time + timedelta(minutes=30):
            blocked_event = event
            break
    if today_status != "OK" or week_status != "OK":
        state = today_status if today_status != "OK" else week_status
    elif blocked_event:
        state = "NO_TRADE_NEWS_WINDOW"
    else:
        state = "ALLOW"
    return {
        "timestamp_utc": now_iso(),
        "today_loaded": today_status == "OK",
        "week_loaded": week_status == "OK",
        "source": "MT5_MQL5_ECONOMIC_CALENDAR_CACHE" if today_status == "OK" and week_status == "OK" else "FTMO_MQL5_CACHE_MISSING",
        "state": state,
        "gate": "ALLOW" if state == "ALLOW" else "NO_TRADE",
        "today_status": today_status,
        "week_status": week_status,
        "today_events_count": len(today_events),
        "week_events_count": len(week_events),
        "next_blocking_event": next_event,
        "blocking_event": blocked_event,
        "cache_paths": {k: str(v) for k, v in paths.items()},
        "fail_closed": True,
    }


def detect_symbol(symbol_candidates: list[str] | None = None) -> dict[str, Any]:
    candidates = symbol_candidates or ["EURUSD", "EURUSD.", "EURUSDm", "EURUSD.r", "EURUSD.raw", "EURUSDpro", "EURUSD_i"]
    mt5, error = ensure_mt5()
    status: dict[str, Any] = {
        "timestamp_utc": now_iso(),
        "state": "ERROR_FAIL_CLOSED",
        "symbol": None,
        "bid": None,
        "ask": None,
        "spread_pips": None,
        "digits": None,
        "point": None,
        "contract_size": None,
        "min_lot": None,
        "max_lot": None,
        "lot_step": None,
        "stops_level": None,
        "freeze_level": None,
        "trade_mode": None,
        "filling_mode": None,
        "order_mode": None,
        "tick_age_seconds": None,
        "server_offset_seconds": None,
        "m3_bars": False,
        "h1_bars": False,
        "reason": "",
    }
    if mt5 is None:
        status["state"] = "NO_TRADE_SYMBOL_NOT_FOUND"
        status["reason"] = error
        return status
    selected = None
    info = None
    for candidate in candidates:
        info = mt5.symbol_info(candidate)
        if info is not None:
            selected = candidate
            break
    if selected is None or info is None:
        try:
            symbols = mt5.symbols_get("*EURUSD*") or []
            for item in symbols:
                info = mt5.symbol_info(item.name)
                if info is not None:
                    selected = item.name
                    break
        except Exception:
            pass
    if selected is None or info is None:
        status["state"] = "NO_TRADE_SYMBOL_NOT_FOUND"
        status["reason"] = "No EURUSD equivalent found"
        return status
    try:
        mt5.symbol_select(selected, True)
    except Exception:
        pass
    tick = mt5.symbol_info_tick(selected)
    if tick is None:
        status["state"] = "NO_TRADE_NO_TICKS"
        status["symbol"] = selected
        status["reason"] = "No current tick"
        return status
    info_dict = info._asdict() if hasattr(info, "_asdict") else {}
    point = float(info_dict.get("point", 0.00001) or 0.00001)
    pip = point * 10 if point < 0.001 else point
    bid = float(getattr(tick, "bid", 0.0) or 0.0)
    ask = float(getattr(tick, "ask", 0.0) or 0.0)
    tick_time = int(getattr(tick, "time", 0) or 0)
    age = now_utc().timestamp() - tick_time if tick_time else 999999
    server_offset = tick_time - now_utc().timestamp() if tick_time else None
    status.update(
        {
            "symbol": selected,
            "bid": bid,
            "ask": ask,
            "spread_pips": round((ask - bid) / pip, 3) if bid > 0 and ask > 0 else None,
            "digits": info_dict.get("digits"),
            "point": point,
            "contract_size": float(info_dict.get("trade_contract_size", 100000.0) or 100000.0),
            "min_lot": float(info_dict.get("volume_min", 0.01) or 0.01),
            "max_lot": float(info_dict.get("volume_max", 100.0) or 100.0),
            "lot_step": float(info_dict.get("volume_step", 0.01) or 0.01),
            "stops_level": int(info_dict.get("trade_stops_level", 0) or 0),
            "freeze_level": int(info_dict.get("trade_freeze_level", 0) or 0),
            "trade_mode": info_dict.get("trade_mode"),
            "filling_mode": info_dict.get("filling_mode"),
            "order_mode": info_dict.get("order_mode"),
            "tick_age_seconds": round(age, 3),
            "server_offset_seconds": round(server_offset, 3) if server_offset is not None else None,
        }
    )
    if bid <= 0 or ask <= 0:
        status["state"] = "NO_TRADE_NO_TICKS"
        status["reason"] = "Missing BID/ASK"
        return status
    offset_hours = round((server_offset or 0.0) / 3600.0)
    offset_residual = abs((server_offset or 0.0) - offset_hours * 3600.0)
    plausible_server_offset = -12 <= offset_hours <= 14 and offset_residual <= 900
    if age > 180 and not plausible_server_offset:
        status["state"] = "NO_TRADE_DATA_STALE"
        status["reason"] = f"Tick stale {age:.1f}s"
        return status
    if status["spread_pips"] is None or status["spread_pips"] > 2.0:
        status["state"] = "NO_TRADE_SPREAD_TOO_HIGH"
        status["reason"] = f"Spread too high or missing: {status['spread_pips']}"
        return status
    try:
        m3 = mt5.copy_rates_from_pos(selected, getattr(mt5, "TIMEFRAME_M3"), 0, 50)
        h1 = mt5.copy_rates_from_pos(selected, getattr(mt5, "TIMEFRAME_H1"), 0, 50)
        status["m3_bars"] = m3 is not None and len(m3) > 0
        status["h1_bars"] = h1 is not None and len(h1) > 0
    except Exception:
        status["m3_bars"] = False
        status["h1_bars"] = False
    if not status["m3_bars"] or not status["h1_bars"]:
        status["state"] = "NO_TRADE_DATA_STALE"
        status["reason"] = "M3/H1 bars unavailable"
        return status
    status["state"] = "ALLOW"
    status["reason"] = "Symbol/data/spread/stoplevel gate passed"
    return status


def time_gate(symbol_status: dict[str, Any] | None = None) -> dict[str, Any]:
    now = now_utc()
    ny = now.astimezone(NY)
    status = {
        "timestamp_utc": now.isoformat(),
        "ny_time": ny.isoformat(),
        "argentina_time": now.astimezone(AR).isoformat(),
        "weekday_ny": ny.strftime("%A"),
        "ny_dst_active": bool(ny.dst()),
        "server_time_validated": False,
        "server_offset_seconds": None,
        "state": "ERROR_FAIL_CLOSED",
        "reason": "",
    }
    mt5, error = ensure_mt5()
    if mt5 is None:
        status["state"] = "NO_TRADE_SERVER_TIME_UNVALIDATED"
        status["reason"] = error
        return status
    symbol = (symbol_status or {}).get("symbol") or "EURUSD"
    try:
        tick = mt5.symbol_info_tick(symbol)
    except Exception:
        tick = None
    if tick is None or not getattr(tick, "time", 0):
        status["state"] = "NO_TRADE_SERVER_TIME_UNVALIDATED"
        status["reason"] = "No tick timestamp available for server time validation"
        return status
    tick_utc = datetime.fromtimestamp(int(tick.time), tz=timezone.utc)
    age = (now - tick_utc).total_seconds()
    offset = (tick_utc - now).total_seconds()
    offset_hours = round(offset / 3600.0)
    offset_residual = abs(offset - offset_hours * 3600.0)
    plausible_server_offset = -12 <= offset_hours <= 14 and offset_residual <= 900
    status["server_offset_seconds"] = round(offset, 3)
    status["server_offset_hours_rounded"] = offset_hours
    if (age < -5 or age > 180) and not plausible_server_offset:
        status["state"] = "NO_TRADE_SERVER_TIME_UNVALIDATED"
        status["reason"] = f"Tick time not current enough: {age:.1f}s"
        return status
    status["server_time_validated"] = True
    if ny.weekday() >= 5:
        status["state"] = "NO_TRADE_WEEKEND"
        status["reason"] = "NY weekend"
        return status
    if ny.weekday() == 4 and ny.time() >= time(16, 55):
        status["state"] = "NO_TRADE_FRIDAY_CUTOFF"
        status["reason"] = "Friday hard close reached"
        return status
    if not (time(7, 0) <= ny.time() <= time(16, 30)):
        status["state"] = "NO_TRADE_OUTSIDE_WINDOW"
        status["reason"] = "Outside 07:00-16:30 NY trading window"
        return status
    status["state"] = "ALLOW"
    status["reason"] = "NY/session/server time gate passed"
    return status


def round_down_lot(raw: float, step: float) -> float:
    if step <= 0:
        return raw
    decimals = max(0, min(8, int(abs(round(math.log10(step)))) if step < 1 else 0))
    return round(math.floor(raw / step) * step, decimals)


def lot_gate_10k(symbol_status: dict[str, Any], account_status: dict[str, Any]) -> dict[str, Any]:
    balance = float(account_status.get("balance") or 10000.0)
    min_lot = float(symbol_status.get("min_lot") or 0.01)
    lot_step = float(symbol_status.get("lot_step") or 0.01)
    spread = float(symbol_status.get("spread_pips") or 0.0)
    pip_value_per_lot = 10.0
    risks = [0.005, 0.0075, 0.01]
    stops = [3, 5, 8, 10, 15, 20]
    rows: list[dict[str, Any]] = []
    allowed_050 = False
    allowed_075 = False
    for risk in risks:
        for stop in stops:
            risk_usd = balance * risk
            raw_lot = risk_usd / (stop * pip_value_per_lot)
            lot = round_down_lot(raw_lot, lot_step)
            reason = "ALLOW"
            allowed = True
            if risk >= 0.01:
                allowed = False
                reason = "RISK_1PCT_PROHIBITED"
                lot = 0.0
            elif stop < 5:
                allowed = False
                reason = "STOP_TOO_SMALL"
                lot = 0.0
            elif spread and spread / stop > 0.12:
                allowed = False
                reason = "SPREAD_SL_RATIO_TOO_HIGH"
                lot = 0.0
            elif lot < min_lot:
                min_risk = min_lot * stop * pip_value_per_lot
                if min_risk > risk_usd:
                    allowed = False
                    reason = "MIN_LOT_EXCEEDS_RISK"
                    lot = 0.0
                else:
                    lot = min_lot
            actual_risk = lot * stop * pip_value_per_lot
            actual_pct = actual_risk / balance if balance else 0.0
            if actual_pct - risk > 1e-9:
                allowed = False
                reason = "ROUNDED_LOT_EXCEEDS_RISK"
            if allowed and risk == 0.005:
                allowed_050 = True
            if allowed and risk == 0.0075:
                allowed_075 = True
            rows.append(
                {
                    "balance": balance,
                    "risk": risk,
                    "stop_pips": stop,
                    "risk_usd": round(risk_usd, 4),
                    "raw_lot": round(raw_lot, 5),
                    "rounded_lot": round(lot, 4),
                    "actual_risk_usd": round(actual_risk, 4),
                    "actual_risk_pct": round(actual_pct, 6),
                    "allowed": allowed,
                    "reason": reason,
                }
            )
    state = "ALLOW" if allowed_050 else "NO_TRADE_MIN_LOT_EXCEEDS_RISK"
    return {
        "timestamp_utc": now_iso(),
        "state": state,
        "balance": balance,
        "symbol": symbol_status.get("symbol"),
        "min_lot": min_lot,
        "lot_step": lot_step,
        "pip_value_per_lot_usd": pip_value_per_lot,
        "risk_050_allowed": allowed_050,
        "risk_075_trial_allowed": allowed_075,
        "risk_100_allowed": False,
        "rows": rows,
        "reason": "0.50% lot scenarios available" if allowed_050 else "No 0.50% lot scenario available",
    }


def order_send_readiness_audit(
    symbol_status: dict[str, Any] | None = None,
    account_status: dict[str, Any] | None = None,
    risk: float = 0.005,
    stop_pips: float = 10.0,
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "timestamp_utc": now_iso(),
        "state": "BLOCKED_MT5_UNAVAILABLE",
        "order_send_executed": False,
        "order_check_executed": False,
        "order_check_pass": False,
        "order_check_retcode": None,
        "order_check_comment": None,
        "margin_required": None,
        "request": None,
        "account_trade_allowed": None,
        "account_trade_expert": None,
        "account_trade_mode": None,
        "account_trade_mode_label": None,
        "terminal_connected": None,
        "terminal_trade_allowed": None,
        "tradeapi_disabled": None,
        "dlls_allowed": None,
        "company": None,
        "server": None,
        "balance": None,
        "currency": None,
        "symbol": None,
        "volume": None,
        "reason": "",
        "orders_message": "ORDENES: BLOQUEADAS",
        "action_required": "Revisar MT5",
        "permission_conclusion": "MT5_NO_AUDITADO",
    }
    mt5, error = ensure_mt5()
    if mt5 is None:
        status["reason"] = error
        return status

    try:
        terminal = mt5.terminal_info()
        account = mt5.account_info()
    except Exception as exc:
        status["state"] = "BLOCKED_ACCOUNT_INFO_UNAVAILABLE"
        status["reason"] = str(exc)
        return status
    if terminal is None or account is None:
        status["state"] = "BLOCKED_ACCOUNT_INFO_UNAVAILABLE"
        status["reason"] = "terminal_info or account_info unavailable"
        return status

    terminal_dict = terminal._asdict() if hasattr(terminal, "_asdict") else {}
    account_dict = account._asdict() if hasattr(account, "_asdict") else {}
    mode_map = {
        getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", 0): "DEMO",
        getattr(mt5, "ACCOUNT_TRADE_MODE_CONTEST", 1): "CONTEST",
        getattr(mt5, "ACCOUNT_TRADE_MODE_REAL", 2): "REAL",
    }
    trade_mode = account_dict.get("trade_mode")
    mode_label = mode_map.get(trade_mode, "UNKNOWN")
    company = str(account_dict.get("company", "") or "")
    server = str(account_dict.get("server", "") or "")
    name = str(account_dict.get("name", "") or "")
    combined = f"{company} {server} {name}".lower()
    terminal_trade_allowed = bool(terminal_dict.get("trade_allowed"))
    tradeapi_disabled = bool(terminal_dict.get("tradeapi_disabled"))
    account_trade_allowed = bool(account_dict.get("trade_allowed"))

    status.update(
        {
            "account_trade_allowed": account_trade_allowed,
            "account_trade_expert": bool(account_dict.get("trade_expert")),
            "account_trade_mode": trade_mode,
            "account_trade_mode_label": mode_label,
            "terminal_connected": bool(terminal_dict.get("connected")),
            "terminal_trade_allowed": terminal_trade_allowed,
            "tradeapi_disabled": tradeapi_disabled,
            "dlls_allowed": bool(terminal_dict.get("dlls_allowed")),
            "company": company,
            "server": server,
            "balance": float(account_dict.get("balance", 0.0) or 0.0),
            "currency": account_dict.get("currency"),
        }
    )

    if "exness" in combined or mode_label == "REAL":
        status["state"] = "EMERGENCY_ABORT_REAL_MONEY_DETECTED"
        status["reason"] = "Real or Exness account detected"
        status["permission_conclusion"] = "EMERGENCY_ABORT_REAL_MONEY_DETECTED"
        return status
    if "ftmo" not in combined or mode_label != "DEMO":
        status["state"] = "BLOCKED_ACCOUNT_NOT_FTMO_DEMO"
        status["reason"] = "Account is not confirmed as FTMO demo"
        status["permission_conclusion"] = "CUENTA_NO_CONFIRMADA"
        return status
    if not account_trade_allowed:
        status["state"] = "BLOCKED_ACCOUNT_TRADE_NOT_ALLOWED"
        status["reason"] = "Account trade_allowed is false"
        status["permission_conclusion"] = "CUENTA_NO_PERMITE_TRADING"
        return status

    symbol = str((symbol_status or {}).get("symbol") or "EURUSD")
    try:
        mt5.symbol_select(symbol, True)
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
    except Exception as exc:
        status["state"] = "BLOCKED_SYMBOL_INFO_UNAVAILABLE"
        status["reason"] = str(exc)
        return status
    if info is None or tick is None:
        status["state"] = "BLOCKED_SYMBOL_INFO_UNAVAILABLE"
        status["reason"] = "symbol_info or symbol_info_tick unavailable"
        return status

    info_dict = info._asdict() if hasattr(info, "_asdict") else {}
    digits = int(info_dict.get("digits", 5) or 5)
    point = float(info_dict.get("point", 0.00001) or 0.00001)
    pip = point * 10 if point < 0.001 else point
    min_lot = float(info_dict.get("volume_min", 0.01) or 0.01)
    max_lot = float(info_dict.get("volume_max", 50.0) or 50.0)
    lot_step = float(info_dict.get("volume_step", 0.01) or 0.01)
    stop_pips = max(float(stop_pips), 5.0)
    risk_usd = float(status["balance"] or 10000.0) * float(risk)
    raw_volume = risk_usd / (stop_pips * 10.0)
    volume = round_down_lot(raw_volume, lot_step)
    volume = max(min_lot, min(max_lot, volume))
    price = float(getattr(tick, "ask", 0.0) or 0.0)
    if price <= 0:
        status["state"] = "BLOCKED_NO_TICK_PRICE"
        status["reason"] = "No ASK price available"
        return status
    sl = round(price - stop_pips * pip, digits)
    tp = round(price + (stop_pips * 1.4) * pip, digits)
    filling = getattr(mt5, "ORDER_FILLING_IOC", 1)
    request = {
        "action": getattr(mt5, "TRADE_ACTION_DEAL", 1),
        "symbol": symbol,
        "volume": float(volume),
        "type": getattr(mt5, "ORDER_TYPE_BUY", 0),
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 370037,
        "comment": "PHASE37ZH_ORDER_CHECK_ONLY",
        "type_time": getattr(mt5, "ORDER_TIME_GTC", 0),
        "type_filling": filling,
    }
    status["symbol"] = symbol
    status["volume"] = float(volume)
    status["request"] = request

    if not hasattr(mt5, "order_check"):
        status["state"] = "BLOCKED_ORDER_CHECK_UNAVAILABLE"
        status["reason"] = "mt5.order_check unavailable"
        return status
    try:
        check = mt5.order_check(request)
        status["order_check_executed"] = True
    except Exception as exc:
        status["state"] = "BLOCKED_ORDER_CHECK_FAILED"
        status["reason"] = f"order_check exception: {exc}"
        return status
    if check is None:
        status["state"] = "BLOCKED_ORDER_CHECK_FAILED"
        status["reason"] = f"order_check returned None: {mt5.last_error() if hasattr(mt5, 'last_error') else 'unknown'}"
        return status
    check_dict = check._asdict() if hasattr(check, "_asdict") else {}
    retcode = check_dict.get("retcode", getattr(check, "retcode", None))
    comment = str(check_dict.get("comment", getattr(check, "comment", "")) or "")
    pass_codes = {
        0,
        getattr(mt5, "TRADE_RETCODE_DONE", 10009),
        getattr(mt5, "TRADE_RETCODE_PLACED", 10008),
        getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010),
    }
    status.update(
        {
            "order_check_retcode": retcode,
            "order_check_comment": comment,
            "margin_required": check_dict.get("margin"),
            "order_check_pass": retcode in pass_codes and "disable" not in comment.lower(),
        }
    )

    if not terminal_trade_allowed or tradeapi_disabled:
        status["state"] = "BLOCKED_AUTOTRADING_DISABLED"
        status["reason"] = "Terminal trade_allowed is false or tradeapi_disabled is true"
        status["orders_message"] = "ORDENES: BLOQUEADAS POR MT5"
        status["permission_conclusion"] = "AUTOTRADING_API_BLOQUEADO_POR_TERMINAL"
        if tradeapi_disabled:
            status["action_required"] = "Opciones MT5: desmarcar bloqueo Python API"
        else:
            status["action_required"] = "Activar Trading algoritmico y reiniciar MT5 si sigue igual"
        return status
    if not status["order_check_pass"]:
        status["state"] = "BLOCKED_ORDER_CHECK_FAILED"
        status["reason"] = f"order_check failed retcode={retcode} comment={comment}"
        status["orders_message"] = "ORDENES: BLOQUEADAS POR ORDER_CHECK"
        status["action_required"] = "Revisar request y permisos MT5"
        status["permission_conclusion"] = "ORDER_CHECK_FALLO"
        return status

    status["state"] = "ORDER_CHECK_PASS_NO_ORDER_SENT"
    status["reason"] = "order_check passed and no order_send was executed"
    status["orders_message"] = "ORDENES: LISTAS EN DEMO"
    status["action_required"] = "Ninguna"
    status["permission_conclusion"] = "ORDER_CHECK_PASS_READY"
    return status


def signal_sync() -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    patterns = ["Phase25", "TP 1.4", "BE 0.4", "BF70", "H1 Fractal", "M3 CHOCH", "manipulante"]
    candidate_dirs = [LAB / "src", MANIPULANTE, ROOT / "ESTRATEGIAS"]
    for base in candidate_dirs:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".py", ".json", ".md", ".txt"}:
                continue
            if any(part in {"__pycache__", "outputs"} for part in path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            score = sum(1 for pattern in patterns if pattern.lower() in text.lower())
            if score:
                findings.append({"path": str(path), "score": score})
    excluded_name_tokens = [
        "phase",
        "orchestrator",
        "audit",
        "report",
        "validation",
        "validator",
        "router",
        "runner",
        "consumer",
        "support",
        "account_gate",
        "time_gate",
        "symbol_data_gate",
        "lot",
    ]
    signal_engine_paths = [
        row for row in findings
        if row["path"].lower().endswith(".py")
        and re.search(r"(signal|choch|sweep|manipulante)", Path(row["path"]).name, re.I)
        and not any(token in Path(row["path"]).name.lower() for token in excluded_name_tokens)
    ]
    live_ready = [
        row for row in signal_engine_paths
        if "phase37" not in row["path"].lower() and "order" not in row["path"].lower()
    ]
    state = "NO_SIGNAL_ENGINE_FOUND"
    matches = False
    if live_ready:
        state = "MANIPULANTE_SIGNAL_REVIEW_REQUIRED"
    return {
        "timestamp_utc": now_iso(),
        "state": state,
        "signal_engine_found": bool(live_ready),
        "manipulante_match": matches,
        "candidate_findings": sorted(findings, key=lambda row: row["score"], reverse=True)[:50],
        "reason": "No live Phase25 signal engine with callable MANIPULANTE signal contract was found",
    }


def order_send_safety() -> dict[str, Any]:
    router = ROOT / "mt5_demo_executor_lab" / "mt5_order_router.py"
    text = router.read_text(encoding="utf-8", errors="ignore") if router.exists() else ""
    direct = re.search(r"(?<!active_)mt5\.order_send\s*\(", text) is not None
    guarded = "safe_order_send_guarded" in text and "active_mt5.order_send(request)" in text
    return {
        "timestamp_utc": now_iso(),
        "router_exists": router.exists(),
        "safe_wrapper_present": "safe_order_send_guarded" in text,
        "direct_mt5_order_send": direct,
        "guarded_active_order_send": guarded,
        "state": "PASS" if router.exists() and not direct and guarded else "BLOCKER",
        "real_blocked_by_default": True,
    }


def confirmation_file_status() -> dict[str, Any]:
    path = MANIPULANTE / "13_FTMO_TRIAL_AUTOMATION" / "I_CONFIRM_FTMO_TRIAL_AUTO.txt"
    required = [
        "I UNDERSTAND THIS IS FTMO FREE TRIAL DEMO ONLY",
        "I CONFIRM NO REAL MONEY",
        "I CONFIRM MANIPULANTE ONLY",
        "RISK_DEFAULT=0.50",
        "ONE_TRADE_PER_DAY",
        "NEWS_GATE_REQUIRED",
        "DATA_GATE_REQUIRED",
    ]
    if not path.exists():
        return {"present": False, "valid": False, "path": str(path), "reason": "CONFIRMATION_FILE_MISSING"}
    content = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    missing = [line for line in required if line not in content]
    return {"present": True, "valid": not missing, "path": str(path), "reason": "OK" if not missing else "MISSING:" + ",".join(missing)}


def strategy_config_gate() -> dict[str, Any]:
    path = MANIPULANTE / "01_ESTRATEGIA_AUTORIDAD" / "manipulante_config.json"
    status = {"path": str(path), "exists": path.exists(), "state": "NO_TRADE", "reason": ""}
    if not path.exists():
        status["reason"] = "manipulante_config.json missing"
        return status
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        status["reason"] = str(exc)
        return status
    tp = payload.get("tp_r")
    be = payload.get("be_r")
    bf = payload.get("body_filter")
    source = str(payload.get("source_phase", ""))
    max_trades = payload.get("max_trades_per_day")
    weekend = payload.get("global_weekend_policy", {})
    if (
        source == "PHASE25_AUTHORITY"
        and abs(float(tp) - 1.4) < 1e-9
        and abs(float(be) - 0.4) < 1e-9
        and abs(float(bf) - 0.7) < 1e-9
        and max_trades == 1
        and weekend.get("hard_close_time_ny") == "16:55"
        and weekend.get("weekend_holding_allowed") is False
    ):
        status["state"] = "MANIPULANTE_MATCH"
        status["reason"] = "Config contains TP 1.4 / BE 0.4 / BF70 evidence"
    else:
        status["reason"] = "Config does not expose required parameter evidence"
    return status


def include_file_for_zip(path: Path) -> bool:
    if not path.is_file():
        return False
    rel = path.relative_to(ROOT)
    rel_str = rel.as_posix()
    parts = set(rel.parts)
    if parts & {".git", ".venv", ".venv_fixed", ".pkg", ".vendor_duka", ".vendor_duka2", "__pycache__", "data", "ARCHIVE_SUPERSEDED"}:
        return False
    lower = rel_str.lower()
    if lower.endswith((".zip", ".zipbak", ".pkl", ".pyc")):
        return False
    if any(token in lower for token in SECRET_TOKENS):
        return False
    if path.stat().st_size > 2 * 1024 * 1024:
        return False
    if rel.parts[0] in {"MANIPULANTE", "ESTRATEGIAS", "mt5_demo_executor_lab"}:
        return True
    if rel.parts[0] == "BOT_V2_DAYTIME_LAB":
        if rel_str.startswith("BOT_V2_DAYTIME_LAB/data/"):
            return False
        if rel_str.startswith("BOT_V2_DAYTIME_LAB/outputs/") and "phase37_ftmo_swing_trial_auto" not in rel_str:
            return False
        if rel_str.startswith("BOT_V2_DAYTIME_LAB/reports/") and "PHASE37" not in rel_str:
            return False
        return True
    return path.name in {
        "00_READ_THIS_FIRST.md",
        "01_CURRENT_PROJECT_STATUS.md",
        "01_CURRENT_PROJECT_STATUS.json",
        "02_STRATEGY_AUTHORITY_MAP.md",
        "02_STRATEGY_AUTHORITY_MAP.json",
        "ZIP_CONTENTS_MANIFEST.md",
    }
