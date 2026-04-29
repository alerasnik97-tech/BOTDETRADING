from __future__ import annotations
import json
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

CONFIG_PATH = Path(r"c:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\MANIPULANTE\13_FTMO_TRIAL_AUTOMATION\runner_lifecycle_config.json")
NY = ZoneInfo("America/New_York")

def load_config():
    if not CONFIG_PATH.exists():
        return {}
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

def get_ny_time() -> datetime:
    return datetime.now(NY)

def get_session_state(position_open: bool = False) -> str:
    now = get_ny_time()
    now_time = now.time()
    weekday = now.weekday()
    
    cfg = load_config()
    
    # Weekend
    if weekday >= 5: # Saturday=5, Sunday=6
        return "WEEKEND_BLOCK"
    
    # Friday Hard Close
    f_close = time.fromisoformat(cfg.get("friday_hard_close", "16:55"))
    if weekday == 4 and now_time >= f_close:
        return "FRIDAY_HARD_CLOSE_REQUIRED"
    
    # Daily States
    start = time.fromisoformat(cfg.get("session_start", "07:00"))
    cutoff = time.fromisoformat(cfg.get("no_new_trades_after", "16:30"))
    force_close = time.fromisoformat(cfg.get("forced_safe_close", "19:45"))
    retry_until = time.fromisoformat(cfg.get("retry_close_until", "19:49"))
    verify1 = time.fromisoformat(cfg.get("verify_flat_first", "19:50"))
    verify2 = time.fromisoformat(cfg.get("verify_flat_second", "19:55"))
    shutdown = time.fromisoformat(cfg.get("daily_shutdown", "20:00"))
    
    if now_time < start:
        return "BEFORE_SESSION_WAIT"
    
    if start <= now_time < cutoff:
        return "SESSION_ACTIVE"
    
    if cutoff <= now_time < force_close:
        return "MANAGE_ONLY" if position_open else "NO_NEW_TRADES_AFTER_CUTOFF"
    
    if force_close <= now_time <= retry_until:
        return "FORCED_SAFE_CLOSE_REQUIRED"
    
    if retry_until < now_time < verify1:
        return "CRITICAL_RETRY_CLOSE" if position_open else "VERIFY_FLAT_BEFORE_SHUTDOWN"
    
    if verify1 <= now_time < verify2:
        return "VERIFY_FLAT_FIRST"
    
    if verify2 <= now_time < shutdown:
        return "VERIFY_FLAT_SECOND"
    
    if now_time >= shutdown:
        return "DAILY_AUTO_SHUTDOWN"
    
    return "UNKNOWN"

def can_open_new_trades() -> bool:
    state = get_session_state()
    return state == "SESSION_ACTIVE"

def should_manage_only() -> bool:
    state = get_session_state(position_open=True)
    return state in {"MANAGE_ONLY", "FORCED_SAFE_CLOSE_REQUIRED", "CRITICAL_RETRY_CLOSE", "VERIFY_FLAT_FIRST", "VERIFY_FLAT_SECOND"}

def should_force_safe_close() -> bool:
    state = get_session_state(position_open=True)
    return state in {"FORCED_SAFE_CLOSE_REQUIRED", "FRIDAY_HARD_CLOSE_REQUIRED", "CRITICAL_RETRY_CLOSE"}

def should_verify_flat() -> bool:
    state = get_session_state()
    return state in {"VERIFY_FLAT_FIRST", "VERIFY_FLAT_SECOND", "VERIFY_FLAT_BEFORE_SHUTDOWN"}

def should_daily_shutdown() -> bool:
    state = get_session_state()
    return state == "DAILY_AUTO_SHUTDOWN"
