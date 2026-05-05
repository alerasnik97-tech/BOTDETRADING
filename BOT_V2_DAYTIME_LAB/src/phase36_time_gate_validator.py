from __future__ import annotations

import json
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "BOT_V2_DAYTIME_LAB" / "outputs" / "phase36r_37a_micro_real_gate" / "time_gate"
NY = ZoneInfo("America/New_York")
AR = ZoneInfo("America/Argentina/Buenos_Aires")


def get_time_gate_status(now_utc: datetime | None = None, require_server_time_for_real: bool = True) -> dict[str, Any]:
    now_utc = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    now_ny = now_utc.astimezone(NY)
    now_ar = now_utc.astimezone(AR)
    status = {
        "timestamp_utc": now_utc.isoformat(),
        "ny_time": now_ny.isoformat(),
        "argentina_time": now_ar.isoformat(),
        "weekday": now_ny.strftime("%A"),
        "ny_dst_active": bool(now_ny.dst()),
        "server_time_validated": False,
        "server_time": None,
        "state": "ALLOW",
        "real_state": "NO_TRADE",
        "reason": "inside MANIPULANTE time window",
    }
    if now_ny.weekday() >= 5:
        status["state"] = "NO_TRADE_WEEKEND"
        status["real_state"] = "NO_TRADE"
        status["reason"] = "weekend"
        return status
    if now_ny.weekday() == 4 and now_ny.time() >= time(16, 55):
        status["state"] = "NO_TRADE_FRIDAY_CUTOFF"
        status["real_state"] = "NO_TRADE"
        status["reason"] = "Friday hard close reached"
        return status
    if not (time(7, 0) <= now_ny.time() <= time(16, 30)):
        status["state"] = "NO_TRADE_OUTSIDE_WINDOW"
        status["real_state"] = "NO_TRADE"
        status["reason"] = "outside 07:00-16:30 NY"
        return status
    if require_server_time_for_real:
        status["state"] = "WARNING_SERVER_TIME_REQUIRES_MANUAL_CONFIRMATION"
        status["real_state"] = "NO_TRADE"
        status["reason"] = "NY time is inside window but MT5 server time is not validated"
        return status
    status["real_state"] = "ALLOW"
    return status


def write_outputs() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    status = get_time_gate_status()
    (OUT / "phase36r_time_gate.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    md = [
        "# Phase36R Time Gate",
        "",
        f"- state: {status['state']}",
        f"- real_state: {status['real_state']}",
        f"- NY time: {status['ny_time']}",
        f"- Argentina time: {status['argentina_time']}",
        f"- server_time_validated: {status['server_time_validated']}",
        f"- reason: {status['reason']}",
    ]
    (OUT / "phase36r_time_gate.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return status


if __name__ == "__main__":
    print(json.dumps(write_outputs(), indent=2))
