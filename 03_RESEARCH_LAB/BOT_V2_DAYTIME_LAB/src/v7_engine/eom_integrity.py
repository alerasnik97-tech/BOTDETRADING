from __future__ import annotations

from datetime import datetime
from typing import Any


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _parse_dt(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def actual_window_before_intended(record: dict[str, Any]) -> bool:
    actual = _parse_dt(record.get("actual_tick_window_end") or record.get("tick_window_end"))
    intended = _parse_dt(record.get("intended_position_end"))
    if actual is None or intended is None:
        return True
    if actual.tzinfo is None and intended.tzinfo is not None:
        actual = actual.replace(tzinfo=intended.tzinfo)
    if intended.tzinfo is None and actual.tzinfo is not None:
        intended = intended.replace(tzinfo=actual.tzinfo)
    return actual < intended


def classify_eom(record: dict[str, Any]) -> tuple[bool, str]:
    eom_type = str(record.get("eom_type", "")).strip().upper()
    exit_reason = str(record.get("exit_reason", "")).strip().upper()
    tick_window_complete = _as_bool(record.get("tick_window_complete"))

    if eom_type == "ARTIFICIAL_TRUNCATION":
        return True, "ARTIFICIAL_TRUNCATION"
    if not tick_window_complete:
        return True, "TICK_WINDOW_INCOMPLETE"
    if actual_window_before_intended(record):
        return True, "ACTUAL_BEFORE_INTENDED"
    if exit_reason == "EOM" and eom_type not in {"REAL_DATA_END", "SESSION_FORCED_EXIT", "NO_EOM"}:
        return True, "UNRECOGNIZED_EOM_TYPE"
    return False, ""


def metric_inclusion(record: dict[str, Any]) -> tuple[bool, str]:
    artificial, reason = classify_eom(record)
    if artificial:
        return False, reason
    if not _as_bool(record.get("valid_closed_trade", True)):
        return False, "INVALID_CLOSED_TRADE"
    return True, ""


def compute_net_r_metrics(records: list[dict[str, Any]]) -> dict[str, float | int]:
    values = []
    for record in records:
        included, _ = metric_inclusion(record)
        if included:
            values.append(float(record.get("net_r", 0.0)))

    wins = [value for value in values if value > 0]
    losses = [value for value in values if value <= 0]
    loss_sum = sum(losses)
    if not values:
        pf = 0.0
    elif loss_sum < 0:
        pf = sum(wins) / abs(loss_sum)
    else:
        pf = 999.0 if wins else 0.0

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)

    n = len(values)
    return {
        "N": n,
        "PF_net": round(pf, 4),
        "total_net_r": round(sum(values), 4),
        "expectancy": round(sum(values) / n, 4) if n else 0.0,
        "winrate": round(len(wins) / n, 4) if n else 0.0,
        "max_dd": round(max_dd, 4),
    }
