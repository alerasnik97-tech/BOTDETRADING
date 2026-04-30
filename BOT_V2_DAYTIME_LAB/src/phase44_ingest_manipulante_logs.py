from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase44_observability_db import (
    JSONL_PATH,
    MANIPULANTE_DIR,
    append_event,
    connect,
    init_db,
    insert_daily_scorecard,
    insert_decision,
    insert_heartbeat,
    insert_incident,
    mask_sensitive,
    parse_key_value_text,
    safe_json,
    timestamp_bundle,
)


LOG_ROOT = MANIPULANTE_DIR / "10_LOGS_PAPER" / "ftmo_trial_bot"
SCORECARD_ROOT = MANIPULANTE_DIR / "15_FORWARD_DEMO_SCORECARD"
QUICK_STATUS = LOG_ROOT / "quick_status.txt"
HEARTBEAT_JSON = LOG_ROOT / "heartbeat.json"
DECISIONS_CSV = LOG_ROOT / "decisions.csv"
FORWARD_DASHBOARD = SCORECARD_ROOT / "FORWARD_DEMO_DASHBOARD.md"
STRESS_ROOT = SCORECARD_ROOT / "stress_tests"
ISO_START_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return mask_sensitive(json.loads(path.read_text(encoding="utf-8", errors="replace")))


def bool_text(value: Any) -> str:
    if isinstance(value, bool):
        return "SI" if value else "NO"
    if value is None:
        return "UNKNOWN"
    text = str(value).strip()
    if text.upper() in {"TRUE", "SI", "YES", "1"}:
        return "SI"
    if text.upper() in {"FALSE", "NO", "0"}:
        return "NO"
    return text


def heartbeat_row_from_json(data: dict[str, Any]) -> dict[str, Any]:
    ts_utc, ts_arg, ts_ny = timestamp_bundle(data.get("timestamp_local") or data.get("timestamp_ny"))
    position_state = str(data.get("position_state") or "UNKNOWN").upper()
    mt5_status = "CONNECTED" if data.get("terminal_connected") is True else "UNKNOWN"
    if data.get("terminal_connected") is False:
        mt5_status = "DISCONNECTED"
    return {
        "timestamp_utc": ts_utc,
        "timestamp_arg": ts_arg,
        "timestamp_ny": ts_ny,
        "bot_status": data.get("runner_status") or data.get("session_state"),
        "account": data.get("server") or data.get("account_company"),
        "mode": data.get("account_mode"),
        "runner_pid": str(data.get("pid") or ""),
        "mt5_status": mt5_status,
        "news_status": data.get("news_gate"),
        "orders_status": data.get("order_readiness_state") or data.get("orders_message"),
        "open_position": "SI" if position_state not in {"FLAT", "NO", "NONE", "NULL"} else "NO",
        "safe_to_turn_off_pc": bool_text(data.get("safe_to_turn_off_pc")),
        "last_decision": data.get("last_decision"),
        "raw_source": safe_json({"source": str(HEARTBEAT_JSON.relative_to(MANIPULANTE_DIR.parent)), "data": data}),
    }


def heartbeat_row_from_quick_status(data: dict[str, str]) -> dict[str, Any]:
    ts_utc, ts_arg, ts_ny = timestamp_bundle(datetime.now(timezone.utc).isoformat())
    terminal_connected = data.get("TERMINAL_CONNECTED", "")
    mt5_status = "CONNECTED" if terminal_connected.upper() == "SI" else "UNKNOWN"
    if terminal_connected.upper() == "NO":
        mt5_status = "DISCONNECTED"
    runner_text = str(data.get("RUNNER", "")).upper()
    bot_status = "RUNNING" if runner_text in {"ACTIVO", "RUNNING"} else (data.get("ESTADO_GENERAL") or data.get("RUNNER"))
    return {
        "timestamp_utc": ts_utc,
        "timestamp_arg": ts_arg,
        "timestamp_ny": ts_ny,
        "bot_status": bot_status,
        "account": data.get("CUENTA"),
        "mode": "DEMO" if "DEMO" in str(data.get("CUENTA", "")).upper() else "",
        "runner_pid": "",
        "mt5_status": mt5_status,
        "news_status": data.get("NEWS"),
        "orders_status": data.get("ORDENES") or data.get("ORDER_SEND"),
        "open_position": data.get("OPERACION_ABIERTA"),
        "safe_to_turn_off_pc": data.get("SEGURO_APAGAR_PC"),
        "last_decision": data.get("ULTIMA_DECISION"),
        "raw_source": safe_json({"source": str(QUICK_STATUS.relative_to(MANIPULANTE_DIR.parent)), "data": data}),
    }


def parse_gates(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return mask_sensitive(parsed if isinstance(parsed, dict) else {})
    except json.JSONDecodeError:
        return {}


def looks_like_timestamp(value: Any) -> bool:
    return bool(ISO_START_RE.match(str(value or "")))


def normalize_decision_rows(source_file: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not source_file.exists():
        return rows
    with source_file.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        raw_rows = list(reader)
    if not raw_rows:
        return rows
    header = raw_rows[0]
    for raw in raw_rows[1:]:
        if not raw:
            continue
        if len(raw) >= 13 and looks_like_timestamp(raw[0]) and looks_like_timestamp(raw[1]):
            rows.append(
                {
                    "timestamp": raw[1],
                    "timestamp_arg": raw[0],
                    "timestamp_ny": raw[1],
                    "final_decision": raw[2],
                    "reason": raw[3],
                    "signal_status": raw[4],
                    "order_sent": raw[5],
                    "account": raw[6],
                    "news_gate": raw[7],
                    "data_gate": raw[8],
                    "time_gate": raw[9],
                    "session_state": raw[10],
                    "position_state": raw[11],
                    "gates_status": raw[12],
                }
            )
            continue
        mapped = {header[i]: raw[i] for i in range(min(len(header), len(raw)))}
        if len(raw) > len(header):
            mapped["extra_fields"] = json.dumps(raw[len(header) :], ensure_ascii=False)
        rows.append(mapped)
    return rows


def decision_row(row: dict[str, str], source_file: Path) -> dict[str, Any]:
    gates = parse_gates(row.get("gates_status", ""))
    ts_utc, ts_arg, ts_ny = timestamp_bundle(row.get("timestamp"))
    return {
        "timestamp_utc": ts_utc,
        "timestamp_arg": ts_arg,
        "timestamp_ny": ts_ny,
        "decision": row.get("final_decision"),
        "reason": row.get("reason"),
        "session_state": row.get("session_state") or gates.get("lifecycle_state") or gates.get("session_state") or row.get("time_gate") or gates.get("time_gate"),
        "news_gate": row.get("news_gate") or gates.get("api_live_news_gate") or gates.get("live_news_gate") or gates.get("news_gate"),
        "data_gate": row.get("data_gate") or gates.get("data_quality_gate"),
        "signal_gate": row.get("signal_status") or gates.get("signal_engine_gate"),
        "order_sent": bool_text(row.get("order_sent")),
        "source_file": str(source_file.relative_to(MANIPULANTE_DIR.parent)),
    }


def should_create_incident(text: str, order_sent: str) -> tuple[bool, str, str]:
    up = text.upper()
    if order_sent.upper() in {"TRUE", "SI", "YES", "1"}:
        return True, "CRITICAL", "order_sent_detected"
    if any(token in up for token in ("CRITICAL", "ERROR", "FAIL")):
        return True, "CRITICAL" if "CRITICAL" in up else "WARNING", "log_error_keyword"
    return False, "", ""


def ingest_heartbeats(conn) -> int:
    count = 0
    if HEARTBEAT_JSON.exists():
        data = read_json(HEARTBEAT_JSON)
        row = heartbeat_row_from_json(data)
        insert_heartbeat(conn, row)
        append_event("heartbeat", str(row.get("bot_status") or "UNKNOWN"), str(HEARTBEAT_JSON), row)
        count += 1
    if QUICK_STATUS.exists():
        data = parse_key_value_text(QUICK_STATUS)
        row = heartbeat_row_from_quick_status(data)
        insert_heartbeat(conn, row)
        append_event("quick_status", str(row.get("bot_status") or "UNKNOWN"), str(QUICK_STATUS), row)
        count += 1
    return count


def ingest_decisions(conn) -> int:
    if not DECISIONS_CSV.exists():
        return 0
    count = 0
    for src_row in normalize_decision_rows(DECISIONS_CSV):
        row = decision_row(src_row, DECISIONS_CSV)
        insert_decision(conn, row)
        append_event("decision", str(row.get("decision") or "UNKNOWN"), str(DECISIONS_CSV), row)
        merged = " ".join(str(src_row.get(k, "")) for k in ("final_decision", "reason", "signal_status", "gates_status"))
        create, severity, category = should_create_incident(merged, str(src_row.get("order_sent", "")))
        if create:
            insert_incident(
                conn,
                {
                    "timestamp_utc": row["timestamp_utc"],
                    "severity": severity,
                    "category": category,
                    "description": f"{row.get('decision')}: {row.get('reason')}",
                    "resolved": "NO",
                    "source": str(DECISIONS_CSV.relative_to(MANIPULANTE_DIR.parent)),
                },
            )
        count += 1
    return count


def collect_dashboard_notes() -> str:
    notes: list[str] = []
    if FORWARD_DASHBOARD.exists():
        text = FORWARD_DASHBOARD.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            clean = line.strip(" -")
            if "Resultado" in clean or "Veredicto" in clean or "Trades Tomados" in clean or "Errores Criticos" in clean:
                notes.append(clean)
    if STRESS_ROOT.exists():
        for path in sorted(STRESS_ROOT.glob("*.csv")):
            notes.append(f"stress_csv={path.name}")
    return "; ".join(notes)


def ingest_daily_scorecard(conn) -> dict[str, Any]:
    rows = conn.execute("SELECT * FROM decisions ORDER BY id DESC").fetchall()
    latest_hb = conn.execute("SELECT * FROM bot_heartbeats ORDER BY id DESC LIMIT 1").fetchone()
    if latest_hb and latest_hb["timestamp_ny"]:
        date = str(latest_hb["timestamp_ny"])[:10]
    else:
        date = datetime.now().date().isoformat()

    day_rows = [
        dict(row)
        for row in rows
        if str(row["timestamp_ny"]).startswith(date) and not looks_like_timestamp(row["decision"])
    ]
    trades_taken = sum(1 for row in day_rows if str(row.get("order_sent", "")).upper() in {"TRUE", "SI", "YES", "1"})
    no_trade_count = sum(1 for row in day_rows if "NO_TRADE" in str(row.get("decision", "")).upper() or "STOP" in str(row.get("decision", "")).upper())
    critical_errors = conn.execute(
        "SELECT COUNT(*) AS n FROM incidents WHERE timestamp_utc >= ? AND severity = 'CRITICAL'",
        (f"{date}T00:00:00",),
    ).fetchone()["n"]
    warnings = conn.execute(
        "SELECT COUNT(*) AS n FROM incidents WHERE timestamp_utc >= ? AND severity = 'WARNING'",
        (f"{date}T00:00:00",),
    ).fetchone()["n"]
    open_position_end = latest_hb["open_position"] if latest_hb else "UNKNOWN"
    safe_shutdown = latest_hb["safe_to_turn_off_pc"] if latest_hb else "UNKNOWN"
    if critical_errors:
        verdict = "OBS_DAY_CRITICAL"
    elif warnings:
        verdict = "OBS_DAY_WARNING"
    elif trades_taken:
        verdict = "OBS_DAY_CLEAN_TRADE"
    else:
        verdict = "OBS_DAY_CLEAN_NO_TRADE"
    row = {
        "date": date,
        "verdict": verdict,
        "trades_taken": trades_taken,
        "no_trade_count": no_trade_count,
        "warnings": int(warnings),
        "critical_errors": int(critical_errors),
        "open_position_end": open_position_end,
        "safe_shutdown": safe_shutdown,
        "notes": collect_dashboard_notes(),
    }
    insert_daily_scorecard(conn, row)
    append_event("daily_scorecard", verdict, "phase44_ingest_manipulante_logs.py", row)
    return row


def ingest() -> dict[str, Any]:
    init_db()
    with connect() as conn:
        heartbeats = ingest_heartbeats(conn)
        decisions = ingest_decisions(conn)
        daily = ingest_daily_scorecard(conn)
        conn.commit()
        counts = {
            "bot_heartbeats": conn.execute("SELECT COUNT(*) AS n FROM bot_heartbeats").fetchone()["n"],
            "decisions": conn.execute("SELECT COUNT(*) AS n FROM decisions").fetchone()["n"],
            "daily_scorecards": conn.execute("SELECT COUNT(*) AS n FROM daily_scorecards").fetchone()["n"],
            "incidents": conn.execute("SELECT COUNT(*) AS n FROM incidents").fetchone()["n"],
        }
    result = {
        "heartbeats_ingested_this_run": heartbeats,
        "decisions_seen_this_run": decisions,
        "daily_scorecard": daily,
        "db_counts": counts,
        "jsonl_path": str(JSONL_PATH),
        "read_only_sources": [
            str(QUICK_STATUS),
            str(HEARTBEAT_JSON),
            str(DECISIONS_CSV),
            str(FORWARD_DASHBOARD),
            str(STRESS_ROOT),
        ],
        "safety": {
            "mt5_touched": False,
            "orders_sent": False,
            "strategy_modified": False,
        },
    }
    append_event("ingest_complete", "OK", "phase44_ingest_manipulante_logs.py", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Read MANIPULANTE paper/demo logs into local Phase44 SQLite observability DB.")
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()
    result = ingest()
    if args.print_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Ingest complete. DB counts: {result['db_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
