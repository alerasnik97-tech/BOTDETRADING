from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from typing import Any

from phase44_observability_db import DAILY_DIR, append_event, connect, init_db


ISO_START_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T")


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def load_rows_for_date(conn, table: str, date: str) -> list[dict[str, Any]]:
    timestamp_col = "timestamp_ny" if table in {"bot_heartbeats", "decisions"} else "timestamp_utc"
    rows = conn.execute(f"SELECT * FROM {table} WHERE {timestamp_col} LIKE ? ORDER BY id", (f"{date}%",)).fetchall()
    data = [dict(row) for row in rows]
    if table == "decisions":
        data = [row for row in data if not ISO_START_RE.match(str(row.get("decision") or ""))]
    return data


def estimate_active_time(heartbeats: list[dict[str, Any]]) -> str:
    active = [row for row in heartbeats if "RUN" in str(row.get("bot_status", "")).upper() or "ACTIVO" in str(row.get("bot_status", "")).upper()]
    stamps = [parse_dt(row.get("timestamp_utc")) for row in active]
    stamps = [stamp for stamp in stamps if stamp is not None]
    if len(stamps) < 2:
        return "UNKNOWN_OR_SINGLE_SNAPSHOT"
    delta = max(stamps) - min(stamps)
    return str(delta)


def decide_verdict(trades: int, warnings: int, critical: int) -> str:
    if critical:
        return "OBS_DAY_CRITICAL"
    if warnings:
        return "OBS_DAY_WARNING"
    if trades:
        return "OBS_DAY_CLEAN_TRADE"
    return "OBS_DAY_CLEAN_NO_TRADE"


def generate_daily_report(date: str | None = None) -> dict[str, Any]:
    init_db()
    with connect() as conn:
        if not date:
            row = conn.execute("SELECT date FROM daily_scorecards ORDER BY id DESC LIMIT 1").fetchone()
            date = row["date"] if row else datetime.now().date().isoformat()
        heartbeats = load_rows_for_date(conn, "bot_heartbeats", date)
        decisions = load_rows_for_date(conn, "decisions", date)
        incidents = conn.execute("SELECT * FROM incidents WHERE timestamp_utc LIKE ? ORDER BY id", (f"{date}%",)).fetchall()
        incidents = [dict(row) for row in incidents]
        scorecard = conn.execute("SELECT * FROM daily_scorecards WHERE date = ? ORDER BY id DESC LIMIT 1", (date,)).fetchone()
        scorecard = dict(scorecard) if scorecard else {}

    trades = sum(1 for row in decisions if str(row.get("order_sent", "")).upper() in {"TRUE", "SI", "YES", "1"})
    no_trade = sum(1 for row in decisions if "NO_TRADE" in str(row.get("decision", "")).upper() or "STOP" in str(row.get("decision", "")).upper())
    news_blocks = sum(1 for row in decisions if "NO_TRADE" in str(row.get("news_gate", "")).upper() or "NEWS" in str(row.get("decision", "")).upper())
    critical = sum(1 for row in incidents if row.get("severity") == "CRITICAL")
    warnings = sum(1 for row in incidents if row.get("severity") == "WARNING")
    open_position = heartbeats[-1].get("open_position", "UNKNOWN") if heartbeats else "UNKNOWN"
    safe_shutdown = heartbeats[-1].get("safe_to_turn_off_pc", "UNKNOWN") if heartbeats else "UNKNOWN"
    verdict = scorecard.get("verdict") or decide_verdict(trades, warnings, critical)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "date": date,
        "verdict": verdict,
        "bot_active_time_estimated": estimate_active_time(heartbeats),
        "heartbeat_count": len(heartbeats),
        "decision_count": len(decisions),
        "trades_taken": trades,
        "no_trade_count": no_trade,
        "news_block_count": news_blocks,
        "incident_count": len(incidents),
        "critical_errors": critical,
        "warnings": warnings,
        "open_position_end": open_position,
        "safe_shutdown": safe_shutdown,
        "latest_decision": decisions[-1].get("decision") if decisions else "UNKNOWN",
        "scorecard": scorecard,
        "next_steps": "Mantener observabilidad read-only. Revisar STATUS antes de apagar si safe_shutdown no es SI.",
        "safety": {
            "mt5_touched": False,
            "orders_sent": False,
            "strategy_modified": False,
        },
    }
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DAILY_DIR / f"{date}_daily_observability_report.json"
    md_path = DAILY_DIR / f"{date}_daily_observability_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    append_event("daily_report", verdict, "phase44_generate_daily_report.py", report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# MANIPULANTE - Daily Observability Report - {report['date']}",
        "",
        f"- generated_at_utc: {report['generated_at_utc']}",
        f"- veredicto: {report['verdict']}",
        "",
        "## Estado del Dia",
        f"- tiempo activo estimado: {report['bot_active_time_estimated']}",
        f"- heartbeats: {report['heartbeat_count']}",
        f"- decisiones: {report['decision_count']}",
        f"- ultima decision: {report['latest_decision']}",
        "",
        "## Operacion",
        f"- trades: {report['trades_taken']}",
        f"- no_trade_count: {report['no_trade_count']}",
        f"- noticias/bloqueos: {report['news_block_count']}",
        f"- operacion abierta al cierre observado: {report['open_position_end']}",
        f"- seguro apagar: {report['safe_shutdown']}",
        "",
        "## Incidentes",
        f"- warnings: {report['warnings']}",
        f"- errores criticos: {report['critical_errors']}",
        f"- incidentes totales: {report['incident_count']}",
        "",
        "## Proximo Paso",
        report["next_steps"],
        "",
        "## Seguridad",
        "- mt5_touched: false",
        "- orders_sent: false",
        "- strategy_modified: false",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Phase44 daily observability report.")
    parser.add_argument("--date", help="YYYY-MM-DD. Defaults to latest scorecard date.")
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()
    report = generate_daily_report(args.date)
    if args.print_json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"Daily report: {report['date']} {report['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
