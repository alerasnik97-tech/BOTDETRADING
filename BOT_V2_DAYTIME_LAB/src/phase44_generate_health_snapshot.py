from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from phase44_observability_db import (
    DAILY_DIR,
    MANIPULANTE_DIR,
    append_event,
    connect,
    init_db,
    latest_row,
    recent_rows,
)


PROMOTION_GATE = MANIPULANTE_DIR / "15_FORWARD_DEMO_SCORECARD" / "promotion_gate" / "MANIPULANTE_PROMOTION_GATE_TO_REAL.md"
FORWARD_DASHBOARD = MANIPULANTE_DIR / "15_FORWARD_DEMO_SCORECARD" / "FORWARD_DEMO_DASHBOARD.md"
ISO_START_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T")


def read_promotion_gate_status() -> dict[str, Any]:
    data = {"path": str(PROMOTION_GATE), "exists": PROMOTION_GATE.exists(), "summary": "UNKNOWN"}
    if not PROMOTION_GATE.exists():
        return data
    text = PROMOTION_GATE.read_text(encoding="utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    interesting = [line for line in lines if any(token in line.upper() for token in ("PROMOTION", "REAL", "PASS", "FAIL", "NO ", "GATE"))]
    data["summary"] = " | ".join(interesting[:6]) if interesting else lines[0][:200]
    return data


def read_forward_dashboard_summary() -> dict[str, Any]:
    data = {"path": str(FORWARD_DASHBOARD), "exists": FORWARD_DASHBOARD.exists(), "stress_status": "UNKNOWN", "promotion_progress": "UNKNOWN"}
    if not FORWARD_DASHBOARD.exists():
        return data
    text = FORWARD_DASHBOARD.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        clean = line.strip()
        upper = clean.upper()
        if "STRESS" in upper or "16/16" in upper:
            data["stress_status"] = clean
        if "HACIA 20" in upper or "HACIA 30" in upper:
            prev = data["promotion_progress"]
            data["promotion_progress"] = clean if prev == "UNKNOWN" else f"{prev} | {clean}"
    return data


def count_recent_incidents(conn, severity: str | None = None) -> int:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    if severity:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM incidents WHERE timestamp_utc >= ? AND severity = ?",
            (cutoff, severity),
        ).fetchone()
    else:
        row = conn.execute("SELECT COUNT(*) AS n FROM incidents WHERE timestamp_utc >= ?", (cutoff,)).fetchone()
    return int(row["n"])


def latest_valid_decision(conn) -> dict[str, Any] | None:
    rows = conn.execute("SELECT * FROM decisions ORDER BY id DESC LIMIT 100").fetchall()
    for row in rows:
        data = dict(row)
        if not ISO_START_RE.match(str(data.get("decision") or "")):
            return data
    return None


def assess_health(latest_hb: dict[str, Any] | None, critical: int, warnings: int) -> tuple[str, str]:
    if not latest_hb:
        return "HEALTH_STOPPED", "No hay heartbeat disponible. Revise START/STATUS antes de operar."
    bot_status = str(latest_hb.get("bot_status") or "").upper()
    news_status = str(latest_hb.get("news_status") or "").upper()
    orders_status = str(latest_hb.get("orders_status") or "").upper()
    open_position = str(latest_hb.get("open_position") or "").upper()
    if critical > 0:
        return "HEALTH_CRITICAL", "Hay incidentes criticos recientes. No cierre PC sin revisar MT5/STATUS."
    if open_position in {"SI", "YES", "TRUE"}:
        return "HEALTH_CRITICAL", "Operacion abierta detectada. No apagar PC."
    if "STOP" in bot_status or bot_status in {"APAGADO", "STOPPED"}:
        return "HEALTH_STOPPED", "Bot detenido. Use START solo si no hay riesgo abierto."
    if "NO_TRADE" in news_status or "BLOCK" in news_status or "GATEADO" in orders_status:
        return "HEALTH_BLOCKED_BY_RULE", "Bot observado pero bloqueado por regla. No requiere accion si es esperado."
    if warnings > 0:
        return "HEALTH_WARNING", "Hay warnings recientes. Revisar incidentes antes de cerrar el dia."
    return "HEALTH_OK", "Estado operativo normal de observabilidad."


def generate_snapshot() -> dict[str, Any]:
    init_db()
    with connect() as conn:
        latest_hb = latest_row(conn, "bot_heartbeats")
        latest_decision = latest_valid_decision(conn)
        latest_scorecard = latest_row(conn, "daily_scorecards")
        recent_incidents = recent_rows(conn, "incidents", 20)
        critical_24h = count_recent_incidents(conn, "CRITICAL")
        warning_24h = count_recent_incidents(conn, "WARNING")

    status, recommendation = assess_health(latest_hb, critical_24h, warning_24h)
    promotion = read_promotion_gate_status()
    dashboard = read_forward_dashboard_summary()
    snapshot = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "health_status": status,
        "bot_active": bool(latest_hb and "RUN" in str(latest_hb.get("bot_status", "")).upper()),
        "account": latest_hb.get("account") if latest_hb else "UNKNOWN",
        "mode": latest_hb.get("mode") if latest_hb else "UNKNOWN",
        "runner_pid": latest_hb.get("runner_pid") if latest_hb else "UNKNOWN",
        "orders_status": latest_hb.get("orders_status") if latest_hb else "UNKNOWN",
        "news_status": latest_hb.get("news_status") if latest_hb else "UNKNOWN",
        "last_decision": (latest_decision or {}).get("decision") or (latest_hb or {}).get("last_decision") or "UNKNOWN",
        "open_position": latest_hb.get("open_position") if latest_hb else "UNKNOWN",
        "safe_to_turn_off_pc": latest_hb.get("safe_to_turn_off_pc") if latest_hb else "UNKNOWN",
        "critical_errors_24h": critical_24h,
        "warnings_24h": warning_24h,
        "daily_scorecard": latest_scorecard,
        "promotion_gate": promotion,
        "stress_status": dashboard.get("stress_status"),
        "promotion_progress": dashboard.get("promotion_progress"),
        "recent_incidents": recent_incidents,
        "recommendation": recommendation,
        "safety": {
            "read_only": True,
            "mt5_touched": False,
            "orders_sent": False,
            "strategy_modified": False,
        },
    }
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DAILY_DIR / "latest_health_snapshot.json"
    md_path = DAILY_DIR / "latest_health_snapshot.md"
    json_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(snapshot), encoding="utf-8")
    append_event("health_snapshot", status, "phase44_generate_health_snapshot.py", snapshot)
    return snapshot


def render_markdown(snapshot: dict[str, Any]) -> str:
    scorecard = snapshot.get("daily_scorecard") or {}
    lines = [
        "# MANIPULANTE - Latest Health Snapshot",
        "",
        f"- generated_at_utc: {snapshot['generated_at_utc']}",
        f"- estado: {snapshot['health_status']}",
        f"- recomendacion: {snapshot['recommendation']}",
        "",
        "## Estado Actual",
        f"- bot activo: {snapshot['bot_active']}",
        f"- cuenta: {snapshot['account']}",
        f"- modo: {snapshot['mode']}",
        f"- runner PID: {snapshot['runner_pid'] or 'UNKNOWN'}",
        f"- ordenes: {snapshot['orders_status']}",
        f"- noticias: {snapshot['news_status']}",
        f"- ultima decision: {snapshot['last_decision']}",
        f"- operacion abierta: {snapshot['open_position']}",
        f"- seguro apagar PC: {snapshot['safe_to_turn_off_pc']}",
        "",
        "## Ultimas 24h",
        f"- errores criticos: {snapshot['critical_errors_24h']}",
        f"- warnings: {snapshot['warnings_24h']}",
        "",
        "## Scorecard del Dia",
        f"- fecha: {scorecard.get('date', 'UNKNOWN')}",
        f"- veredicto: {scorecard.get('verdict', 'UNKNOWN')}",
        f"- trades: {scorecard.get('trades_taken', 'UNKNOWN')}",
        f"- no_trade_count: {scorecard.get('no_trade_count', 'UNKNOWN')}",
        "",
        "## Promotion Gate",
        f"- estado: {snapshot['promotion_gate'].get('summary', 'UNKNOWN')}",
        f"- progreso: {snapshot.get('promotion_progress')}",
        "",
        "## Stress Tests",
        f"- status: {snapshot.get('stress_status')}",
        "",
        "## Seguridad",
        "- read_only: true",
        "- mt5_touched: false",
        "- orders_sent: false",
        "- strategy_modified: false",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate MANIPULANTE Phase44 read-only health snapshot.")
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()
    snapshot = generate_snapshot()
    if args.print_json:
        print(json.dumps(snapshot, indent=2, ensure_ascii=False))
    else:
        print(f"Health snapshot: {snapshot['health_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
