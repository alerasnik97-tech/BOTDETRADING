from __future__ import annotations

import argparse
import html
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase44_observability_db import DASHBOARD_DIR, DB_PATH, DAILY_DIR, connect, init_db, recent_rows


HTML_PATH = DASHBOARD_DIR / "dashboard.html"
ISO_START_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T")


def load_dashboard_data() -> dict[str, Any]:
    init_db()
    with connect() as conn:
        latest_hb = conn.execute("SELECT * FROM bot_heartbeats ORDER BY id DESC LIMIT 1").fetchone()
        latest_decision = conn.execute("SELECT * FROM decisions ORDER BY id DESC LIMIT 1").fetchone()
        latest_scorecard = conn.execute("SELECT * FROM daily_scorecards ORDER BY id DESC LIMIT 1").fetchone()
        incidents = recent_rows(conn, "incidents", 20)
        decisions = [
            row
            for row in recent_rows(conn, "decisions", 80)
            if not ISO_START_RE.match(str(row.get("decision") or ""))
        ][:20]
        fills = recent_rows(conn, "fills_manual", 20)
        counts = {
            "heartbeats": conn.execute("SELECT COUNT(*) AS n FROM bot_heartbeats").fetchone()["n"],
            "decisions": conn.execute("SELECT COUNT(*) AS n FROM decisions").fetchone()["n"],
            "daily_scorecards": conn.execute("SELECT COUNT(*) AS n FROM daily_scorecards").fetchone()["n"],
            "incidents": conn.execute("SELECT COUNT(*) AS n FROM incidents").fetchone()["n"],
            "fills_manual": conn.execute("SELECT COUNT(*) AS n FROM fills_manual").fetchone()["n"],
        }
    latest_health = {}
    health_path = DAILY_DIR / "latest_health_snapshot.json"
    if health_path.exists():
        latest_health = json.loads(health_path.read_text(encoding="utf-8", errors="replace"))
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "db_path": str(DB_PATH),
        "latest_health": latest_health,
        "latest_heartbeat": dict(latest_hb) if latest_hb else {},
        "latest_decision": dict(latest_decision) if latest_decision else {},
        "latest_scorecard": dict(latest_scorecard) if latest_scorecard else {},
        "decisions": decisions,
        "incidents": incidents,
        "fills_manual": fills,
        "counts": counts,
    }


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "<p class='muted'>Sin datos.</p>"
    header = "".join(f"<th>{esc(col)}</th>" for col in columns)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{esc(row.get(col, ''))}</td>" for col in columns) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def metric(label: str, value: Any) -> str:
    return f"<div class='metric'><span>{esc(label)}</span><strong>{esc(value)}</strong></div>"


def build_static_html() -> Path:
    data = load_dashboard_data()
    hb = data["latest_heartbeat"]
    health = data["latest_health"]
    scorecard = data["latest_scorecard"]
    fills = data["fills_manual"]
    net_r = sum(float(row.get("net_r") or 0) for row in fills if row.get("net_r") not in (None, ""))
    commissions = sum(float(row.get("commission") or 0) for row in fills if row.get("commission") not in (None, ""))
    content = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MANIPULANTE Observability</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 0; background: #f6f7f9; color: #17202a; }}
    header {{ background: #17202a; color: white; padding: 18px 28px; }}
    main {{ padding: 22px 28px; max-width: 1200px; margin: 0 auto; }}
    section {{ background: white; border: 1px solid #d7dce2; border-radius: 6px; padding: 16px; margin-bottom: 16px; }}
    h1 {{ font-size: 24px; margin: 0; }}
    h2 {{ font-size: 18px; margin: 0 0 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; }}
    .metric {{ border: 1px solid #dfe4ea; border-radius: 6px; padding: 10px; background: #fbfcfd; min-height: 54px; }}
    .metric span {{ display: block; color: #5b6673; font-size: 12px; margin-bottom: 6px; }}
    .metric strong {{ font-size: 15px; word-break: break-word; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e4e8ed; text-align: left; padding: 8px; vertical-align: top; }}
    th {{ background: #f0f3f6; }}
    .muted {{ color: #6b7683; }}
    .notice {{ color: #5b6673; font-size: 13px; }}
  </style>
</head>
<body>
  <header>
    <h1>MANIPULANTE Observability</h1>
    <div class="notice">Read-only local dashboard. No envia ordenes. No modifica estrategia.</div>
  </header>
  <main>
    <section>
      <h2>Estado actual</h2>
      <div class="grid">
        {metric('Health', health.get('health_status', 'UNKNOWN'))}
        {metric('Bot status', hb.get('bot_status', 'UNKNOWN'))}
        {metric('Cuenta', hb.get('account', 'UNKNOWN'))}
        {metric('Modo', hb.get('mode', 'UNKNOWN'))}
        {metric('Runner PID', hb.get('runner_pid', ''))}
        {metric('Noticias', hb.get('news_status', 'UNKNOWN'))}
        {metric('Ordenes', hb.get('orders_status', 'UNKNOWN'))}
        {metric('Operacion abierta', hb.get('open_position', 'UNKNOWN'))}
        {metric('Seguro apagar PC', hb.get('safe_to_turn_off_pc', 'UNKNOWN'))}
        {metric('Ultima decision', health.get('last_decision', hb.get('last_decision', 'UNKNOWN')))}
      </div>
    </section>
    <section>
      <h2>Scorecard diario</h2>
      <div class="grid">
        {metric('Fecha', scorecard.get('date', 'UNKNOWN'))}
        {metric('Veredicto', scorecard.get('verdict', 'UNKNOWN'))}
        {metric('Trades', scorecard.get('trades_taken', 'UNKNOWN'))}
        {metric('No trade', scorecard.get('no_trade_count', 'UNKNOWN'))}
        {metric('Warnings', scorecard.get('warnings', 'UNKNOWN'))}
        {metric('Criticos', scorecard.get('critical_errors', 'UNKNOWN'))}
      </div>
    </section>
    <section>
      <h2>Stress tests y promotion gate</h2>
      <div class="grid">
        {metric('Stress status', health.get('stress_status', 'UNKNOWN'))}
        {metric('Promotion progress', health.get('promotion_progress', 'UNKNOWN'))}
        {metric('Promotion gate', (health.get('promotion_gate') or {{}}).get('summary', 'UNKNOWN'))}
      </div>
    </section>
    <section>
      <h2>Ultimas 20 decisiones</h2>
      {table(data['decisions'], ['timestamp_ny', 'decision', 'reason', 'news_gate', 'data_gate', 'signal_gate', 'order_sent'])}
    </section>
    <section>
      <h2>Incidents</h2>
      {table(data['incidents'], ['timestamp_utc', 'severity', 'category', 'description', 'resolved', 'source'])}
    </section>
    <section>
      <h2>Trades / fills manuales</h2>
      <div class="grid">
        {metric('BE netos / net R', round(net_r, 4))}
        {metric('Comisiones', round(commissions, 4))}
      </div>
      {table(fills, ['date', 'ticket', 'side', 'lot', 'entry_actual', 'exit_actual', 'commission', 'swap', 'slippage_pips', 'gross_r', 'net_r', 'outcome'])}
    </section>
    <section>
      <h2>DB</h2>
      <div class="grid">
        {metric('DB path', data['db_path'])}
        {metric('Heartbeats', data['counts']['heartbeats'])}
        {metric('Decisiones', data['counts']['decisions'])}
        {metric('Incidents', data['counts']['incidents'])}
      </div>
    </section>
  </main>
</body>
</html>
"""
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    HTML_PATH.write_text(content, encoding="utf-8")
    return HTML_PATH


def run_streamlit_dashboard() -> None:
    import streamlit as st

    data = load_dashboard_data()
    hb = data["latest_heartbeat"]
    health = data["latest_health"]
    scorecard = data["latest_scorecard"]

    st.set_page_config(page_title="MANIPULANTE Observability", layout="wide")
    st.title("MANIPULANTE Observability")
    st.caption("Read-only local dashboard. No envia ordenes. No modifica estrategia.")
    cols = st.columns(5)
    cols[0].metric("Health", health.get("health_status", "UNKNOWN"))
    cols[1].metric("Bot", hb.get("bot_status", "UNKNOWN"))
    cols[2].metric("Noticias", hb.get("news_status", "UNKNOWN"))
    cols[3].metric("Ordenes", hb.get("orders_status", "UNKNOWN"))
    cols[4].metric("Open position", hb.get("open_position", "UNKNOWN"))

    st.subheader("Estado actual")
    st.json({
        "account": hb.get("account"),
        "mode": hb.get("mode"),
        "runner_pid": hb.get("runner_pid"),
        "safe_to_turn_off_pc": hb.get("safe_to_turn_off_pc"),
        "last_decision": health.get("last_decision", hb.get("last_decision")),
        "recommendation": health.get("recommendation"),
    })

    st.subheader("Scorecard diario")
    st.json(scorecard)

    st.subheader("Stress tests y promotion gate")
    st.json({
        "stress_status": health.get("stress_status"),
        "promotion_progress": health.get("promotion_progress"),
        "promotion_gate": health.get("promotion_gate"),
    })

    st.subheader("Ultimas 20 decisiones")
    st.dataframe(data["decisions"], use_container_width=True)

    st.subheader("Incidents")
    st.dataframe(data["incidents"], use_container_width=True)

    st.subheader("Trades / fills manuales")
    st.dataframe(data["fills_manual"], use_container_width=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase44 MANIPULANTE read-only dashboard.")
    parser.add_argument("--export-html", action="store_true", help="Generate fallback static dashboard.html and exit.")
    args = parser.parse_args()
    if args.export_html:
        path = build_static_html()
        print(f"HTML dashboard ready: {path}")
        return 0
    try:
        run_streamlit_dashboard()
    except ModuleNotFoundError:
        path = build_static_html()
        print(f"Streamlit not installed. HTML dashboard ready: {path}")
    except sqlite3.Error as exc:
        path = build_static_html()
        print(f"SQLite dashboard fallback generated after DB warning: {exc}. HTML: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
