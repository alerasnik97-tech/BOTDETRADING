from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
MANIPULANTE_DIR = ROOT / "MANIPULANTE"
OBS_DIR = MANIPULANTE_DIR / "16_OBSERVABILITY"
DB_PATH = OBS_DIR / "db" / "manipulante_observability.sqlite"
JSONL_PATH = OBS_DIR / "jsonl" / "bot_events.jsonl"
DAILY_DIR = OBS_DIR / "daily"
DASHBOARD_DIR = OBS_DIR / "dashboard"
DOCS_DIR = OBS_DIR / "docs"
EXPORTS_DIR = OBS_DIR / "exports"
NY = ZoneInfo("America/New_York")
ARG = ZoneInfo("America/Argentina/Buenos_Aires")

SENSITIVE_KEY_RE = re.compile(
    r"(password|passwd|token|secret|api[_-]?key|apikey|private[_-]?key|"
    r"access[_-]?token|refresh[_-]?token|client[_-]?secret|authorization|"
    r"bearer|login|account_number)",
    re.IGNORECASE,
)
LONG_SECRET_RE = re.compile(r"(?i)(bearer\s+)?[A-Za-z0-9_./+=:-]{24,}")


def ensure_observability_dirs() -> None:
    for path in (OBS_DIR, DB_PATH.parent, JSONL_PATH.parent, DAILY_DIR, DASHBOARD_DIR, DOCS_DIR, EXPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_arg() -> str:
    return datetime.now(ARG).isoformat()


def now_ny() -> str:
    return datetime.now(NY).isoformat()


def parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def timestamp_bundle(value: Any | None = None) -> tuple[str, str, str]:
    dt = parse_dt(value)
    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (
        dt.astimezone(timezone.utc).isoformat(),
        dt.astimezone(ARG).isoformat(),
        dt.astimezone(NY).isoformat(),
    )


def mask_scalar(key: str, value: Any) -> Any:
    if value is None:
        return None
    if SENSITIVE_KEY_RE.search(key):
        text = str(value)
        if len(text) <= 4:
            return "***"
        return f"{text[:2]}***{text[-2:]}"
    if isinstance(value, str) and ("password" in value.lower() or "token" in value.lower()):
        return "***MASKED***"
    return value


def mask_sensitive(value: Any, key: str = "") -> Any:
    if isinstance(value, dict):
        return {str(k): mask_sensitive(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [mask_sensitive(v, key) for v in value]
    masked = mask_scalar(key, value)
    if isinstance(masked, str) and SENSITIVE_KEY_RE.search(key):
        return masked
    if isinstance(masked, str) and any(token in key.lower() for token in ("authorization", "bearer")):
        return "***MASKED***"
    return masked


def safe_json(value: Any) -> str:
    return json.dumps(mask_sensitive(value), ensure_ascii=False, sort_keys=True)


def parse_key_value_text(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = str(mask_sensitive(value.strip(), key.strip()))
    return data


def connect() -> sqlite3.Connection:
    ensure_observability_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH) -> dict[str, Any]:
    ensure_observability_dirs()
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS bot_heartbeats (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              timestamp_utc TEXT NOT NULL,
              timestamp_arg TEXT,
              timestamp_ny TEXT,
              bot_status TEXT,
              account TEXT,
              mode TEXT,
              runner_pid TEXT,
              mt5_status TEXT,
              news_status TEXT,
              orders_status TEXT,
              open_position TEXT,
              safe_to_turn_off_pc TEXT,
              last_decision TEXT,
              raw_source TEXT
            );

            CREATE TABLE IF NOT EXISTS decisions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              timestamp_utc TEXT NOT NULL,
              timestamp_arg TEXT,
              timestamp_ny TEXT,
              decision TEXT,
              reason TEXT,
              session_state TEXT,
              news_gate TEXT,
              data_gate TEXT,
              signal_gate TEXT,
              order_sent TEXT,
              source_file TEXT
            );

            CREATE TABLE IF NOT EXISTS daily_scorecards (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              date TEXT NOT NULL,
              verdict TEXT,
              trades_taken INTEGER,
              no_trade_count INTEGER,
              warnings INTEGER,
              critical_errors INTEGER,
              open_position_end TEXT,
              safe_shutdown TEXT,
              notes TEXT
            );

            CREATE TABLE IF NOT EXISTS incidents (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              timestamp_utc TEXT NOT NULL,
              severity TEXT,
              category TEXT,
              description TEXT,
              resolved TEXT,
              source TEXT
            );

            CREATE TABLE IF NOT EXISTS fills_manual (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              date TEXT,
              ticket TEXT,
              side TEXT,
              lot REAL,
              entry_expected REAL,
              entry_actual REAL,
              exit_expected REAL,
              exit_actual REAL,
              commission REAL,
              swap REAL,
              slippage_pips REAL,
              gross_r REAL,
              net_r REAL,
              outcome TEXT,
              notes TEXT
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_bot_heartbeats_unique
              ON bot_heartbeats(timestamp_utc, raw_source);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_decisions_unique
              ON decisions(timestamp_utc, decision, source_file);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_scorecards_date
              ON daily_scorecards(date);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_incidents_unique
              ON incidents(timestamp_utc, severity, category, description, source);
            """
        )
        conn.commit()
        tables = [row[0] for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        return {"db_path": str(db_path), "tables": tables, "created": True}
    finally:
        conn.close()


def insert_heartbeat(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO bot_heartbeats (
          timestamp_utc, timestamp_arg, timestamp_ny, bot_status, account, mode,
          runner_pid, mt5_status, news_status, orders_status, open_position,
          safe_to_turn_off_pc, last_decision, raw_source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row.get("timestamp_utc"),
            row.get("timestamp_arg"),
            row.get("timestamp_ny"),
            row.get("bot_status"),
            row.get("account"),
            row.get("mode"),
            row.get("runner_pid"),
            row.get("mt5_status"),
            row.get("news_status"),
            row.get("orders_status"),
            row.get("open_position"),
            row.get("safe_to_turn_off_pc"),
            row.get("last_decision"),
            row.get("raw_source"),
        ),
    )


def insert_decision(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO decisions (
          timestamp_utc, timestamp_arg, timestamp_ny, decision, reason,
          session_state, news_gate, data_gate, signal_gate, order_sent, source_file
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row.get("timestamp_utc"),
            row.get("timestamp_arg"),
            row.get("timestamp_ny"),
            row.get("decision"),
            row.get("reason"),
            row.get("session_state"),
            row.get("news_gate"),
            row.get("data_gate"),
            row.get("signal_gate"),
            row.get("order_sent"),
            row.get("source_file"),
        ),
    )


def insert_daily_scorecard(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO daily_scorecards (
          date, verdict, trades_taken, no_trade_count, warnings, critical_errors,
          open_position_end, safe_shutdown, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(date) DO UPDATE SET
          verdict=excluded.verdict,
          trades_taken=excluded.trades_taken,
          no_trade_count=excluded.no_trade_count,
          warnings=excluded.warnings,
          critical_errors=excluded.critical_errors,
          open_position_end=excluded.open_position_end,
          safe_shutdown=excluded.safe_shutdown,
          notes=excluded.notes
        """,
        (
            row.get("date"),
            row.get("verdict"),
            row.get("trades_taken"),
            row.get("no_trade_count"),
            row.get("warnings"),
            row.get("critical_errors"),
            row.get("open_position_end"),
            row.get("safe_shutdown"),
            row.get("notes"),
        ),
    )


def insert_incident(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO incidents (
          timestamp_utc, severity, category, description, resolved, source
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            row.get("timestamp_utc"),
            row.get("severity"),
            row.get("category"),
            row.get("description"),
            row.get("resolved", "NO"),
            row.get("source"),
        ),
    )


def append_event(event_type: str, status: str, source: str, details: dict[str, Any]) -> None:
    ensure_observability_dirs()
    event = {
        "timestamp": utc_now(),
        "event_type": event_type,
        "status": status,
        "source": source,
        "details": mask_sensitive(details),
    }
    with JSONL_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def latest_row(conn: sqlite3.Connection, table: str, order_col: str = "id") -> dict[str, Any] | None:
    allowed = {"bot_heartbeats", "decisions", "daily_scorecards", "incidents", "fills_manual"}
    if table not in allowed:
        raise ValueError(f"Unsupported table: {table}")
    row = conn.execute(f"SELECT * FROM {table} ORDER BY {order_col} DESC LIMIT 1").fetchone()
    return dict(row) if row else None


def recent_rows(conn: sqlite3.Connection, table: str, limit: int = 20) -> list[dict[str, Any]]:
    allowed = {"bot_heartbeats", "decisions", "daily_scorecards", "incidents", "fills_manual"}
    if table not in allowed:
        raise ValueError(f"Unsupported table: {table}")
    rows = conn.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


def table_count(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
    return int(row["n"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Create Phase44 MANIPULANTE observability SQLite schema.")
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()
    result = init_db()
    append_event("db_init", "OK", "phase44_observability_db.py", result)
    if args.print_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"SQLite ready: {DB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
