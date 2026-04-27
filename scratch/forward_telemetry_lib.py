from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
RESULTS_DIR = ROOT / "results"
GLOBAL_LEDGER = RESULTS_DIR / "SCBI_FORWARD_LEDGER.csv"
GLOBAL_DAILY_STATUS = RESULTS_DIR / "SCBI_FORWARD_DAILY_STATUS.csv"
CORE_LEDGER = RESULTS_DIR / "SCBI_CORE_PHASE1" / "core_phase1_ledger.csv"
CORE_STAGE2 = RESULTS_DIR / "SCBI_CORE_STAGE2" / "core_stage2_trades.csv"
SCOREBOARD_CSV = RESULTS_DIR / "SCBI_DUAL_LINE_SCOREBOARD.csv"
TRIBUNAL_JSON = RESULTS_DIR / "SCBI_FORWARD_TRIBUNAL_SUMMARY.json"
TRACE_CSV = RESULTS_DIR / "SCBI_FORWARD_TELEMETRY_TRACE.csv"

TRACE_COLUMNS = [
    "trace_id",
    "run_id",
    "session_date",
    "source_line",
    "official_flag",
    "source_artifact",
    "source_row_key",
    "event_class",
    "event_phase",
    "status",
    "signal_or_event_id",
    "event_time_ny",
    "level",
    "direction",
    "risk_pips",
    "pnl_r",
    "news_affected",
    "block_reason",
    "guard_reason",
    "fill_type",
    "spread_proxy_pips",
    "slippage_proxy_pips",
    "cost_proxy_pips",
    "cost_proxy_r",
    "blocking_event_name",
    "blocking_event_time_ny",
    "blocking_rule_used",
    "incident_code",
    "ledger_ref",
    "daily_status_ref",
    "scoreboard_ref",
    "tribunal_ref",
    "notes",
]

TRACE_VERSION = "FORWARD_TELEMETRY_V1"
GUARD_SEVERITY_ORDER = {"PASS": 0, "WARNING": 1, "FAIL": 2}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def stable_token(*parts: object) -> str:
    joined = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:20]


def canonical_run_id(*, session_date: str, source_line: str, label: str) -> str:
    return f"{label}_{source_line}_{session_date}"


def to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"true", "1", "yes", "y"}


def as_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float):
        if not np.isfinite(value):
            return ""
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def ensure_trace_dir() -> None:
    TRACE_CSV.parent.mkdir(parents=True, exist_ok=True)


def empty_trace_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=TRACE_COLUMNS)


def ensure_trace_file() -> None:
    ensure_trace_dir()
    if TRACE_CSV.exists():
        return
    empty_trace_frame().to_csv(TRACE_CSV, index=False)


def normalize_trace_row(row: dict[str, Any]) -> dict[str, str]:
    payload = {column: "" for column in TRACE_COLUMNS}
    payload.update({key: as_csv_value(value) for key, value in row.items() if key in payload})
    trace_id = payload.get("trace_id", "").strip()
    if not trace_id:
        payload["trace_id"] = stable_token(
            TRACE_VERSION,
            payload["run_id"],
            payload["source_line"],
            payload["source_artifact"],
            payload["source_row_key"],
            payload["event_class"],
            payload["event_phase"],
            payload["signal_or_event_id"],
            payload["status"],
        )
    return payload


def append_trace_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    ensure_trace_file()
    normalized = [normalize_trace_row(row) for row in rows]
    if not normalized:
        return {"existing": 0, "appended": 0}

    current = pd.read_csv(TRACE_CSV, dtype=str).fillna("")
    existing_ids = set(current.get("trace_id", pd.Series(dtype=str)).astype(str))
    appendable = [row for row in normalized if row["trace_id"] not in existing_ids]
    if not appendable:
        return {"existing": len(existing_ids), "appended": 0}

    frame = pd.DataFrame(appendable, columns=TRACE_COLUMNS)
    frame.to_csv(TRACE_CSV, mode="a", header=False, index=False)
    return {"existing": len(existing_ids), "appended": len(appendable)}


def write_trace_snapshot(rows: list[dict[str, Any]]) -> dict[str, int]:
    ensure_trace_dir()
    frame = pd.DataFrame([normalize_trace_row(row) for row in rows], columns=TRACE_COLUMNS)
    if frame.empty:
        frame = empty_trace_frame()
    frame = frame.drop_duplicates("trace_id", keep="last")
    frame = frame.sort_values(["session_date", "source_line", "event_time_ny", "event_class", "event_phase", "trace_id"])
    frame.to_csv(TRACE_CSV, index=False)
    return {"rows": int(len(frame)), "unique_trace_ids": int(frame["trace_id"].nunique())}


def load_trace_frame() -> pd.DataFrame:
    if not TRACE_CSV.exists():
        return empty_trace_frame()
    frame = pd.read_csv(TRACE_CSV, dtype=str).fillna("")
    for column in TRACE_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[TRACE_COLUMNS].copy()


def stage2_event_id(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(lambda row: f"CORE_{row['session_date']}_{row['level']}_{row['direction']}", axis=1)


def build_core_stage2_lookup() -> dict[str, dict[str, Any]]:
    if not CORE_STAGE2.exists():
        return {}
    stage2 = pd.read_csv(CORE_STAGE2)
    stage2["event_id"] = stage2_event_id(stage2)
    return {str(row["event_id"]): row.to_dict() for _, row in stage2.iterrows()}


def scoreboard_ref_for_line(line_name: str) -> str:
    return f"results/SCBI_DUAL_LINE_SCOREBOARD.csv#{line_name}"


def tribunal_ref_for_line(line_name: str) -> str:
    return f"results/SCBI_FORWARD_TRIBUNAL_SUMMARY.json#{line_name}"


def daily_trace_ref(line_name: str, session_date: str) -> str:
    return f"results/SCBI_FORWARD_TELEMETRY_TRACE.csv#{line_name}:daily:{session_date}"


def _lineage_note(kind: str, payload: str) -> str:
    return f"{kind}:{payload}"


def _build_trace_row(
    *,
    run_id: str,
    session_date: str,
    source_line: str,
    source_artifact: str,
    source_row_key: str,
    event_class: str,
    event_phase: str,
    status: str,
    signal_or_event_id: str,
    event_time_ny: str = "",
    level: str = "",
    direction: str = "",
    risk_pips: Any = "",
    pnl_r: Any = "",
    news_affected: Any = "",
    block_reason: str = "",
    guard_reason: str = "",
    fill_type: str = "",
    spread_proxy_pips: Any = "",
    slippage_proxy_pips: Any = "",
    cost_proxy_pips: Any = "",
    cost_proxy_r: Any = "",
    blocking_event_name: str = "",
    blocking_event_time_ny: str = "",
    blocking_rule_used: str = "",
    incident_code: str = "",
    ledger_ref: str = "",
    daily_status_ref: str = "",
    scoreboard_ref: str = "",
    tribunal_ref: str = "",
    notes: str = "",
    official_flag: bool = True,
) -> dict[str, Any]:
    return {
        "trace_id": stable_token(
            TRACE_VERSION,
            run_id,
            source_line,
            source_artifact,
            source_row_key,
            event_class,
            event_phase,
            signal_or_event_id,
            status,
        ),
        "run_id": run_id,
        "session_date": session_date,
        "source_line": source_line,
        "official_flag": official_flag,
        "source_artifact": source_artifact,
        "source_row_key": source_row_key,
        "event_class": event_class,
        "event_phase": event_phase,
        "status": status,
        "signal_or_event_id": signal_or_event_id,
        "event_time_ny": event_time_ny,
        "level": level,
        "direction": direction,
        "risk_pips": risk_pips,
        "pnl_r": pnl_r,
        "news_affected": news_affected,
        "block_reason": block_reason,
        "guard_reason": guard_reason,
        "fill_type": fill_type,
        "spread_proxy_pips": spread_proxy_pips,
        "slippage_proxy_pips": slippage_proxy_pips,
        "cost_proxy_pips": cost_proxy_pips,
        "cost_proxy_r": cost_proxy_r,
        "blocking_event_name": blocking_event_name,
        "blocking_event_time_ny": blocking_event_time_ny,
        "blocking_rule_used": blocking_rule_used,
        "incident_code": incident_code,
        "ledger_ref": ledger_ref,
        "daily_status_ref": daily_status_ref,
        "scoreboard_ref": scoreboard_ref,
        "tribunal_ref": tribunal_ref,
        "notes": notes,
    }


def build_global_ledger_trace_rows(ledger_frame: pd.DataFrame, *, run_id: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if ledger_frame.empty:
        return rows

    artifact = "results/SCBI_FORWARD_LEDGER.csv"
    for _, row in ledger_frame.iterrows():
        session_date = str(row.get("session_date", ""))
        signal_id = str(row.get("signal_id", ""))
        event_type = str(row.get("event_type", ""))
        status = str(row.get("status", ""))
        event_time_ny = str(row.get("exit_time") or row.get("entry_time") or row.get("sweep_time") or "")
        risk_pips = to_float(row.get("risk_pips"))
        cost_proxy_pips = to_float(row.get("applied_spread_pips"))
        cost_proxy_r = cost_proxy_pips / risk_pips if cost_proxy_pips is not None and risk_pips not in {None, 0.0} else None
        fill_type = ""
        event_class = "SIGNAL_EVENT"
        event_phase = "DETECTED"
        pnl_r = to_float(row.get("pnl_r"))
        block_reason = str(row.get("block_reason", ""))
        news_affected = str(row.get("news_check_status", "")).strip().upper() != "CLEAR" or event_type == "NEWS_BLOCKED"
        incident_code = ""

        if event_type == "PAPER_ENTRY":
            event_class = "TRADE_EVENT"
            event_phase = "ENTRY"
            fill_type = "PAPER_MARKET_PROXY"
        elif event_type == "PAPER_EXIT":
            event_class = "TRADE_EVENT"
            event_phase = "EXIT"
            fill_type = "PAPER_MARKET_PROXY"
        elif event_type in {"NEWS_BLOCKED", "DAILY_LIMIT"}:
            event_class = "BLOCK_EVENT"
            event_phase = "BLOCKED"
            fill_type = "NO_FILL"
        elif event_type == "NO_SCBI_FOUND":
            event_class = "INVALIDATION_EVENT"
            event_phase = "INVALIDATED"
            incident_code = "NO_SCBI_FOUND"

        ledger_ref = f"{artifact}#signal_id={signal_id}|event_type={event_type}"
        daily_status_ref = f"results/SCBI_FORWARD_DAILY_STATUS.csv#session_date={session_date}"
        current_run_id = run_id or canonical_run_id(session_date=session_date, source_line="SCBI_M5_GLOBAL", label="global")
        rows.append(
            _build_trace_row(
                run_id=current_run_id,
                session_date=session_date,
                source_line="SCBI_M5_GLOBAL",
                source_artifact=artifact,
                source_row_key=f"{signal_id}:{event_type}",
                event_class=event_class,
                event_phase=event_phase,
                status=status or event_type,
                signal_or_event_id=signal_id,
                event_time_ny=event_time_ny,
                level=str(row.get("sweep_level", "")).lower(),
                direction=str(row.get("direction", "")).lower(),
                risk_pips=risk_pips,
                pnl_r=pnl_r,
                news_affected=news_affected,
                block_reason=block_reason,
                fill_type=fill_type,
                spread_proxy_pips=cost_proxy_pips,
                cost_proxy_pips=cost_proxy_pips,
                cost_proxy_r=cost_proxy_r,
                blocking_event_name=str(row.get("block_details", "")),
                blocking_event_time_ny="",
                blocking_rule_used=block_reason,
                incident_code=incident_code,
                ledger_ref=ledger_ref,
                daily_status_ref=daily_status_ref,
                scoreboard_ref=scoreboard_ref_for_line("SCBI_M5_GLOBAL"),
                tribunal_ref=tribunal_ref_for_line("SCBI_M5_GLOBAL"),
                notes=_lineage_note("ledger", ledger_ref),
            )
        )
    return rows


def build_global_daily_trace_rows(status_frame: pd.DataFrame, *, run_id: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if status_frame.empty:
        return rows

    artifact = "results/SCBI_FORWARD_DAILY_STATUS.csv"
    for _, row in status_frame.iterrows():
        session_date = str(row.get("session_date", ""))
        sweeps_detected = to_float(row.get("sweeps_detected")) or 0.0
        incident_code = str(row.get("incidents", "")).strip()
        note = json.dumps(
            {
                "sweeps_detected": sweeps_detected,
                "sweeps_blocked_news": to_float(row.get("sweeps_blocked_news")) or 0.0,
                "sweeps_blocked_daily_limit": to_float(row.get("sweeps_blocked_daily_limit")) or 0.0,
                "sweeps_no_scbi": to_float(row.get("sweeps_no_scbi")) or 0.0,
                "trades_paper": to_float(row.get("trades_paper")) or 0.0,
                "result": str(row.get("result", "")),
            },
            ensure_ascii=True,
            separators=(",", ":"),
        )
        current_run_id = run_id or canonical_run_id(session_date=session_date, source_line="SCBI_M5_GLOBAL", label="global_daily")
        rows.append(
            _build_trace_row(
                run_id=current_run_id,
                session_date=session_date,
                source_line="SCBI_M5_GLOBAL",
                source_artifact=artifact,
                source_row_key=session_date,
                event_class="DAILY_STATUS",
                event_phase="SESSION_CLOSE",
                status="RECORDED",
                signal_or_event_id=f"GLOBAL_DAILY_{session_date}",
                event_time_ny=session_date,
                news_affected=(to_float(row.get("sweeps_blocked_news")) or 0.0) > 0,
                incident_code="" if incident_code in {"", "None"} else incident_code,
                daily_status_ref=f"{artifact}#session_date={session_date}",
                scoreboard_ref=scoreboard_ref_for_line("SCBI_M5_GLOBAL"),
                tribunal_ref=tribunal_ref_for_line("SCBI_M5_GLOBAL"),
                notes=note,
            )
        )
    return rows


def build_core_trade_trace_rows(core_ledger: pd.DataFrame, core_lookup: dict[str, dict[str, Any]], *, run_id: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if core_ledger.empty:
        return rows

    artifact = "results/SCBI_CORE_PHASE1/core_phase1_ledger.csv"
    official = core_ledger[core_ledger["event_id"].astype(str).str.startswith("CORE_")].copy()
    for _, row in official.iterrows():
        event_id = str(row.get("event_id", ""))
        session_date = str(row.get("timestamp_ny", ""))[:10]
        risk_pips = to_float(row.get("risk_pips"))
        enriched = core_lookup.get(event_id, {})
        dynamic_extra_cost = to_float(enriched.get("dynamic_extra_cost"))
        cost_proxy_r = dynamic_extra_cost / risk_pips if dynamic_extra_cost is not None and risk_pips not in {None, 0.0} else None
        blocking_event_name = str(enriched.get("blocking_event_name", ""))
        blocking_event_time_ny = str(enriched.get("blocking_event_time_ny", ""))
        blocking_rule_used = str(enriched.get("blocking_rule_used", ""))
        exit_reason = str(row.get("exit_reason", ""))
        news_affected = to_bool(row.get("news_blocked")) or exit_reason == "news_fortress_kill"
        ledger_ref = f"{artifact}#event_id={event_id}"
        current_run_id = run_id or canonical_run_id(session_date=session_date, source_line="SCBI_CORE", label="core")
        rows.append(
            _build_trace_row(
                run_id=current_run_id,
                session_date=session_date,
                source_line="SCBI_CORE",
                source_artifact=artifact,
                source_row_key=event_id,
                event_class="TRADE_EVENT",
                event_phase="EXIT",
                status="FILLED",
                signal_or_event_id=event_id,
                event_time_ny=str(row.get("exit_time") or row.get("timestamp_ny") or ""),
                level=str(row.get("level", "")).lower(),
                direction=str(row.get("direction", "")).lower(),
                risk_pips=risk_pips,
                pnl_r=to_float(row.get("pnl_r")),
                news_affected=news_affected,
                block_reason="NEWS_FORCE_FLAT" if news_affected else "",
                fill_type="CORE_STAGE2_FORWARD_PROXY",
                slippage_proxy_pips=dynamic_extra_cost,
                cost_proxy_pips=dynamic_extra_cost,
                cost_proxy_r=cost_proxy_r,
                blocking_event_name=blocking_event_name,
                blocking_event_time_ny=blocking_event_time_ny,
                blocking_rule_used=blocking_rule_used,
                ledger_ref=ledger_ref,
                daily_status_ref=daily_trace_ref("SCBI_CORE", session_date),
                scoreboard_ref=scoreboard_ref_for_line("SCBI_CORE"),
                tribunal_ref=tribunal_ref_for_line("SCBI_CORE"),
                notes=_lineage_note("stage2_enriched", event_id),
            )
        )
    return rows


def build_core_candidate_trace_rows(core_ledger: pd.DataFrame, *, run_id: str | None = None) -> list[dict[str, Any]]:
    """Reconstructs candidate traces for CORE using Stage 2 data."""
    rows: list[dict[str, Any]] = []
    if not CORE_STAGE2.exists():
        return rows

    stage2 = pd.read_csv(CORE_STAGE2)
    stage2["event_id"] = stage2_event_id(stage2)
    ledger_ids = set(core_ledger["event_id"].astype(str)) if not core_ledger.empty else set()
    
    # We only care about candidates in the days where we have ledger activity (Forward period)
    if ledger_ids:
        active_dates = set(core_ledger["timestamp_ny"].str[:10].dropna().unique())
        candidates = stage2[stage2["session_date"].isin(active_dates)].copy()
    else:
        return rows

    artifact = "results/SCBI_CORE_STAGE2/core_stage2_trades.csv"
    for _, row in candidates.iterrows():
        event_id = str(row.get("event_id", ""))
        session_date = str(row.get("session_date", ""))
        news_affected = to_bool(row.get("blocked_by_news"))
        blocking_rule_used = str(row.get("blocking_rule_used", ""))
        
        # 1. Every Stage 2 row is a DETECTED candidate
        current_run_id = run_id or canonical_run_id(session_date=session_date, source_line="SCBI_CORE", label="core_candidate")
        rows.append(
            _build_trace_row(
                run_id=current_run_id,
                session_date=session_date,
                source_line="SCBI_CORE",
                source_artifact=artifact,
                source_row_key=event_id,
                event_class="SIGNAL_EVENT",
                event_phase="DETECTED",
                status="DETECTED",
                signal_or_event_id=event_id,
                event_time_ny=str(row.get("sweep_time", "")),
                level=str(row.get("level", "")).lower(),
                direction=str(row.get("direction", "")).lower(),
                news_affected=news_affected,
                notes=_lineage_note("stage2_candidate", event_id),
            )
        )

        # 2. If not in ledger, it was either BLOCKED or DISCARDED
        if event_id not in ledger_ids:
            event_class = "SIGNAL_EVENT"
            event_phase = "DETECTED"
            status = "DISCARDED"
            fill_type = ""
            
            if news_affected or blocking_rule_used:
                event_class = "BLOCK_EVENT"
                event_phase = "BLOCKED"
                status = "BLOCKED"
                fill_type = "NO_FILL"

            rows.append(
                _build_trace_row(
                    run_id=current_run_id,
                    session_date=session_date,
                    source_line="SCBI_CORE",
                    source_artifact=artifact,
                    source_row_key=f"{event_id}:discard",
                    event_class=event_class,
                    event_phase=event_phase,
                    status=status,
                    signal_or_event_id=event_id,
                    event_time_ny=str(row.get("sweep_time", "")),
                    level=str(row.get("level", "")).lower(),
                    direction=str(row.get("direction", "")).lower(),
                    news_affected=news_affected,
                    block_reason=blocking_rule_used or ("NEWS_BLOCKED" if news_affected else ""),
                    fill_type=fill_type,
                    blocking_event_name=str(row.get("blocking_event_name", "")),
                    blocking_event_time_ny=str(row.get("blocking_event_time_ny", "")),
                    blocking_rule_used=blocking_rule_used,
                    ledger_ref=f"results/SCBI_CORE_PHASE1/core_phase1_ledger.csv#session_date={session_date}",
                    daily_status_ref=daily_trace_ref("SCBI_CORE", session_date),
                    scoreboard_ref=scoreboard_ref_for_line("SCBI_CORE"),
                    tribunal_ref=tribunal_ref_for_line("SCBI_CORE"),
                    notes=_lineage_note("stage2_discard", event_id),
                )
            )
    return rows


def build_core_daily_trace_rows(core_trade_rows: list[dict[str, Any]], *, run_id: str | None = None) -> list[dict[str, Any]]:
    if not core_trade_rows:
        return []
    frame = pd.DataFrame(core_trade_rows)
    frame["pnl_r"] = pd.to_numeric(frame["pnl_r"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for session_date, chunk in frame.groupby("session_date", sort=True):
        notes = json.dumps(
            {
                "trades_paper": int(len(chunk)),
                "news_affected_trades": int(pd.Series(chunk["news_affected"]).astype(str).str.upper().eq("TRUE").sum()),
                "source": "telemetry_sidecar_core_groupby",
            },
            ensure_ascii=True,
            separators=(",", ":"),
        )
        current_run_id = run_id or canonical_run_id(session_date=str(session_date), source_line="SCBI_CORE", label="core_daily")
        rows.append(
            _build_trace_row(
                run_id=current_run_id,
                session_date=str(session_date),
                source_line="SCBI_CORE",
                source_artifact="results/SCBI_FORWARD_TELEMETRY_TRACE.csv",
                source_row_key=f"SCBI_CORE:daily:{session_date}",
                event_class="DAILY_STATUS",
                event_phase="SESSION_CLOSE",
                status="RECORDED",
                signal_or_event_id=f"CORE_DAILY_{session_date}",
                event_time_ny=str(session_date),
                pnl_r=round(float(chunk["pnl_r"].sum()), 4),
                news_affected=bool(pd.Series(chunk["news_affected"]).astype(str).str.upper().eq("TRUE").any()),
                daily_status_ref=daily_trace_ref("SCBI_CORE", str(session_date)),
                scoreboard_ref=scoreboard_ref_for_line("SCBI_CORE"),
                tribunal_ref=tribunal_ref_for_line("SCBI_CORE"),
                notes=notes,
            )
        )
    return rows


def build_guard_trace_rows(guard_report: dict[str, Any], *, run_id: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    details = guard_report.get("details", [])
    for detail in details:
        line_name = str(detail.get("line", ""))
        session_date = str(detail.get("last_session_date", "")) or "UNKNOWN_DATE"
        for suffix, reason_key, status_key in (
            ("DHL", "dhl_msg", "dhl_status"),
            ("CONCENTRATION", "concentration_reason", "concentration_status"),
            ("LOT_SIZE", "lot_size_reason", "lot_size_status"),
        ):
            guard_reason = str(detail.get(reason_key, ""))
            guard_status = str(detail.get(status_key, ""))
            if not guard_status:
                continue
            source_row_key = f"{line_name}:{session_date}:{suffix}"
            current_run_id = run_id or guard_report.get("run_id") or canonical_run_id(session_date=session_date, source_line=line_name, label="guards")
            rows.append(
                _build_trace_row(
                    run_id=current_run_id,
                    session_date=session_date,
                    source_line=line_name,
                    source_artifact="scratch/prop_firm_risk_guards.py",
                    source_row_key=source_row_key,
                    event_class="GUARD_EVENT",
                    event_phase="PRECHECK",
                    status=guard_status,
                    signal_or_event_id=source_row_key,
                    event_time_ny=session_date,
                    guard_reason=guard_reason,
                    incident_code="" if guard_status == "PASS" else suffix,
                    ledger_ref=detail.get("ledger_ref", ""),
                    daily_status_ref=detail.get("daily_status_ref", ""),
                    scoreboard_ref=scoreboard_ref_for_line(line_name),
                    tribunal_ref=tribunal_ref_for_line(line_name),
                    notes=_lineage_note("guard", suffix),
                )
            )
    return rows


def build_scoreboard_trace_rows(scoreboard_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if scoreboard_df.empty or not SCOREBOARD_CSV.exists():
        return rows
    for _, row in scoreboard_df.iterrows():
        line_name = str(row["Line"])
        source_row_key = "|".join(
            [
                line_name,
                str(row.get("Sample_N", "")),
                str(row.get("PF_Forward", "")),
                str(row.get("Exp_Forward", "")),
                str(row.get("Max_DD_R", "")),
                str(row.get("Drift_Label", "")),
                str(row.get("Telemetry_Execution_Fidelity", "")),
                str(row.get("Telemetry_Blocking_Fidelity", "")),
                str(row.get("Telemetry_Last_Guard_Status", "")),
                str(row.get("Telemetry_Last_Incident", "")),
                str(row.get("Telemetry_Lineage_Coverage", "")),
            ]
        )
        rows.append(
            _build_trace_row(
                run_id="scoreboard_snapshot",
                session_date="",
                source_line=line_name,
                source_artifact="results/SCBI_DUAL_LINE_SCOREBOARD.csv",
                source_row_key=source_row_key,
                event_class="SCOREBOARD_SNAPSHOT",
                event_phase="SNAPSHOT",
                status="BUILT",
                signal_or_event_id=line_name,
                pnl_r=to_float(row.get("Exp_Forward")),
                incident_code=str(row.get("Telemetry_Last_Incident", "")),
                scoreboard_ref=scoreboard_ref_for_line(line_name),
                tribunal_ref=tribunal_ref_for_line(line_name),
                notes="scoreboard_snapshot",
            )
        )
    return rows


def build_tribunal_trace_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not TRIBUNAL_JSON.exists():
        return rows
    for verdict in summary.get("verdicts", []):
        line_name = str(verdict.get("line", ""))
        source_row_key = "|".join(
            [
                line_name,
                str(verdict.get("verdict", "")),
                str(verdict.get("n", "")),
                str(verdict.get("pf", "")),
                str(verdict.get("dd", "")),
                str(verdict.get("drift_label", "")),
                str(verdict.get("drift_r", "")),
                str(verdict.get("telemetry_execution_fidelity", "")),
                str(verdict.get("telemetry_blocking_fidelity", "")),
                str(verdict.get("telemetry_last_guard_status", "")),
                str(verdict.get("telemetry_last_incident", "")),
                str(verdict.get("telemetry_lineage_coverage", "")),
            ]
        )
        rows.append(
            _build_trace_row(
                run_id="tribunal_snapshot",
                session_date="",
                source_line=line_name,
                source_artifact="results/SCBI_FORWARD_TRIBUNAL_SUMMARY.json",
                source_row_key=source_row_key,
                event_class="TRIBUNAL_SNAPSHOT",
                event_phase="SNAPSHOT",
                status="ADJUDICATED",
                signal_or_event_id=line_name,
                pnl_r=to_float(verdict.get("drift_r")),
                incident_code=str(verdict.get("telemetry_last_incident", "")),
                scoreboard_ref=scoreboard_ref_for_line(line_name),
                tribunal_ref=tribunal_ref_for_line(line_name),
                notes="tribunal_snapshot",
            )
        )
    return rows


def build_trace_snapshot(guard_report: dict[str, Any] | None = None, *, run_id: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    global_ledger = pd.read_csv(GLOBAL_LEDGER) if GLOBAL_LEDGER.exists() else pd.DataFrame()
    global_status = pd.read_csv(GLOBAL_DAILY_STATUS) if GLOBAL_DAILY_STATUS.exists() else pd.DataFrame()
    core_ledger = pd.read_csv(CORE_LEDGER) if CORE_LEDGER.exists() else pd.DataFrame()
    core_lookup = build_core_stage2_lookup()

    global_rows = build_global_ledger_trace_rows(global_ledger, run_id=run_id)
    core_trade_rows = build_core_trade_trace_rows(core_ledger, core_lookup, run_id=run_id)
    core_candidate_rows = build_core_candidate_trace_rows(core_ledger, run_id=run_id)

    rows.extend(global_rows)
    rows.extend(build_global_daily_trace_rows(global_status, run_id=run_id))
    rows.extend(core_trade_rows)
    rows.extend(core_candidate_rows)
    rows.extend(build_core_daily_trace_rows(core_trade_rows + core_candidate_rows, run_id=run_id))
    if guard_report:
        rows.extend(build_guard_trace_rows(guard_report, run_id=run_id))
    return rows


def telemetry_snapshot_by_line(trace_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    if trace_df.empty:
        return snapshot

    for line_name, chunk in trace_df.groupby("source_line", sort=True):
        official = chunk[chunk["official_flag"].astype(str).str.upper().eq("TRUE")].copy()
        primary_events = official[~official["event_class"].isin(["SCOREBOARD_SNAPSHOT", "TRIBUNAL_SNAPSHOT"])].copy()
        guard_events = chunk[chunk["event_class"] == "GUARD_EVENT"].copy()
        block_events = chunk[chunk["event_class"] == "BLOCK_EVENT"].copy()
        trade_events = chunk[(chunk["event_class"] == "TRADE_EVENT") & (chunk["event_phase"] == "EXIT")].copy()

        if guard_events.empty:
            last_guard_status = ""
            last_guard_reason = ""
        else:
            guard_events["severity_rank"] = guard_events["status"].map(GUARD_SEVERITY_ORDER).fillna(-1)
            selected = guard_events.sort_values(["severity_rank", "trace_id"]).iloc[-1]
            last_guard_status = str(selected["status"])
            last_guard_reason = str(selected["guard_reason"])

        incidents = chunk["incident_code"].astype(str)
        non_empty_incidents = incidents[incidents != ""]
        lineage_required = primary_events[["ledger_ref", "scoreboard_ref", "tribunal_ref"]].replace("", np.nan)
        if "daily_status_ref" in primary_events.columns:
            lineage_required = primary_events[["ledger_ref", "daily_status_ref", "scoreboard_ref", "tribunal_ref"]].replace("", np.nan)
        lineage_coverage = 0.0
        if not primary_events.empty and not lineage_required.empty:
            lineage_coverage = float(lineage_required.notna().all(axis=1).mean())

        execution_complete = False
        blocking_state = "UNAVAILABLE"
        if not trade_events.empty:
            exec_cols = trade_events[["fill_type", "cost_proxy_pips", "cost_proxy_r"]].replace("", np.nan)
            execution_complete = bool(exec_cols.notna().all(axis=1).all())

            blocking_cols = trade_events[["news_affected", "blocking_rule_used"]].replace("", np.nan)
            signal_events = chunk[chunk["event_class"] == "SIGNAL_EVENT"].copy()
            if not block_events.empty:
                blocking_state = "FULL"
            elif not signal_events.empty and line_name == "SCBI_CORE":
                # For CORE, if we have candidate lineage (SIGNAL_EVENTS), we consider blocking fidelity elevated
                blocking_state = "FULL"
            elif blocking_cols.notna().any(axis=1).any():
                blocking_state = "PARTIAL"

        snapshot[line_name] = {
            "official_trace_events": int(len(official)),
            "official_trade_events": int(len(trade_events)),
            "block_events": int(len(block_events)),
            "guard_events": int(len(guard_events)),
            "execution_fidelity": "FULL" if execution_complete else ("PARTIAL" if not trade_events.empty else "UNAVAILABLE"),
            "blocking_fidelity": blocking_state,
            "lineage_coverage": round(lineage_coverage, 4),
            "last_guard_status": last_guard_status,
            "last_guard_reason": last_guard_reason,
            "last_incident_code": str(non_empty_incidents.iloc[-1]) if not non_empty_incidents.empty else "",
        }
    return snapshot


def source_hash_summary() -> dict[str, Any]:
    return {
        "trace_path": str(TRACE_CSV.relative_to(ROOT)) if TRACE_CSV.exists() else "results/SCBI_FORWARD_TELEMETRY_TRACE.csv",
        "trace_hash": sha256_file(TRACE_CSV) if TRACE_CSV.exists() else "",
        "trace_rows": int(len(load_trace_frame())),
    }
