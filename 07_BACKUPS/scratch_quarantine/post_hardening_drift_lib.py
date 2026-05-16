from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scratch.forward_telemetry_lib import TRACE_CSV, load_trace_frame, telemetry_snapshot_by_line

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
RESULTS_DIR = ROOT / "results"

NY_TZ = "America/New_York"
BASELINE_VERSION = "POST_HARDENING_SIGNAL_DRIFT_V1"
BASELINE_JSON = RESULTS_DIR / "SCBI_SIGNAL_DRIFT_BASELINE.json"
REPORT_JSON = RESULTS_DIR / "SCBI_SIGNAL_DRIFT_REPORT.json"
VALIDATION_JSON = RESULTS_DIR / "SCBI_SIGNAL_DRIFT_VALIDATION.json"
SCOREBOARD_CSV = RESULTS_DIR / "SCBI_DUAL_LINE_SCOREBOARD.csv"
TRIBUNAL_JSON = RESULTS_DIR / "SCBI_FORWARD_TRIBUNAL_SUMMARY.json"
POST_STATUS_JSON = ROOT / "POST_HARDENING_DRIFT_STATUS.json"

MIN_COMPARABLE_N = 10
MIN_FREQUENCY_WEEKS = 4
VALIDATION_WINDOWS = (20, 40)
SENSITIVITY_SAMPLE_WINDOWS = 12

FREQUENCY_REL_THRESHOLD = 0.30
SESSION_SHARE_THRESHOLD = 0.20
LIQUIDITY_SHARE_THRESHOLD = 0.18
DIRECTION_SHARE_THRESHOLD = 0.20
BLOCKING_SHARE_THRESHOLD = 0.10
EXECUTION_BUCKET_THRESHOLD = 0.20
EXECUTION_COST_DELTA_R_THRESHOLD = 0.015
EXPECTANCY_WARNING_DELTA = -0.35
EXPECTANCY_STRUCTURAL_DELTA = -0.55
PF_WARNING_THRESHOLD = 1.20
PF_STRUCTURAL_THRESHOLD = 1.00
PF_CATASTROPHIC_THRESHOLD = 0.50
EXPECTANCY_CATASTROPHIC_DELTA = -0.75
MAX_DD_WORSENING_R = 4.0

DIMENSION_NAMES = {
    "A_SIGNAL_FREQUENCY_DRIFT": "SIGNAL_FREQUENCY_DRIFT",
    "B_SESSION_COMPOSITION_DRIFT": "SESSION_COMPOSITION_DRIFT",
    "C_DIRECTIONAL_DRIFT": "DIRECTIONAL_DRIFT",
    "D_LIQUIDITY_SOURCE_OR_TRIGGER_DRIFT": "LIQUIDITY_SOURCE_OR_TRIGGER_DRIFT",
    "E_PERFORMANCE_DISTRIBUTION_DRIFT": "PERFORMANCE_DISTRIBUTION_DRIFT",
    "F_BLOCKING_PROFILE_DRIFT": "BLOCKING_PROFILE_DRIFT",
    "G_EXECUTION_PROFILE_DRIFT": "EXECUTION_PROFILE_DRIFT",
}

SESSION_ORDER = ("london", "asia", "prev_day", "unknown")
GLOBAL_LEVEL_ORDER = ("london_l", "london_h", "asia_l", "asia_h", "pdl", "pdh", "unknown")
CORE_LEVEL_ORDER = ("london_l", "london_h", "asia_l", "asia_h", "unknown")
RISK_BUCKET_ORDER = ("tight", "medium", "wide", "unknown")

LINE_CONFIGS: dict[str, dict[str, Any]] = {
    "SCBI_M5_GLOBAL": {
        "historical_path": RESULTS_DIR / "SCBI_2020_2025_DURABILITY" / "trades_baseline.csv",
        "forward_path": RESULTS_DIR / "SCBI_FORWARD_LEDGER.csv",
        "level_order": GLOBAL_LEVEL_ORDER,
        "historical_cost_model": "INSTITUTIONAL_0P4_PROXY_FROM_0P3_BASELINE",
        "historical_cost_note": "Se aplica delta fijo de 0.1 pips sobre trades historicos de 0.3 pips para rebaselinar bajo costo institucional vigente.",
        "historical_time_note": "Los timestamps historicos muestran offsets mixtos -05:00/-04:00, confirmando awareness DST en los CSV recalculados.",
        "forward_cost_note": "Ledger forward expone spread aplicado por trade, pero no slippage granular por fill; la comparacion de ejecucion queda parcial.",
    },
    "SCBI_CORE": {
        "historical_path": RESULTS_DIR / "SCBI_CORE_STAGE2" / "core_stage2_trades.csv",
        "forward_path": RESULTS_DIR / "SCBI_CORE_PHASE1" / "core_phase1_ledger.csv",
        "level_order": CORE_LEVEL_ORDER,
        "historical_cost_model": "STAGE2_DYNAMIC_EXECUTION_HARDENED",
        "historical_cost_note": "Se usa pnl_r_dynamic del Stage-2 para absorber el slippage endurecido disponible en la serie historica CORE.",
        "historical_time_note": "Los timestamps historicos muestran offsets mixtos -05:00/-04:00, confirmando awareness DST en los CSV recalculados.",
        "forward_cost_note": "Ledger forward CORE no expone spread/slippage por fill; la comparacion de ejecucion queda parcial y solo usa proxies de riesgo.",
    },
}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_ny_timestamps(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series.astype(str), utc=True, errors="coerce")
    return parsed.dt.tz_convert(NY_TZ)


def session_from_level(level: str) -> str:
    value = str(level or "").strip().lower()
    if value.startswith("london_"):
        return "london"
    if value.startswith("asia_"):
        return "asia"
    if value in {"pdh", "pdl"}:
        return "prev_day"
    return "unknown"


def risk_bucket(value: Any) -> str:
    try:
        risk = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not np.isfinite(risk):
        return "unknown"
    if risk < 5.0:
        return "tight"
    if risk < 10.0:
        return "medium"
    return "wide"


def normalize_distribution(series: pd.Series, keys: tuple[str, ...]) -> dict[str, float]:
    values = series.fillna("unknown").astype(str).str.lower()
    counts = values.value_counts(normalize=True)
    return {key: round(float(counts.get(key, 0.0)), 4) for key in keys}


def compute_pf(pnls: pd.Series) -> float:
    values = pnls.dropna().astype(float)
    if values.empty:
        return 0.0
    wins = values[values > 0.0]
    losses = values[values <= 0.0]
    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    if gross_loss <= 0.0:
        return 999.0
    return round(gross_profit / gross_loss, 4)


def compute_max_dd(pnls: pd.Series) -> float:
    values = pnls.dropna().astype(float)
    if values.empty:
        return 0.0
    equity = values.cumsum()
    peak = equity.cummax()
    return round(float((equity - peak).min()), 4)


def share_diff(current: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    keys = sorted(set(current) | set(baseline))
    return {key: round(float(current.get(key, 0.0) - baseline.get(key, 0.0)), 4) for key in keys}


def max_abs_share_diff(current: dict[str, float], baseline: dict[str, float]) -> float:
    diffs = share_diff(current, baseline).values()
    return round(max((abs(value) for value in diffs), default=0.0), 4)


def fidelity_from_flags(full: bool, partial: bool = False) -> str:
    if full:
        return "FULL"
    if partial:
        return "PARTIAL"
    return "UNAVAILABLE"


def standardized_columns() -> list[str]:
    return [
        "line",
        "official_id",
        "session_date",
        "event_time_ny",
        "level",
        "session_bucket",
        "direction",
        "risk_pips",
        "risk_bucket",
        "pnl_r",
        "news_affected",
        "cost_delta_pips",
        "cost_delta_r",
    ]


def empty_standardized_frame(line_name: str) -> pd.DataFrame:
    return pd.DataFrame(columns=standardized_columns()).assign(line=line_name)


def load_historical_frame(line_name: str) -> pd.DataFrame:
    config = LINE_CONFIGS[line_name]
    path = config["historical_path"]
    if not path.exists():
        return empty_standardized_frame(line_name)

    raw = pd.read_csv(path)
    raw["line"] = line_name
    raw["level"] = raw["level"].astype(str).str.lower()
    raw["direction"] = raw["direction"].astype(str).str.lower()
    raw["session_bucket"] = raw["level"].map(session_from_level)
    raw["session_date"] = pd.to_datetime(raw["session_date"], errors="coerce")
    raw["event_time_ny"] = parse_ny_timestamps(raw["entry_time"])
    raw["risk_pips"] = pd.to_numeric(raw["risk_pips"], errors="coerce")
    raw["risk_bucket"] = raw["risk_pips"].apply(risk_bucket)
    raw["official_id"] = raw.index.map(lambda idx: f"{line_name}_HIST_{idx:05d}")
    raw["news_affected"] = raw.get("blocked_by_news", False).fillna(False).astype(bool) | raw["exit_reason"].astype(str).eq("news_fortress_kill")

    if line_name == "SCBI_M5_GLOBAL":
        raw["cost_delta_pips"] = 0.1
        raw["cost_delta_r"] = raw["cost_delta_pips"] / raw["risk_pips"]
        raw["pnl_r"] = pd.to_numeric(raw["pnl_r"], errors="coerce") - raw["cost_delta_r"]
    else:
        raw["cost_delta_pips"] = pd.to_numeric(raw["dynamic_extra_cost"], errors="coerce")
        raw["cost_delta_r"] = raw["cost_delta_pips"] / raw["risk_pips"]
        raw["pnl_r"] = pd.to_numeric(raw["pnl_r_dynamic"], errors="coerce")

    return raw[standardized_columns()].copy()


def _empty_forward_bundle(line_name: str, *, source_path: str) -> dict[str, Any]:
    return {
        "source_path": source_path,
        "source_hash": "",
        "source_rows": 0,
        "standardized_trades": empty_standardized_frame(line_name),
        "duplicate_official_ids": 0,
        "pipeline_notes": ["FORWARD_SOURCE_MISSING"],
        "forward_dimension_fidelity": {
            "A_SIGNAL_FREQUENCY_DRIFT": "UNAVAILABLE",
            "B_SESSION_COMPOSITION_DRIFT": "UNAVAILABLE",
            "C_DIRECTIONAL_DRIFT": "UNAVAILABLE",
            "D_LIQUIDITY_SOURCE_OR_TRIGGER_DRIFT": "UNAVAILABLE",
            "E_PERFORMANCE_DISTRIBUTION_DRIFT": "UNAVAILABLE",
            "F_BLOCKING_PROFILE_DRIFT": "UNAVAILABLE",
            "G_EXECUTION_PROFILE_DRIFT": "UNAVAILABLE",
        },
        "telemetry_snapshot": {},
    }


def load_forward_bundle_from_trace(line_name: str) -> dict[str, Any] | None:
    if not TRACE_CSV.exists():
        return None

    trace = load_trace_frame()
    if trace.empty:
        return None

    line_trace = trace[
        trace["source_line"].astype(str).eq(line_name)
        & trace["official_flag"].astype(str).str.upper().eq("TRUE")
    ].copy()
    if line_trace.empty:
        return None

    trade_trace = line_trace[
        line_trace["event_class"].astype(str).eq("TRADE_EVENT")
        & line_trace["event_phase"].astype(str).eq("EXIT")
    ].copy()
    if trade_trace.empty:
        return None

    duplicate_count = int(trade_trace["signal_or_event_id"].duplicated().sum())
    trade_trace = trade_trace.sort_values(["session_date", "event_time_ny", "trace_id"]).drop_duplicates("signal_or_event_id", keep="last")
    trade_trace["line"] = line_name
    trade_trace["official_id"] = trade_trace["signal_or_event_id"]
    trade_trace["session_date"] = pd.to_datetime(trade_trace["session_date"], errors="coerce")
    trade_trace["event_time_ny"] = parse_ny_timestamps(trade_trace["event_time_ny"])
    trade_trace["level"] = trade_trace["level"].astype(str).str.lower()
    trade_trace["session_bucket"] = trade_trace["level"].map(session_from_level)
    trade_trace["direction"] = trade_trace["direction"].astype(str).str.lower()
    trade_trace["risk_pips"] = pd.to_numeric(trade_trace["risk_pips"], errors="coerce")
    trade_trace["risk_bucket"] = trade_trace["risk_pips"].apply(risk_bucket)
    trade_trace["pnl_r"] = pd.to_numeric(trade_trace["pnl_r"], errors="coerce")
    trade_trace["news_affected"] = trade_trace["news_affected"].astype(str).str.upper().eq("TRUE")
    trade_trace["cost_delta_pips"] = pd.to_numeric(trade_trace["cost_proxy_pips"], errors="coerce")
    trade_trace["cost_delta_r"] = pd.to_numeric(trade_trace["cost_proxy_r"], errors="coerce")
    standardized = trade_trace[standardized_columns()].copy()

    telemetry_snapshot = telemetry_snapshot_by_line(trace).get(line_name, {})
    notes = [f"TRACE_SOURCE={TRACE_CSV.relative_to(ROOT)}"]
    if duplicate_count:
        notes.append(f"DUPLICATE_OFFICIAL_IDS_DROPPED={duplicate_count}")

    forward_dimension_fidelity = {
        "A_SIGNAL_FREQUENCY_DRIFT": fidelity_from_flags(not standardized.empty),
        "B_SESSION_COMPOSITION_DRIFT": fidelity_from_flags(standardized["session_bucket"].notna().all()),
        "C_DIRECTIONAL_DRIFT": fidelity_from_flags(standardized["direction"].notna().all()),
        "D_LIQUIDITY_SOURCE_OR_TRIGGER_DRIFT": fidelity_from_flags(standardized["level"].notna().all()),
        "E_PERFORMANCE_DISTRIBUTION_DRIFT": fidelity_from_flags(standardized["pnl_r"].notna().all()),
        "F_BLOCKING_PROFILE_DRIFT": str(telemetry_snapshot.get("blocking_fidelity", "UNAVAILABLE")),
        "G_EXECUTION_PROFILE_DRIFT": str(telemetry_snapshot.get("execution_fidelity", "UNAVAILABLE")),
    }

    return {
        "source_path": str(TRACE_CSV.relative_to(ROOT)),
        "source_hash": sha256_file(TRACE_CSV),
        "source_rows": int(len(line_trace)),
        "standardized_trades": standardized,
        "duplicate_official_ids": duplicate_count,
        "pipeline_notes": notes,
        "forward_dimension_fidelity": forward_dimension_fidelity,
        "telemetry_snapshot": telemetry_snapshot,
    }


def load_forward_bundle_from_legacy(line_name: str) -> dict[str, Any]:
    config = LINE_CONFIGS[line_name]
    path = config["forward_path"]
    if not path.exists():
        return _empty_forward_bundle(line_name, source_path=str(path.relative_to(ROOT)))

    raw = pd.read_csv(path)
    notes: list[str] = []

    if line_name == "SCBI_M5_GLOBAL":
        official = raw[(raw["event_type"] == "PAPER_EXIT") & raw["pnl_r"].notna()].copy()
        duplicate_count = int(official["signal_id"].duplicated().sum())
        official = official.sort_values("event_timestamp").drop_duplicates("signal_id", keep="last")
        official["line"] = line_name
        official["official_id"] = official["signal_id"]
        official["session_date"] = pd.to_datetime(official["session_date"], errors="coerce")
        official["event_time_ny"] = parse_ny_timestamps(official["exit_time"].fillna(official["entry_time"]))
        official["level"] = official["sweep_level"].astype(str).str.lower()
        official["session_bucket"] = official["level"].map(session_from_level)
        official["direction"] = official["direction"].astype(str).str.lower()
        official["risk_pips"] = pd.to_numeric(official["risk_pips"], errors="coerce")
        official["risk_bucket"] = official["risk_pips"].apply(risk_bucket)
        official["pnl_r"] = pd.to_numeric(official["pnl_r"], errors="coerce")
        official["news_affected"] = official["exit_type"].astype(str).eq("news_fortress_kill") | official["news_check_status"].astype(str).ne("CLEAR")
        official["cost_delta_pips"] = pd.to_numeric(official["applied_spread_pips"], errors="coerce")
        official["cost_delta_r"] = official["cost_delta_pips"] / official["risk_pips"]
        standardized = official[standardized_columns()].copy()
        execution_partial = True
        blocking_partial = True
    else:
        official = raw[raw["event_id"].astype(str).str.startswith("CORE_") & raw["pnl_r"].notna()].copy()
        duplicate_count = int(official["event_id"].duplicated().sum())
        official = official.drop_duplicates("event_id", keep="last")
        official["line"] = line_name
        official["official_id"] = official["event_id"]
        official["session_date"] = pd.to_datetime(official["timestamp_ny"].astype(str).str.slice(0, 10), errors="coerce")
        official["event_time_ny"] = parse_ny_timestamps(official["exit_time"].fillna(official["timestamp_ny"]))
        official["level"] = official["level"].astype(str).str.lower()
        official["session_bucket"] = official["level"].map(session_from_level)
        official["direction"] = official["direction"].astype(str).str.lower()
        official["risk_pips"] = pd.to_numeric(official["risk_pips"], errors="coerce")
        official["risk_bucket"] = official["risk_pips"].apply(risk_bucket)
        official["pnl_r"] = pd.to_numeric(official["pnl_r"], errors="coerce")
        official["news_affected"] = official["news_blocked"].fillna(False).astype(bool)
        official["cost_delta_pips"] = np.nan
        official["cost_delta_r"] = np.nan
        standardized = official[standardized_columns()].copy()
        execution_partial = True
        blocking_partial = True
        notes.append("CORE_FORWARD_HAS_NO_PER_FILL_COST_FIELDS")

    if duplicate_count:
        notes.append(f"DUPLICATE_OFFICIAL_IDS_DROPPED={duplicate_count}")

    forward_dimension_fidelity = {
        "A_SIGNAL_FREQUENCY_DRIFT": fidelity_from_flags(not standardized.empty),
        "B_SESSION_COMPOSITION_DRIFT": fidelity_from_flags(standardized["session_bucket"].notna().all()),
        "C_DIRECTIONAL_DRIFT": fidelity_from_flags(standardized["direction"].notna().all()),
        "D_LIQUIDITY_SOURCE_OR_TRIGGER_DRIFT": fidelity_from_flags(standardized["level"].notna().all()),
        "E_PERFORMANCE_DISTRIBUTION_DRIFT": fidelity_from_flags(standardized["pnl_r"].notna().all()),
        "F_BLOCKING_PROFILE_DRIFT": fidelity_from_flags(False, partial=blocking_partial),
        "G_EXECUTION_PROFILE_DRIFT": fidelity_from_flags(False, partial=execution_partial),
    }

    return {
        "source_path": str(path.relative_to(ROOT)),
        "source_hash": sha256_file(path),
        "source_rows": int(len(raw)),
        "standardized_trades": standardized,
        "duplicate_official_ids": duplicate_count,
        "pipeline_notes": notes,
        "forward_dimension_fidelity": forward_dimension_fidelity,
        "telemetry_snapshot": {},
    }


def load_forward_bundle(line_name: str) -> dict[str, Any]:
    bundle_from_trace = load_forward_bundle_from_trace(line_name)
    if bundle_from_trace is not None:
        return bundle_from_trace
    return load_forward_bundle_from_legacy(line_name)


def historical_dimension_fidelity(line_name: str) -> dict[str, str]:
    return {
        "A_SIGNAL_FREQUENCY_DRIFT": "FULL",
        "B_SESSION_COMPOSITION_DRIFT": "FULL",
        "C_DIRECTIONAL_DRIFT": "FULL",
        "D_LIQUIDITY_SOURCE_OR_TRIGGER_DRIFT": "FULL",
        "E_PERFORMANCE_DISTRIBUTION_DRIFT": "FULL",
        "F_BLOCKING_PROFILE_DRIFT": "PARTIAL",
        "G_EXECUTION_PROFILE_DRIFT": "PARTIAL" if line_name == "SCBI_M5_GLOBAL" else "FULL",
    }


def compute_metrics(df: pd.DataFrame, level_order: tuple[str, ...]) -> dict[str, Any]:
    ordered = df.sort_values(["session_date", "event_time_ny", "official_id"]).copy()
    pnls = ordered["pnl_r"].dropna().astype(float)
    sessions = ordered["session_bucket"] if "session_bucket" in ordered.columns else pd.Series(dtype="object")
    weekly_counts = pd.Series(dtype="float64")
    if "session_date" in ordered.columns and not ordered["session_date"].dropna().empty:
        weekly_counts = ordered["session_date"].dropna().dt.to_period("W").value_counts().sort_index()

    risk_pips = ordered["risk_pips"].dropna().astype(float)
    cost_delta_pips = pd.to_numeric(ordered["cost_delta_pips"], errors="coerce").dropna()
    cost_delta_r = pd.to_numeric(ordered["cost_delta_r"], errors="coerce").dropna()
    news_affected = ordered["news_affected"].fillna(False).astype(bool) if "news_affected" in ordered.columns else pd.Series(dtype="bool")

    performance = {
        "n": int(len(pnls)),
        "expectancy": round(float(pnls.mean()), 4) if not pnls.empty else 0.0,
        "std_pnl": round(float(pnls.std(ddof=0)), 4) if len(pnls) > 1 else 0.0,
        "pf": compute_pf(pnls),
        "win_rate": round(float((pnls > 0.0).mean()), 4) if not pnls.empty else 0.0,
        "max_dd": compute_max_dd(pnls),
        "q10": round(float(pnls.quantile(0.10)), 4) if not pnls.empty else 0.0,
        "q25": round(float(pnls.quantile(0.25)), 4) if not pnls.empty else 0.0,
        "q50": round(float(pnls.quantile(0.50)), 4) if not pnls.empty else 0.0,
        "q75": round(float(pnls.quantile(0.75)), 4) if not pnls.empty else 0.0,
        "q90": round(float(pnls.quantile(0.90)), 4) if not pnls.empty else 0.0,
    }

    frequency = {
        "avg_weekly": round(float(weekly_counts.mean()), 4) if not weekly_counts.empty else 0.0,
        "std_weekly": round(float(weekly_counts.std(ddof=0)), 4) if len(weekly_counts) > 1 else 0.0,
        "weeks_observed": int(len(weekly_counts)),
        "min_weekly": int(weekly_counts.min()) if not weekly_counts.empty else 0,
        "max_weekly": int(weekly_counts.max()) if not weekly_counts.empty else 0,
    }

    return {
        "frequency": frequency,
        "session_composition": normalize_distribution(sessions, SESSION_ORDER),
        "directional": normalize_distribution(ordered["direction"], ("long", "short", "unknown")),
        "liquidity_source": normalize_distribution(ordered["level"], level_order),
        "performance_distribution": performance,
        "blocking_profile": {
            "news_affected_rate": round(float(news_affected.mean()), 4) if not news_affected.empty else 0.0,
        },
        "execution_profile": {
            "avg_risk_pips": round(float(risk_pips.mean()), 4) if not risk_pips.empty else 0.0,
            "median_risk_pips": round(float(risk_pips.median()), 4) if not risk_pips.empty else 0.0,
            "risk_bucket_mix": normalize_distribution(ordered["risk_bucket"], RISK_BUCKET_ORDER),
            "avg_cost_delta_pips": round(float(cost_delta_pips.mean()), 4) if not cost_delta_pips.empty else None,
            "avg_cost_delta_r": round(float(cost_delta_r.mean()), 4) if not cost_delta_r.empty else None,
        },
        "last_activity_ny": str(ordered["event_time_ny"].max()) if not ordered.empty else "",
    }


def window_summary(df: pd.DataFrame, window: int) -> dict[str, Any]:
    ordered = df.sort_values(["session_date", "event_time_ny", "official_id"]).reset_index(drop=True)
    if len(ordered) < window:
        return {"window": window, "count": 0}

    rows = []
    for start in range(0, len(ordered) - window + 1):
        chunk = ordered.iloc[start : start + window].copy()
        metrics = compute_metrics(chunk, GLOBAL_LEVEL_ORDER if chunk["line"].iloc[0] == "SCBI_M5_GLOBAL" else CORE_LEVEL_ORDER)
        rows.append(
            {
                "expectancy": metrics["performance_distribution"]["expectancy"],
                "pf": metrics["performance_distribution"]["pf"],
                "max_dd": metrics["performance_distribution"]["max_dd"],
                "long_share": metrics["directional"]["long"],
            }
        )

    frame = pd.DataFrame(rows)
    return {
        "window": window,
        "count": int(len(frame)),
        "expectancy_p05": round(float(frame["expectancy"].quantile(0.05)), 4),
        "expectancy_p50": round(float(frame["expectancy"].quantile(0.50)), 4),
        "expectancy_p95": round(float(frame["expectancy"].quantile(0.95)), 4),
        "pf_p05": round(float(frame["pf"].quantile(0.05)), 4),
        "pf_p50": round(float(frame["pf"].quantile(0.50)), 4),
        "pf_p95": round(float(frame["pf"].quantile(0.95)), 4),
        "max_dd_p05": round(float(frame["max_dd"].quantile(0.05)), 4),
        "long_share_p05": round(float(frame["long_share"].quantile(0.05)), 4),
        "long_share_p95": round(float(frame["long_share"].quantile(0.95)), 4),
    }


def build_baselines() -> dict[str, Any]:
    lines: dict[str, Any] = {}
    for line_name, config in LINE_CONFIGS.items():
        historical = load_historical_frame(line_name)
        metrics = compute_metrics(historical, config["level_order"])
        line_payload = {
            "line": line_name,
            "source": {
                "historical_path": str(config["historical_path"].relative_to(ROOT)),
                "historical_hash": sha256_file(config["historical_path"]),
                "historical_rows": int(len(historical)),
                "timezone_model": "AMERICA_NEW_YORK_DSTAWARE_MIXED_OFFSETS",
                "historical_cost_model": config["historical_cost_model"],
                "historical_cost_note": config["historical_cost_note"],
                "historical_time_note": config["historical_time_note"],
                "forward_cost_note": config["forward_cost_note"],
            },
            "historical_dimension_fidelity": historical_dimension_fidelity(line_name),
            "dimensions": metrics,
            "validation_windows": {
                str(window): window_summary(historical, window) for window in VALIDATION_WINDOWS
            },
        }
        lines[line_name] = line_payload

    return {
        "generated_at_utc": now_utc_iso(),
        "baseline_version": BASELINE_VERSION,
        "lines": lines,
    }


def audit_previous_baseline(candidate_baseline: dict[str, Any]) -> dict[str, Any]:
    reasons = [
        "PREVIOUS_BASELINE_WITHOUT_VERSION_METADATA",
        "PREVIOUS_BASELINE_MISSING_STRUCTURED_LINE_NAMESPACE",
        "SCBI_M5_GLOBAL_COST_MODEL_NOT_HARDENED",
        "SCBI_CORE_COST_MODEL_NOT_HARDENED",
        "SCBI_M5_GLOBAL_MISSING_SESSION_COMPOSITION",
        "SCBI_M5_GLOBAL_MISSING_DIRECTIONAL",
        "SCBI_M5_GLOBAL_MISSING_LIQUIDITY_SOURCE",
        "SCBI_M5_GLOBAL_MISSING_BLOCKING_PROFILE",
        "SCBI_M5_GLOBAL_MISSING_EXECUTION_PROFILE",
        "SCBI_CORE_MISSING_SESSION_COMPOSITION",
        "SCBI_CORE_MISSING_DIRECTIONAL",
        "SCBI_CORE_MISSING_LIQUIDITY_SOURCE",
        "SCBI_CORE_MISSING_BLOCKING_PROFILE",
        "SCBI_CORE_MISSING_EXECUTION_PROFILE",
    ]

    global_hist = pd.read_csv(LINE_CONFIGS["SCBI_M5_GLOBAL"]["historical_path"])
    core_hist = pd.read_csv(LINE_CONFIGS["SCBI_CORE"]["historical_path"])
    legacy_global_exp = round(float(pd.to_numeric(global_hist["pnl_r"], errors="coerce").mean()), 4)
    legacy_core_exp = round(float(pd.to_numeric(core_hist["pnl_r"], errors="coerce").mean()), 4)
    new_global_exp = candidate_baseline["lines"]["SCBI_M5_GLOBAL"]["dimensions"]["performance_distribution"]["expectancy"]
    new_core_exp = candidate_baseline["lines"]["SCBI_CORE"]["dimensions"]["performance_distribution"]["expectancy"]
    reasons.append(f"SCBI_M5_GLOBAL_EXPECTANCY_SHIFT_{round(legacy_global_exp - new_global_exp, 4)}")
    reasons.append(f"SCBI_CORE_EXPECTANCY_SHIFT_{round(legacy_core_exp - new_core_exp, 4)}")

    return {
        "status": "REBASELINE_REQUIRED",
        "reasons": sorted(set(reasons)),
        "previous_baseline_present": True,
    }


def dimension_result(status: str, details: dict[str, Any]) -> dict[str, Any]:
    payload = {"status": status}
    payload.update(details)
    return payload


def compare_line_to_baseline(line_name: str, baseline_entry: dict[str, Any], forward_bundle: dict[str, Any]) -> dict[str, Any]:
    config = LINE_CONFIGS[line_name]
    trades = forward_bundle["standardized_trades"].copy()
    metrics = compute_metrics(trades, config["level_order"])
    baseline_metrics = baseline_entry["dimensions"]
    n = metrics["performance_distribution"]["n"]
    weeks = metrics["frequency"]["weeks_observed"]
    dimension_checks: dict[str, Any] = {}
    drift_flags: list[str] = []
    coverage_gaps: list[str] = list(forward_bundle["pipeline_notes"])

    critical_pipeline_issue = bool(forward_bundle["duplicate_official_ids"])
    if critical_pipeline_issue:
        coverage_gaps.append("DUPLICATE_FORWARD_IDS_REQUIRE_PIPELINE_REVIEW")

    if n == 0:
        for dimension_key in DIMENSION_NAMES:
            dimension_checks[dimension_key] = dimension_result("NOT_COMPARABLE", {"reason": "NO_FORWARD_TRADES"})
        verdict = "NOT_COMPARABLE_YET"
        drift_r = None
    else:
        if n < MIN_COMPARABLE_N:
            coverage_gaps.append(f"OFFICIAL_SAMPLE_BELOW_{MIN_COMPARABLE_N}")

        baseline_frequency = baseline_metrics["frequency"]["avg_weekly"]
        current_frequency = metrics["frequency"]["avg_weekly"]
        if n < MIN_COMPARABLE_N or weeks < MIN_FREQUENCY_WEEKS:
            dimension_checks["A_SIGNAL_FREQUENCY_DRIFT"] = dimension_result(
                "NOT_COMPARABLE",
                {
                    "weeks_observed": weeks,
                    "current_avg_weekly": current_frequency,
                    "baseline_avg_weekly": baseline_frequency,
                },
            )
        else:
            rel_delta = 0.0 if baseline_frequency == 0 else (current_frequency - baseline_frequency) / baseline_frequency
            status = "PASS" if abs(rel_delta) <= FREQUENCY_REL_THRESHOLD else "DRIFT"
            if status == "DRIFT":
                drift_flags.append(DIMENSION_NAMES["A_SIGNAL_FREQUENCY_DRIFT"])
            dimension_checks["A_SIGNAL_FREQUENCY_DRIFT"] = dimension_result(
                status,
                {
                    "weeks_observed": weeks,
                    "current_avg_weekly": current_frequency,
                    "baseline_avg_weekly": baseline_frequency,
                    "relative_delta": round(float(rel_delta), 4),
                    "threshold": FREQUENCY_REL_THRESHOLD,
                },
            )

        session_delta = max_abs_share_diff(metrics["session_composition"], baseline_metrics["session_composition"])
        session_status = "NOT_COMPARABLE" if n < MIN_COMPARABLE_N else ("PASS" if session_delta <= SESSION_SHARE_THRESHOLD else "DRIFT")
        if session_status == "DRIFT":
            drift_flags.append(DIMENSION_NAMES["B_SESSION_COMPOSITION_DRIFT"])
        dimension_checks["B_SESSION_COMPOSITION_DRIFT"] = dimension_result(
            session_status,
            {
                "current": metrics["session_composition"],
                "baseline": baseline_metrics["session_composition"],
                "max_abs_delta": session_delta,
                "threshold": SESSION_SHARE_THRESHOLD,
            },
        )

        direction_delta = max_abs_share_diff(metrics["directional"], baseline_metrics["directional"])
        direction_status = "NOT_COMPARABLE" if n < MIN_COMPARABLE_N else ("PASS" if direction_delta <= DIRECTION_SHARE_THRESHOLD else "DRIFT")
        if direction_status == "DRIFT":
            drift_flags.append(DIMENSION_NAMES["C_DIRECTIONAL_DRIFT"])
        dimension_checks["C_DIRECTIONAL_DRIFT"] = dimension_result(
            direction_status,
            {
                "current": metrics["directional"],
                "baseline": baseline_metrics["directional"],
                "max_abs_delta": direction_delta,
                "threshold": DIRECTION_SHARE_THRESHOLD,
            },
        )

        liquidity_delta = max_abs_share_diff(metrics["liquidity_source"], baseline_metrics["liquidity_source"])
        liquidity_status = "NOT_COMPARABLE" if n < MIN_COMPARABLE_N else ("PASS" if liquidity_delta <= LIQUIDITY_SHARE_THRESHOLD else "DRIFT")
        if liquidity_status == "DRIFT":
            drift_flags.append(DIMENSION_NAMES["D_LIQUIDITY_SOURCE_OR_TRIGGER_DRIFT"])
        dimension_checks["D_LIQUIDITY_SOURCE_OR_TRIGGER_DRIFT"] = dimension_result(
            liquidity_status,
            {
                "current": metrics["liquidity_source"],
                "baseline": baseline_metrics["liquidity_source"],
                "max_abs_delta": liquidity_delta,
                "threshold": LIQUIDITY_SHARE_THRESHOLD,
            },
        )

        performance = metrics["performance_distribution"]
        baseline_performance = baseline_metrics["performance_distribution"]
        drift_r = round(float(performance["expectancy"] - baseline_performance["expectancy"]), 4)
        max_dd_limit = round(float(baseline_performance["max_dd"] - MAX_DD_WORSENING_R), 4)
        if n < MIN_COMPARABLE_N:
            performance_status = "NOT_COMPARABLE"
        elif (
            performance["pf"] <= PF_STRUCTURAL_THRESHOLD
            or drift_r <= EXPECTANCY_STRUCTURAL_DELTA
            or performance["max_dd"] <= max_dd_limit
        ):
            performance_status = "STRUCTURAL"
            drift_flags.append(DIMENSION_NAMES["E_PERFORMANCE_DISTRIBUTION_DRIFT"])
        elif performance["pf"] <= PF_WARNING_THRESHOLD or drift_r <= EXPECTANCY_WARNING_DELTA:
            performance_status = "WARNING"
            drift_flags.append(DIMENSION_NAMES["E_PERFORMANCE_DISTRIBUTION_DRIFT"])
        else:
            performance_status = "PASS"
        dimension_checks["E_PERFORMANCE_DISTRIBUTION_DRIFT"] = dimension_result(
            performance_status,
            {
                "current": performance,
                "baseline": baseline_performance,
                "drift_r": drift_r,
                "pf_warning_threshold": PF_WARNING_THRESHOLD,
                "pf_structural_threshold": PF_STRUCTURAL_THRESHOLD,
                "expectancy_warning_delta": EXPECTANCY_WARNING_DELTA,
                "expectancy_structural_delta": EXPECTANCY_STRUCTURAL_DELTA,
                "max_dd_structural_limit": max_dd_limit,
            },
        )

        forward_fidelity = forward_bundle["forward_dimension_fidelity"]["F_BLOCKING_PROFILE_DRIFT"]
        blocking_delta = round(
            float(metrics["blocking_profile"]["news_affected_rate"] - baseline_metrics["blocking_profile"]["news_affected_rate"]),
            4,
        )
        if n < MIN_COMPARABLE_N or forward_fidelity == "UNAVAILABLE":
            blocking_status = "NOT_COMPARABLE"
        elif abs(blocking_delta) > BLOCKING_SHARE_THRESHOLD:
            blocking_status = "DRIFT"
            drift_flags.append(DIMENSION_NAMES["F_BLOCKING_PROFILE_DRIFT"])
        else:
            blocking_status = "PASS"
        dimension_checks["F_BLOCKING_PROFILE_DRIFT"] = dimension_result(
            blocking_status,
            {
                "fidelity": forward_fidelity,
                "current": metrics["blocking_profile"],
                "baseline": baseline_metrics["blocking_profile"],
                "delta": blocking_delta,
                "threshold": BLOCKING_SHARE_THRESHOLD,
            },
        )

        forward_execution_fidelity = forward_bundle["forward_dimension_fidelity"]["G_EXECUTION_PROFILE_DRIFT"]
        bucket_delta = max_abs_share_diff(
            metrics["execution_profile"]["risk_bucket_mix"],
            baseline_metrics["execution_profile"]["risk_bucket_mix"],
        )
        current_avg_cost_delta = metrics["execution_profile"]["avg_cost_delta_r"]
        baseline_avg_cost_delta = baseline_metrics["execution_profile"]["avg_cost_delta_r"]
        comparable_cost_delta = current_avg_cost_delta is not None and baseline_avg_cost_delta is not None
        cost_delta_change = (
            round(float(current_avg_cost_delta - baseline_avg_cost_delta), 4)
            if comparable_cost_delta
            else None
        )
        if n < MIN_COMPARABLE_N or forward_execution_fidelity == "UNAVAILABLE":
            execution_status = "NOT_COMPARABLE"
        elif bucket_delta > EXECUTION_BUCKET_THRESHOLD or (
            cost_delta_change is not None and cost_delta_change > EXECUTION_COST_DELTA_R_THRESHOLD
        ):
            execution_status = "DRIFT"
            drift_flags.append(DIMENSION_NAMES["G_EXECUTION_PROFILE_DRIFT"])
        else:
            execution_status = "PASS"
        dimension_checks["G_EXECUTION_PROFILE_DRIFT"] = dimension_result(
            execution_status,
            {
                "fidelity": forward_execution_fidelity,
                "current": metrics["execution_profile"],
                "baseline": baseline_metrics["execution_profile"],
                "risk_bucket_max_abs_delta": bucket_delta,
                "risk_bucket_threshold": EXECUTION_BUCKET_THRESHOLD,
                "avg_cost_delta_r_threshold": EXECUTION_COST_DELTA_R_THRESHOLD,
                "avg_cost_delta_r_change": cost_delta_change,
            },
        )

        severe_dimension_count = sum(1 for item in dimension_checks.values() if item["status"] == "STRUCTURAL")
        warning_dimension_count = sum(1 for item in dimension_checks.values() if item["status"] in {"DRIFT", "WARNING"})
        catastrophic_performance = performance["pf"] <= PF_CATASTROPHIC_THRESHOLD and drift_r <= EXPECTANCY_CATASTROPHIC_DELTA

        if critical_pipeline_issue:
            verdict = "DATA_OR_PIPELINE_DRIFT"
        elif n < MIN_COMPARABLE_N:
            verdict = "NOT_COMPARABLE_YET"
        elif n >= 40 and (performance_status == "STRUCTURAL" or severe_dimension_count >= 1 or warning_dimension_count >= 2):
            verdict = "STRUCTURAL_DRIFT"
        elif n >= 20 and catastrophic_performance:
            verdict = "STRUCTURAL_DRIFT"
        elif performance_status in {"STRUCTURAL", "WARNING"} or warning_dimension_count >= 1:
            verdict = "TOLERABLE_VARIATION"
        else:
            verdict = "NO_DRIFT"

    comparable_dimensions = {
        key: value["status"] not in {"NOT_COMPARABLE"} for key, value in dimension_checks.items()
    }
    return {
        "line": line_name,
        "verdict": verdict,
        "official_n": n,
        "baseline_version": BASELINE_VERSION,
        "source_path": forward_bundle["source_path"],
        "source_hash": forward_bundle["source_hash"],
        "source_rows": forward_bundle["source_rows"],
        "drift_r": drift_r,
        "forward_dimension_fidelity": forward_bundle["forward_dimension_fidelity"],
        "comparable_dimensions": comparable_dimensions,
        "dimension_results": dimension_checks,
        "drift_flags": sorted(set(drift_flags)),
        "coverage_gaps": sorted(set(coverage_gaps)),
        "current_metrics": metrics,
        "baseline_snapshot": baseline_entry["dimensions"],
    }


def monitor_live_forward(baseline_payload: dict[str, Any], validation_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    lines: dict[str, Any] = {}
    for line_name, line_baseline in baseline_payload["lines"].items():
        lines[line_name] = compare_line_to_baseline(line_name, line_baseline, load_forward_bundle(line_name))

    tribunal_mode = determine_tribunal_mode(validation_payload, lines)
    return {
        "generated_at_utc": now_utc_iso(),
        "baseline_version": baseline_payload["baseline_version"],
        "monitor_validation_verdict": (
            validation_payload["overall"]["verdict"] if validation_payload else "UNKNOWN"
        ),
        "tribunal_integration_mode": tribunal_mode,
        "lines": lines,
    }


def pseudo_forward_bundle(line_name: str, window: pd.DataFrame) -> dict[str, Any]:
    return {
        "source_path": f"pseudo_forward::{line_name}",
        "source_hash": "",
        "source_rows": int(len(window)),
        "standardized_trades": window.copy(),
        "duplicate_official_ids": 0,
        "pipeline_notes": [],
        "forward_dimension_fidelity": {
            "A_SIGNAL_FREQUENCY_DRIFT": "FULL",
            "B_SESSION_COMPOSITION_DRIFT": "FULL",
            "C_DIRECTIONAL_DRIFT": "FULL",
            "D_LIQUIDITY_SOURCE_OR_TRIGGER_DRIFT": "FULL",
            "E_PERFORMANCE_DISTRIBUTION_DRIFT": "FULL",
            "F_BLOCKING_PROFILE_DRIFT": "PARTIAL",
            "G_EXECUTION_PROFILE_DRIFT": "FULL",
        },
    }


def perturb_window(line_name: str, window: pd.DataFrame, perturbation: str) -> pd.DataFrame:
    result = window.copy()
    if result.empty:
        return result

    if perturbation == "frequency_drop":
        start = result["session_date"].min()
        spaced = [start + pd.Timedelta(days=idx * 7) for idx in range(len(result))]
        result["session_date"] = spaced
        result["event_time_ny"] = pd.to_datetime(spaced).tz_localize(NY_TZ) + pd.Timedelta(hours=8)
    elif perturbation == "directional_bias":
        result["direction"] = "long"
    elif perturbation == "liquidity_collapse":
        dominant = "london_l"
        result["level"] = dominant
        result["session_bucket"] = session_from_level(dominant)
    elif perturbation == "performance_crush":
        result["pnl_r"] = pd.to_numeric(result["pnl_r"], errors="coerce") - 0.75
    elif perturbation == "blocking_spike":
        result["news_affected"] = True
    elif perturbation == "execution_shock":
        risk = pd.to_numeric(result["risk_pips"], errors="coerce").clip(lower=0.5)
        extra_r = 0.25 / risk
        result["pnl_r"] = pd.to_numeric(result["pnl_r"], errors="coerce") - extra_r
        base_cost_r = pd.to_numeric(result["cost_delta_r"], errors="coerce").fillna(0.0)
        base_cost_pips = pd.to_numeric(result["cost_delta_pips"], errors="coerce").fillna(0.0)
        result["cost_delta_r"] = base_cost_r + extra_r
        result["cost_delta_pips"] = base_cost_pips + 0.25

    result["risk_bucket"] = result["risk_pips"].apply(risk_bucket)
    return result


def evenly_spaced_windows(df: pd.DataFrame, window: int, count: int) -> list[pd.DataFrame]:
    ordered = df.sort_values(["session_date", "event_time_ny", "official_id"]).reset_index(drop=True)
    total = len(ordered) - window + 1
    if total <= 0:
        return []
    if total <= count:
        starts = list(range(total))
    else:
        starts = sorted({int(round(value)) for value in np.linspace(0, total - 1, num=count)})
    return [ordered.iloc[start : start + window].copy() for start in starts]


def validate_monitor(baseline_payload: dict[str, Any]) -> dict[str, Any]:
    line_results: dict[str, Any] = {}
    severe_fp_rates: list[float] = []
    sensitivity_rates: list[float] = []

    for line_name in LINE_CONFIGS:
        history = load_historical_frame(line_name)
        baseline_entry = baseline_payload["lines"][line_name]
        false_positive_results: dict[str, Any] = {}
        for window in VALIDATION_WINDOWS:
            windows = evenly_spaced_windows(history, window, max(1, len(history) - window + 1))
            severe = 0
            warnings = 0
            for chunk in windows:
                verdict = compare_line_to_baseline(line_name, baseline_entry, pseudo_forward_bundle(line_name, chunk))["verdict"]
                if verdict in {"STRUCTURAL_DRIFT", "DATA_OR_PIPELINE_DRIFT"}:
                    severe += 1
                elif verdict == "TOLERABLE_VARIATION":
                    warnings += 1
            severe_rate = round(float(severe / len(windows)), 4) if windows else 0.0
            warning_rate = round(float(warnings / len(windows)), 4) if windows else 0.0
            severe_fp_rates.append(severe_rate)
            false_positive_results[str(window)] = {
                "windows": len(windows),
                "severe_alert_rate": severe_rate,
                "warning_rate": warning_rate,
                "status": "PASSED" if severe_rate <= 0.10 else "FAILED",
            }

        sensitivity_results: dict[str, Any] = {}
        for perturbation in (
            "frequency_drop",
            "directional_bias",
            "liquidity_collapse",
            "performance_crush",
            "blocking_spike",
            "execution_shock",
        ):
            windows = evenly_spaced_windows(history, 20, SENSITIVITY_SAMPLE_WINDOWS)
            detected = 0
            total = 0
            for chunk in windows:
                perturbed = perturb_window(line_name, chunk, perturbation)
                result = compare_line_to_baseline(line_name, baseline_entry, pseudo_forward_bundle(line_name, perturbed))
                total += 1
                if result["verdict"] in {"TOLERABLE_VARIATION", "STRUCTURAL_DRIFT", "DATA_OR_PIPELINE_DRIFT"}:
                    detected += 1
            rate = round(float(detected / total), 4) if total else 0.0
            sensitivity_rates.append(rate)
            sensitivity_results[perturbation] = {
                "detected": detected,
                "total": total,
                "detection_rate": rate,
                "status": "PASSED" if rate >= 0.80 else "FAILED",
            }

        line_results[line_name] = {
            "false_positive": false_positive_results,
            "sensitivity": sensitivity_results,
        }

    overall_false_positive = max(severe_fp_rates, default=0.0)
    overall_sensitivity = min(sensitivity_rates, default=1.0)
    overall_verdict = (
        "DRIFT_MONITOR_STILL_VALID"
        if overall_false_positive <= 0.10 and overall_sensitivity >= 0.80
        else "DRIFT_MONITOR_NEEDS_RECALIBRATION"
    )
    return {
        "generated_at_utc": now_utc_iso(),
        "baseline_version": baseline_payload["baseline_version"],
        "lines": line_results,
        "overall": {
            "max_severe_false_positive_rate": round(float(overall_false_positive), 4),
            "min_sensitivity_detection_rate": round(float(overall_sensitivity), 4),
            "verdict": overall_verdict,
        },
    }


def determine_tribunal_mode(
    validation_payload: dict[str, Any] | None,
    monitor_lines: dict[str, Any],
) -> str:
    if validation_payload is None or validation_payload["overall"]["verdict"] != "DRIFT_MONITOR_STILL_VALID":
        return "TRIBUNAL_MONITOR_ONLY"

    for report_line in monitor_lines.values():
        forward_fidelity = report_line["forward_dimension_fidelity"]
        if forward_fidelity["G_EXECUTION_PROFILE_DRIFT"] != "FULL":
            return "TRIBUNAL_MONITOR_ONLY"
        if forward_fidelity["F_BLOCKING_PROFILE_DRIFT"] == "UNAVAILABLE":
            return "TRIBUNAL_MONITOR_ONLY"

    return "TRIBUNAL_SAFE_TO_USE"


def final_conclusion(status_payload: dict[str, Any]) -> str:
    if (
        status_payload["taxonomy"]["rebaseline_status"] == "REBASELINE_CONFIRMED"
        and status_payload["taxonomy"]["monitor_status"] == "DRIFT_MONITOR_STILL_VALID"
        and status_payload["taxonomy"]["tribunal_status"] in {"TRIBUNAL_SAFE_TO_USE", "TRIBUNAL_MONITOR_ONLY"}
    ):
        return "POST_HARDENING_DRIFT_STACK_CONFIRMED"
    return "POST_HARDENING_DRIFT_STACK_NEEDS_MORE_WORK"
