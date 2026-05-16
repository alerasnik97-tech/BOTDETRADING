from __future__ import annotations

import json
import shutil
import sys
import traceback
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from research_lab.config import (
    DEFAULT_HIGH_PRECISION_PREPARED_DIR,
    EngineConfig,
    INITIAL_CAPITAL,
    NY_TZ,
    canonical_news_config,
    with_execution_mode,
)
from research_lab.data_loader import (
    fx_market_mask,
    load_high_precision_package,
    prepare_common_frame,
    resample_ohlcv_to_timeframe,
    validate_price_frame,
)
from research_lab.engine import entry_open_index, run_backtest
from research_lab.news_filter import build_entry_block, require_operational_news
from research_lab.report import build_period_stats, build_summary, export_strategy_bundle, summarize_result
import research_lab.strategies.eurusd_ltf_objective_entry_replacement_ecb as strategy_module


CANONICAL_ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo").resolve()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAIR = "EURUSD"
TIMEFRAME = "M3"
SWEEP_TIMEFRAME = "H1"
PIP_SIZE = 0.0001
MIN_SWEEP_PIPS = 0.5
STOP_BUFFER_PIPS = 1.0
TARGET_RR = 1.5
RESULTS_ROOT = PROJECT_ROOT / "results" / "eurusd_ltf_objective_entry_replacement_ecb_autopilot"
STAGE2_RESULTS_DIR = RESULTS_ROOT / "stage2"
FULL_RESULTS_DIR = RESULTS_ROOT / "full_campaign"
PRECHECK_AUDIT_PATH = RESULTS_ROOT / "precheck_audit.json"
CHECKPOINTS_DIR = PROJECT_ROOT / "ecb_stage2_checkpoints"
FAILURE_REPORTS_DIR = RESULTS_ROOT / "failure_reports"
STATUS_PATH = PROJECT_ROOT / "EURUSD_ECB_AUTOPILOT_STATUS.json"
HEARTBEAT_PATH = PROJECT_ROOT / "EURUSD_ECB_AUTOPILOT_HEARTBEAT.md"
RUNBOOK_PATH = PROJECT_ROOT / "EURUSD_ECB_AUTOPILOT_RUNBOOK.md"
STAGE2_DECISION_PATH = PROJECT_ROOT / "EURUSD_ECB_STAGE2_DECISION.md"
FINAL_DECISION_PATH = PROJECT_ROOT / "EURUSD_ECB_FINAL_DECISION.md"
FULL_OOS_PATH = PROJECT_ROOT / "EURUSD_ECB_FULL_CAMPAIGN_OOS_FINAL.md"
BUILD_BUNDLE_SCRIPT = PROJECT_ROOT / "scripts" / "build_chatgpt_bundle.py"
CANONICAL_ZIP = PROJECT_ROOT / "000_PARA_CHATGPT.zip"

SHORT_LEVELS: tuple[tuple[str, str], ...] = (
    ("prev_day_high", "prev_day"),
    ("asia_high", "asia"),
    ("london_high", "london"),
)
LONG_LEVELS: tuple[tuple[str, str], ...] = (
    ("prev_day_low", "prev_day"),
    ("asia_low", "asia"),
    ("london_low", "london"),
)

STAGE2_BLOCKS: tuple[tuple[str, str, str], ...] = (
    ("stage2_2020", "2020-01-01", "2020-12-31"),
    ("stage2_2021", "2021-01-01", "2021-12-31"),
    ("stage2_2022", "2022-01-01", "2022-12-31"),
    ("stage2_2023", "2023-01-01", "2023-12-31"),
    ("stage2_2024", "2024-01-01", "2024-12-31"),
    ("stage2_2025", "2025-01-01", "2025-12-31"),
)

FULL_CAMPAIGN_PERIODS: tuple[tuple[str, str, str], ...] = (
    ("development_2020_2023", "2020-01-01", "2023-12-31"),
    ("validation_2024", "2024-01-01", "2024-12-31"),
    ("holdout_2025", "2025-01-01", "2025-12-31"),
    ("full_2020_2025", "2020-01-01", "2025-12-31"),
)

H6_BENCHMARK = {
    "name": "H6_SILVER_BULLET_HYBRID",
    "profit_factor": 1.29,
    "expectancy_r": 0.089,
    "drawdown_r": -4.37,
}

AUDIT_TARGETS: tuple[str, ...] = (
    "scripts/hypothesis_admission_check.py",
    "scripts/hypothesis_viability_check.py",
    "EURUSD_LTF_OBJECTIVE_ENTRY_SPEC.md",
    "EURUSD_STAGE1_LTF_COMPARISON_RESULTS.md",
    "EURUSD_LTF_ENTRY_FINAL_VERDICT.md",
    "EURUSD_DESIGN_CONSTRAINT_BRIEF.md",
    "EURUSD_H6_SURVIVAL_PROFILE_AND_FAILURE_DELTA.md",
    "CURRENT_STATE_OF_LAB.md",
    "CAMPAIGN_GATEKEEPER_PROTOCOL.md",
    "CAMPAIGN_INTAKE_TEMPLATE.md",
    "RESEARCH_DECISION_MATRIX.md",
    "scripts/build_chatgpt_bundle.py",
    "ZIP_CONTENTS_MANIFEST.md",
    "ZIP_PACKAGING_AUDIT.md",
    "ZIP_DELIVERY_STATUS.md",
    "000_PARA_CHATGPT.zip",
)


@dataclass(frozen=True)
class AutopilotPaths:
    results_root: Path
    stage2_dir: Path
    full_dir: Path
    checkpoints_dir: Path
    failure_reports_dir: Path
    status_path: Path
    heartbeat_path: Path
    runbook_path: Path
    stage2_decision_path: Path
    final_decision_path: Path
    full_oos_path: Path
    precheck_audit_path: Path


def _fail_closed(message: str) -> None:
    raise RuntimeError(f"FAIL-CLOSED: {message}")


def _ensure_within_project(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(CANONICAL_ROOT)
    except ValueError as exc:
        raise RuntimeError(f"FAIL-CLOSED: path fuera del proyecto: {resolved}") from exc
    return resolved


def _ensure_canonical_root() -> None:
    if PROJECT_ROOT.resolve() != CANONICAL_ROOT:
        _fail_closed(f"root no canonico: {PROJECT_ROOT.resolve()}")
    if not CANONICAL_ROOT.exists():
        _fail_closed(f"root inexistente: {CANONICAL_ROOT}")


def _now_ny() -> pd.Timestamp:
    return pd.Timestamp.now(tz=NY_TZ)


def _timestamp_text() -> str:
    return _now_ny().strftime("%Y-%m-%d %H:%M:%S %Z")


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return str(value)
    raise TypeError(f"Tipo no serializable: {type(value)!r}")


def _write_json(path: Path, payload: Any) -> None:
    target = _ensure_within_project(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    target = _ensure_within_project(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _remove_if_exists(path: Path) -> None:
    target = _ensure_within_project(path)
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()


def build_paths() -> AutopilotPaths:
    return AutopilotPaths(
        results_root=_ensure_within_project(RESULTS_ROOT),
        stage2_dir=_ensure_within_project(STAGE2_RESULTS_DIR),
        full_dir=_ensure_within_project(FULL_RESULTS_DIR),
        checkpoints_dir=_ensure_within_project(CHECKPOINTS_DIR),
        failure_reports_dir=_ensure_within_project(FAILURE_REPORTS_DIR),
        status_path=_ensure_within_project(STATUS_PATH),
        heartbeat_path=_ensure_within_project(HEARTBEAT_PATH),
        runbook_path=_ensure_within_project(RUNBOOK_PATH),
        stage2_decision_path=_ensure_within_project(STAGE2_DECISION_PATH),
        final_decision_path=_ensure_within_project(FINAL_DECISION_PATH),
        full_oos_path=_ensure_within_project(FULL_OOS_PATH),
        precheck_audit_path=_ensure_within_project(PRECHECK_AUDIT_PATH),
    )


def build_engine_config(*, execution_mode: str = "high_precision_mode") -> EngineConfig:
    base = EngineConfig(
        pair=PAIR,
        risk_pct=0.5,
        max_spread_pips=2.0,
        slippage_pips=0.1,
        commission_per_lot_roundturn_usd=7.0,
        max_trades_per_day=2,
        enforce_hard_stop=True,
    )
    return with_execution_mode(base, execution_mode)


def build_news_config():
    return canonical_news_config(
        PAIR,
        enabled=True,
        pre_minutes=30,
        post_minutes=60,
        forced_exit_pre_news=True,
        cancel_pending_pre_news=True,
        pre_news_exit_minutes=10,
    )


def _audit_file(relative_path: str) -> dict[str, Any]:
    path = _ensure_within_project(PROJECT_ROOT / relative_path)
    if not path.exists():
        _fail_closed(f"falta artefacto auditado: {relative_path}")
    return {
        "relative_path": relative_path,
        "absolute_path": str(path),
        "size_bytes": int(path.stat().st_size),
        "modified_time": pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").tz_convert(NY_TZ).isoformat(),
    }


def verify_preconditions(paths: AutopilotPaths) -> dict[str, Any]:
    audited_files = [_audit_file(relative_path) for relative_path in AUDIT_TARGETS]
    final_verdict_text = (PROJECT_ROOT / "EURUSD_LTF_ENTRY_FINAL_VERDICT.md").read_text(encoding="utf-8", errors="ignore")
    stage1_text = (PROJECT_ROOT / "EURUSD_STAGE1_LTF_COMPARISON_RESULTS.md").read_text(encoding="utf-8", errors="ignore")
    current_state_text = (PROJECT_ROOT / "CURRENT_STATE_OF_LAB.md").read_text(encoding="utf-8", errors="ignore")
    if "ELIGIBLE_FOR_FULL_CAMPAIGN" not in final_verdict_text or "Solo Gatillo ECB" not in final_verdict_text:
        _fail_closed("ECB no figura como unico gatillo promovido en el veredicto final Stage-1.")
    stage1_upper = stage1_text.upper()
    if "EXTREME_CANDLE_BREAK" not in stage1_upper or "ADMISIBLE" not in stage1_upper:
        _fail_closed("Stage-1 no confirma a ECB como unico sobreviviente admisible.")
    if "ECB" not in current_state_text:
        _fail_closed("CURRENT_STATE_OF_LAB.md no refleja la linea ECB.")

    payload = {
        "checked_at_ny": _timestamp_text(),
        "canonical_root": str(CANONICAL_ROOT),
        "only_promoted_trigger": "ECB",
        "benchmark_locked": H6_BENCHMARK,
        "audited_files": audited_files,
    }
    _write_json(paths.precheck_audit_path, payload)
    return payload


def _body_outside_ratio(row: pd.Series, level_price: float, direction: str) -> float:
    open_price = float(row["open"])
    close_price = float(row["close"])
    body_size = abs(close_price - open_price)
    if body_size <= 0:
        return 0.0
    if direction == "short":
        outside = max(min(max(open_price, close_price) - level_price, body_size), 0.0)
    else:
        outside = max(min(level_price - min(open_price, close_price), body_size), 0.0)
    return float(outside / body_size)


def _h1_sweep_candidate(row: pd.Series, *, direction: str) -> dict[str, Any] | None:
    level_specs = SHORT_LEVELS if direction == "short" else LONG_LEVELS
    for level_name, source_kind in level_specs:
        if level_name not in row.index:
            continue
        level_price = _safe_float(row.get(level_name))
        if level_price is None:
            continue
        if source_kind in {"asia", "london"} and not bool(row.get(f"{source_kind}_complete", False)):
            continue
        if direction == "short":
            if float(row["high"]) < level_price + (MIN_SWEEP_PIPS * PIP_SIZE):
                continue
            if float(row["close"]) >= level_price:
                continue
            if _body_outside_ratio(row, level_price, direction) > 0.5:
                continue
            return {
                "direction": "short",
                "level_name": level_name,
                "source_kind": source_kind,
                "level_price": level_price,
                "sweep_price": float(row["high"]),
            }
        if float(row["low"]) > level_price - (MIN_SWEEP_PIPS * PIP_SIZE):
            continue
        if float(row["close"]) <= level_price:
            continue
        if _body_outside_ratio(row, level_price, direction) > 0.5:
            continue
        return {
            "direction": "long",
            "level_name": level_name,
            "source_kind": source_kind,
            "level_price": level_price,
            "sweep_price": float(row["low"]),
        }
    return None


def _extreme_candle(window: pd.DataFrame, *, direction: str, level_price: float) -> dict[str, Any] | None:
    if window.empty:
        return None
    tolerance = PIP_SIZE / 10.0
    if direction == "short":
        candidates = window.loc[window["high"] >= level_price + (MIN_SWEEP_PIPS * PIP_SIZE)].copy()
        if candidates.empty:
            return None
        extreme_price = float(candidates["high"].max())
        selected = candidates.loc[np.isclose(candidates["high"], extreme_price, atol=tolerance)]
        if len(selected) != 1:
            return None
        row = selected.iloc[0]
        return {
            "timestamp": selected.index[0],
            "extreme_price": extreme_price,
            "entry_trigger": float(row["low"]),
            "opposite_price": float(row["low"]),
        }
    candidates = window.loc[window["low"] <= level_price - (MIN_SWEEP_PIPS * PIP_SIZE)].copy()
    if candidates.empty:
        return None
    extreme_price = float(candidates["low"].min())
    selected = candidates.loc[np.isclose(candidates["low"], extreme_price, atol=tolerance)]
    if len(selected) != 1:
        return None
    row = selected.iloc[0]
    return {
        "timestamp": selected.index[0],
        "extreme_price": extreme_price,
        "entry_trigger": float(row["high"]),
        "opposite_price": float(row["high"]),
    }


def annotate_ecb_frame(m3_frame: pd.DataFrame, h1_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = m3_frame.copy()
    result["ecb_signal"] = False
    result["ecb_direction"] = ""
    result["ecb_stop_price"] = np.nan
    result["ecb_stop_entry_price"] = np.nan
    result["ecb_target_rr"] = TARGET_RR
    result["ecb_source_level_name"] = ""
    result["ecb_source_kind"] = ""
    result["ecb_level_price"] = np.nan
    result["ecb_sweep_price"] = np.nan
    result["ecb_sweep_time_ny"] = ""
    result["ecb_extreme_time_ny"] = ""

    signal_rows: list[dict[str, Any]] = []
    for sweep_ts, row in h1_frame.iterrows():
        short_candidate = _h1_sweep_candidate(row, direction="short")
        long_candidate = _h1_sweep_candidate(row, direction="long")
        if short_candidate is not None and long_candidate is not None:
            continue
        candidate = short_candidate or long_candidate
        if candidate is None:
            continue
        if sweep_ts not in result.index:
            continue

        m3_window = result.loc[(result.index > sweep_ts - pd.Timedelta(hours=1)) & (result.index <= sweep_ts)].copy()
        extreme = _extreme_candle(m3_window, direction=candidate["direction"], level_price=float(candidate["level_price"]))
        if extreme is None:
            continue

        signal_close = float(result.at[sweep_ts, "close"])
        stop_entry_price = float(extreme["entry_trigger"])
        if candidate["direction"] == "short" and stop_entry_price >= signal_close:
            continue
        if candidate["direction"] == "long" and stop_entry_price <= signal_close:
            continue

        stop_price = (
            float(candidate["sweep_price"]) + (STOP_BUFFER_PIPS * PIP_SIZE)
            if candidate["direction"] == "short"
            else float(candidate["sweep_price"]) - (STOP_BUFFER_PIPS * PIP_SIZE)
        )
        if candidate["direction"] == "short" and stop_price <= stop_entry_price:
            continue
        if candidate["direction"] == "long" and stop_price >= stop_entry_price:
            continue

        result.at[sweep_ts, "ecb_signal"] = True
        result.at[sweep_ts, "ecb_direction"] = candidate["direction"]
        result.at[sweep_ts, "ecb_stop_price"] = stop_price
        result.at[sweep_ts, "ecb_stop_entry_price"] = stop_entry_price
        result.at[sweep_ts, "ecb_source_level_name"] = candidate["level_name"]
        result.at[sweep_ts, "ecb_source_kind"] = candidate["source_kind"]
        result.at[sweep_ts, "ecb_level_price"] = float(candidate["level_price"])
        result.at[sweep_ts, "ecb_sweep_price"] = float(candidate["sweep_price"])
        result.at[sweep_ts, "ecb_sweep_time_ny"] = str(sweep_ts)
        result.at[sweep_ts, "ecb_extreme_time_ny"] = str(extreme["timestamp"])
        signal_rows.append(
            {
                "signal_time": sweep_ts,
                "direction": candidate["direction"],
                "source_kind": candidate["source_kind"],
                "source_level_name": candidate["level_name"],
                "level_price": float(candidate["level_price"]),
                "sweep_price": float(candidate["sweep_price"]),
                "stop_entry_price": stop_entry_price,
                "stop_price": stop_price,
                "sweep_time_ny": str(sweep_ts),
                "extreme_time_ny": str(extreme["timestamp"]),
            }
        )

    return result, pd.DataFrame(signal_rows)


def _filtered_high_precision_package(start: str, end: str) -> dict[str, pd.DataFrame]:
    package = load_high_precision_package(PAIR, DEFAULT_HIGH_PRECISION_PREPARED_DIR)
    start_ts = pd.Timestamp(start, tz=NY_TZ)
    end_ts = pd.Timestamp(end, tz=NY_TZ) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)

    filtered: dict[str, pd.DataFrame] = {}
    for side, source in package.items():
        frame = source.loc[(source.index >= start_ts) & (source.index <= end_ts)].copy()
        frame = frame[fx_market_mask(frame.index)].copy()
        validate_price_frame(frame)
        filtered[f"{side}_m1"] = frame
    return filtered


def _align_precision_package(filtered: dict[str, pd.DataFrame], frame_index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    bid_exec = resample_ohlcv_to_timeframe(filtered["bid_m1"], TIMEFRAME).loc[frame_index].copy()
    ask_exec = resample_ohlcv_to_timeframe(filtered["ask_m1"], TIMEFRAME).loc[frame_index].copy()
    mid_exec = resample_ohlcv_to_timeframe(filtered["mid_m1"], TIMEFRAME).loc[frame_index].copy()
    return {
        "bid_m1": filtered["bid_m1"].copy(),
        "ask_m1": filtered["ask_m1"].copy(),
        "mid_m1": filtered["mid_m1"].copy(),
        "bid_exec": bid_exec.copy(),
        "ask_exec": ask_exec.copy(),
        "mid_exec": mid_exec.copy(),
        "bid_m15": bid_exec.copy(),
        "ask_m15": ask_exec.copy(),
        "mid_m15": mid_exec.copy(),
    }


def build_research_frame(start: str, end: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    filtered = _filtered_high_precision_package(start, end)
    m3_frame = prepare_common_frame(filtered["mid_m1"], target_timeframe=TIMEFRAME)
    h1_frame = prepare_common_frame(filtered["mid_m1"], target_timeframe=SWEEP_TIMEFRAME)
    common_index = (
        m3_frame.index
        .intersection(resample_ohlcv_to_timeframe(filtered["bid_m1"], TIMEFRAME).index)
        .intersection(resample_ohlcv_to_timeframe(filtered["ask_m1"], TIMEFRAME).index)
    )
    annotated, signal_log = annotate_ecb_frame(m3_frame.loc[common_index].copy(), h1_frame.copy())
    precision_package = _align_precision_package(filtered, annotated.index)
    return annotated, precision_package, signal_log


def period_slice(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz=NY_TZ)
    end_ts = pd.Timestamp(end, tz=NY_TZ) + pd.Timedelta(days=1) - pd.Timedelta(minutes=3)
    return frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)].copy()


def _filter_signal_log(signal_log: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if signal_log.empty:
        return signal_log.copy()
    start_ts = pd.Timestamp(start, tz=NY_TZ)
    end_ts = pd.Timestamp(end, tz=NY_TZ) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    result = signal_log.copy()
    result["signal_time"] = pd.to_datetime(result["signal_time"], utc=True, errors="coerce").dt.tz_convert(NY_TZ)
    return result.loc[(result["signal_time"] >= start_ts) & (result["signal_time"] <= end_ts)].copy()


def _merge_trade_details(trades_export: pd.DataFrame, signal_log: pd.DataFrame) -> pd.DataFrame:
    if trades_export.empty:
        return trades_export.copy()
    trades = trades_export.copy()
    trades["signal_time"] = (
        pd.to_datetime(trades["signal_time_ny"], errors="coerce")
        .dt.tz_localize(NY_TZ, ambiguous="infer", nonexistent="shift_forward")
        .dt.floor("min")
    )
    if signal_log.empty:
        for column in ("source_kind", "source_level_name", "sweep_time_ny", "extreme_time_ny"):
            trades[column] = ""
        return trades
    details = signal_log.copy()
    details["signal_time"] = pd.to_datetime(details["signal_time"], utc=True, errors="coerce").dt.tz_convert(NY_TZ).dt.floor("min")
    merged = trades.merge(
        details[["signal_time", "source_kind", "source_level_name", "sweep_time_ny", "extreme_time_ny"]],
        on="signal_time",
        how="left",
    )
    for column in ("source_kind", "source_level_name", "sweep_time_ny", "extreme_time_ny"):
        merged[column] = merged[column].fillna("")
    return merged


def evaluate_period(
    *,
    frame: pd.DataFrame,
    precision_package: dict[str, pd.DataFrame],
    signal_log: pd.DataFrame,
    params: dict[str, Any],
    engine_config: EngineConfig,
    news_result: Any,
    news_config: Any,
    start: str,
    end: str,
) -> dict[str, Any]:
    period_frame = period_slice(frame, start, end)
    if period_frame.empty:
        raise ValueError(f"Periodo vacio para {start} -> {end}")
    period_precision = _align_precision_package(
        {"bid_m1": precision_package["bid_m1"], "ask_m1": precision_package["ask_m1"], "mid_m1": precision_package["mid_m1"]},
        period_frame.index,
    )
    period_signal_log = _filter_signal_log(signal_log, start, end)
    news_block = build_entry_block(entry_open_index(period_frame.index), news_result.events, news_config)
    result = run_backtest(
        strategy_module=strategy_module,
        frame=period_frame,
        params=params,
        engine_config=engine_config,
        news_block=news_block,
        news_filter_used=news_result.enabled,
        precision_package=period_precision,
        data_source_used="dukascopy_m1_bid_ask_full",
        news_events=news_result.events,
        news_settings=news_config,
    )
    summary, trades_export, monthly_stats, yearly_stats, equity_export = summarize_result(
        strategy_module.NAME,
        result.trades,
        result.equity_curve,
        params,
        news_result.enabled,
        INITIAL_CAPITAL,
        None,
        costs_used={"execution_mode": engine_config.execution_mode, "cost_profile": engine_config.cost_profile},
        timeframe=TIMEFRAME,
        schedule_used={"sweep_frame": SWEEP_TIMEFRAME, "entry_frame": TIMEFRAME, "entry_mode": "stop", "target_rr": str(TARGET_RR)},
        break_even_setting=None,
    )
    trades_export = _merge_trade_details(trades_export, period_signal_log)
    return {
        "summary": summary,
        "trades_export": trades_export,
        "monthly_stats": monthly_stats,
        "yearly_stats": yearly_stats,
        "equity_export": equity_export,
        "signal_log": period_signal_log,
    }


def _checkpoint_paths(paths: AutopilotPaths, label: str) -> dict[str, Path]:
    root = paths.checkpoints_dir / label
    return {
        "root": root,
        "summary": root / "summary.json",
        "trades": root / "trades.csv",
        "signal_log": root / "signal_log.csv",
        "yearly": root / "yearly_stats.csv",
        "monthly": root / "monthly_stats.csv",
        "equity": root / "equity_curve.csv",
    }


def _checkpoint_exists(paths: AutopilotPaths, label: str) -> bool:
    expected = _checkpoint_paths(paths, label)
    return all(expected[key].exists() for key in ("summary", "trades", "signal_log", "yearly", "monthly", "equity"))


def _save_checkpoint(paths: AutopilotPaths, label: str, payload: dict[str, Any]) -> None:
    target = _checkpoint_paths(paths, label)
    target["root"].mkdir(parents=True, exist_ok=True)
    _write_json(target["summary"], payload["summary"])
    payload["trades_export"].to_csv(target["trades"], index=False)
    payload["signal_log"].to_csv(target["signal_log"], index=False)
    payload["yearly_stats"].to_csv(target["yearly"], index=False)
    payload["monthly_stats"].to_csv(target["monthly"], index=False)
    payload["equity_export"].to_csv(target["equity"], index=False)


def _load_checkpoint(paths: AutopilotPaths, label: str) -> dict[str, Any]:
    target = _checkpoint_paths(paths, label)
    return {
        "summary": json.loads(target["summary"].read_text(encoding="utf-8")),
        "trades_export": pd.read_csv(target["trades"]),
        "signal_log": pd.read_csv(target["signal_log"]),
        "yearly_stats": pd.read_csv(target["yearly"]),
        "monthly_stats": pd.read_csv(target["monthly"]),
        "equity_export": pd.read_csv(target["equity"]),
    }


def _profit_factor_from_r(pnl_r: pd.Series) -> float:
    gross_profit = float(pnl_r[pnl_r > 0].sum())
    gross_loss = float(pnl_r[pnl_r < 0].sum())
    return gross_profit / abs(gross_loss) if gross_loss < 0 else float("inf")


def _max_drawdown_r(pnl_r: pd.Series) -> float:
    if pnl_r.empty:
        return 0.0
    cumulative = pnl_r.cumsum()
    running_peak = cumulative.cummax()
    drawdown = cumulative - running_peak
    return float(drawdown.min())


def _stage2_metrics(trades_export: pd.DataFrame) -> dict[str, Any]:
    pnl_r = pd.to_numeric(trades_export.get("pnl_r", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    return {
        "total_trades": int(len(trades_export)),
        "profit_factor": _profit_factor_from_r(pnl_r),
        "expectancy_r": float(pnl_r.mean()) if len(pnl_r) else 0.0,
        "max_drawdown_r": _max_drawdown_r(pnl_r),
        "wins": int((pnl_r > 0).sum()),
        "losses": int((pnl_r < 0).sum()),
        "breakevens": int((pnl_r == 0).sum()),
    }


def _stage2_gate_snapshot(trades_export: pd.DataFrame) -> dict[str, Any]:
    metrics = _stage2_metrics(trades_export)
    total_trades = int(metrics["total_trades"])
    profit_factor = float(metrics["profit_factor"])
    expectancy_r = float(metrics["expectancy_r"])
    max_drawdown_r = float(metrics["max_drawdown_r"])
    gate_triggered = None
    decision = "CONTINUE_STAGE2"

    if total_trades >= 40 and (profit_factor < 1.00 or expectancy_r <= 0.0 or max_drawdown_r <= -6.0):
        gate_triggered = "GATE_A"
        decision = "REJECT_EARLY"
    elif total_trades >= 80 and (profit_factor < 1.15 or expectancy_r < 0.10 or max_drawdown_r <= -8.0):
        gate_triggered = "GATE_B"
        decision = "REJECT_EARLY"
    elif total_trades >= 100:
        gate_triggered = "GATE_C"
        if profit_factor >= 1.35 and expectancy_r >= 0.15 and max_drawdown_r > -10.0:
            decision = "PROMOTE_TO_FULL_CAMPAIGN"
        elif profit_factor < 1.0 or expectancy_r <= 0.0 or max_drawdown_r <= -10.0:
            decision = "REJECT"
        else:
            decision = "NEEDS_REDESIGN"

    return {
        **metrics,
        "gate_triggered": gate_triggered,
        "decision": decision,
    }


def _equity_export_from_trades(trades_export: pd.DataFrame) -> pd.DataFrame:
    if trades_export.empty:
        return pd.DataFrame(columns=["datetime_ny", "equity", "drawdown_pct"])
    ordered = trades_export.copy()
    ordered["exit_time_ny"] = pd.to_datetime(ordered["exit_time_ny"], errors="coerce")
    ordered = ordered.sort_values(["exit_time_ny", "entry_time_ny"], kind="stable").reset_index(drop=True)
    pnl_usd = pd.to_numeric(ordered["pnl_usd"], errors="coerce").fillna(0.0)
    equity = INITIAL_CAPITAL + pnl_usd.cumsum()
    peak = equity.cummax()
    drawdown_pct = ((equity - peak) / peak.replace(0.0, np.nan)) * 100.0
    return pd.DataFrame(
        {
            "datetime_ny": ordered["exit_time_ny"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "equity": equity,
            "drawdown_pct": drawdown_pct.fillna(0.0),
        }
    )


def _aggregate_blocks(label: str, block_results: dict[str, dict[str, Any]], params: dict[str, Any], engine_config: EngineConfig) -> dict[str, Any]:
    ordered_labels = [key for key in block_results.keys()]
    trades_frames = [block_results[key]["trades_export"] for key in ordered_labels]
    signal_frames = [block_results[key]["signal_log"] for key in ordered_labels]
    trades_export = pd.concat(trades_frames, ignore_index=True) if trades_frames else pd.DataFrame()
    signal_log = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    monthly_stats = build_period_stats(trades_export, "M", INITIAL_CAPITAL)
    yearly_stats = build_period_stats(trades_export, "Y", INITIAL_CAPITAL)
    equity_export = _equity_export_from_trades(trades_export)
    summary = build_summary(
        strategy_name=strategy_module.NAME,
        trades_export=trades_export,
        equity_export=equity_export,
        monthly_stats=monthly_stats,
        yearly_stats=yearly_stats,
        params=params,
        news_filter_used=True,
        selected_score=None,
        costs_used={"execution_mode": engine_config.execution_mode, "cost_profile": engine_config.cost_profile},
        timeframe=TIMEFRAME,
        schedule_used={"aggregation": label, "entry_mode": "stop", "sweep_frame": SWEEP_TIMEFRAME},
        break_even_setting=None,
    )
    stage2_metrics = _stage2_metrics(trades_export)
    summary["profit_factor"] = stage2_metrics["profit_factor"]
    summary["max_drawdown_r_closed_trades"] = stage2_metrics["max_drawdown_r"]
    summary["aggregated_labels"] = ordered_labels
    return {
        "summary": summary,
        "trades_export": trades_export,
        "signal_log": signal_log,
        "monthly_stats": monthly_stats,
        "yearly_stats": yearly_stats,
        "equity_export": equity_export,
    }


def _status_payload(
    *,
    phase: str,
    current_segment: str,
    last_checkpoint: str,
    next_action: str,
    completed_segments: list[str],
    decision: str | None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "updated_at_ny": _timestamp_text(),
        "strategy_name": strategy_module.NAME,
        "phase": phase,
        "current_segment": current_segment,
        "last_checkpoint": last_checkpoint,
        "next_action": next_action,
        "completed_segments": completed_segments,
        "decision": decision,
        "benchmark_reference": H6_BENCHMARK,
        "results_root": str(RESULTS_ROOT),
        "checkpoints_dir": str(CHECKPOINTS_DIR),
        "details": details or {},
    }


def _write_status(paths: AutopilotPaths, payload: dict[str, Any]) -> None:
    _write_json(paths.status_path, payload)
    lines = [
        "# EURUSD ECB Autopilot Heartbeat",
        "",
        f"- Ultima actividad: `{payload['updated_at_ny']}`",
        f"- Fase: `{payload['phase']}`",
        f"- Segmento en curso: `{payload['current_segment']}`",
        f"- Ultimo checkpoint: `{payload['last_checkpoint']}`",
        f"- Proxima accion: `{payload['next_action']}`",
        f"- Segmentos completados: `{', '.join(payload['completed_segments']) if payload['completed_segments'] else 'ninguno'}`",
        f"- Decision actual: `{payload['decision'] or 'PENDING'}`",
        f"- Root resultados: `{payload['results_root']}`",
    ]
    _write_text(paths.heartbeat_path, "\n".join(lines) + "\n")


def write_runbook(paths: AutopilotPaths) -> None:
    lines = [
        "# EURUSD ECB Autopilot Runbook",
        "",
        "## Alcance fijo",
        "",
        "- Activo: `EURUSD`.",
        "- Linea: `EURUSD_LTF_OBJECTIVE_ENTRY_REPLACEMENT`.",
        "- Gatillo unico: `ECB = extreme_candle_break`.",
        "- Benchmark solo comparativo: `H6_SILVER_BULLET_HYBRID`.",
        "- H6 no se toca.",
        "",
        "## Contrato mecanico",
        "",
        "- Sweep HTF: `H1` sobre `prev_day`, `asia`, `london`.",
        "- Confirmacion: cierre H1 de rechazo, no breakout genuino.",
        "- Entrada LTF: `M3`, `stop-entry` en el extremo opuesto de la vela extrema post-sweep dentro de la hora confirmada.",
        "- Stop: extremo del sweep + 1 pip.",
        "- Target: `1.5R` fijo.",
        "- News Fortress: obligatorio, fail-closed.",
        "",
        "## Rutas canonicas",
        "",
        f"- Status: `{paths.status_path}`",
        f"- Heartbeat: `{paths.heartbeat_path}`",
        f"- Checkpoints: `{paths.checkpoints_dir}`",
        f"- Resultados: `{paths.results_root}`",
        "",
        "## Reanudacion",
        "",
        "- El runner reutiliza checkpoints por bloque si ya existen completos.",
        "- Si Stage-2 ya tomo decision valida, no recomputa bloques cerrados.",
        "- Si Full Campaign no fue promovida, cualquier salida vieja de Full se limpia para evitar ambiguedad canonica.",
        "",
        "## Ejecucion local",
        "",
        "- Comando canonico: `python scripts/run_eurusd_ecb_autopilot.py`",
        "",
        "## Cierre",
        "",
        "- Si Stage-2 falla gates, la corrida cierra duro y documenta decision unica.",
        "- Si Stage-2 promueve, se ejecuta una sola Full Campaign con dev/val/holdout y stress conservador.",
        "- El zip maestro se reconstruye solo al cierre canonico.",
        "",
    ]
    _write_text(paths.runbook_path, "\n".join(lines))


def _failure_report(paths: AutopilotPaths, phase: str, exc: BaseException) -> Path:
    paths.failure_reports_dir.mkdir(parents=True, exist_ok=True)
    target = paths.failure_reports_dir / f"{_now_ny().strftime('%Y%m%d_%H%M%S')}_{phase}_failure.json"
    payload = {
        "failed_at_ny": _timestamp_text(),
        "phase": phase,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    _write_json(target, payload)
    return target


def _stage2_decision_doc(stage2_gate: dict[str, Any], block_table: pd.DataFrame, stage2_summary: dict[str, Any]) -> str:
    lines = [
        "# EURUSD ECB Stage-2 Decision",
        "",
        f"- Fecha NY: `{_timestamp_text()}`",
        f"- Benchmark de referencia: `{H6_BENCHMARK['name']}`",
        f"- Decision Stage-2: `{stage2_gate['decision']}`",
        f"- Gate activado: `{stage2_gate['gate_triggered'] or 'ninguno'}`",
        f"- Trades acumulados: `{stage2_gate['total_trades']}`",
        f"- Profit Factor: `{stage2_gate['profit_factor']:.4f}`",
        f"- Expectancy: `{stage2_gate['expectancy_r']:.4f}R`",
        f"- Drawdown cerrado: `{stage2_gate['max_drawdown_r']:.4f}R`",
        "",
        "## Bloques ejecutados",
        "",
    ]
    if block_table.empty:
        lines.append("- Sin bloques ejecutados.")
    else:
        lines.extend(
            [
                "| Bloque | Trades | PF | Expectancy R | DD cerrado R |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in block_table.itertuples(index=False):
            lines.append(
                f"| `{row.block_label}` | {int(row.total_trades)} | {float(row.profit_factor):.4f} | {float(row.expectancy_r):.4f} | {float(row.max_drawdown_r):.4f} |"
            )
    lines.extend(
        [
            "",
            "## Resumen consolidado",
            "",
            f"- Win rate: `{float(stage2_summary['win_rate']):.2f}%`",
            f"- Return %: `{float(stage2_summary['total_return_pct']):.4f}`",
            f"- Max drawdown %: `{float(stage2_summary['max_drawdown_pct']):.4f}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _full_campaign_oos_doc(period_summaries: dict[str, dict[str, Any]], stress_summary: dict[str, Any], final_decision: str, final_reason: str) -> str:
    dev = period_summaries["development_2020_2023"]
    val = period_summaries["validation_2024"]
    hold = period_summaries["holdout_2025"]
    full = period_summaries["full_2020_2025"]
    lines = [
        "# EURUSD ECB Full Campaign OOS Final",
        "",
        f"- Fecha NY: `{_timestamp_text()}`",
        f"- Decision final: `{final_decision}`",
        f"- Motivo: {final_reason}",
        "",
        "| Periodo | Trades | PF | Expectancy R | DD % |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| `development_2020_2023` | {int(dev['total_trades'])} | {float(dev['profit_factor']):.4f} | {float(dev['expectancy_r']):.4f} | {float(dev['max_drawdown_pct']):.4f} |",
        f"| `validation_2024` | {int(val['total_trades'])} | {float(val['profit_factor']):.4f} | {float(val['expectancy_r']):.4f} | {float(val['max_drawdown_pct']):.4f} |",
        f"| `holdout_2025` | {int(hold['total_trades'])} | {float(hold['profit_factor']):.4f} | {float(hold['expectancy_r']):.4f} | {float(hold['max_drawdown_pct']):.4f} |",
        f"| `full_2020_2025` | {int(full['total_trades'])} | {float(full['profit_factor']):.4f} | {float(full['expectancy_r']):.4f} | {float(full['max_drawdown_pct']):.4f} |",
        f"| `stress_full_2020_2025` | {int(stress_summary['total_trades'])} | {float(stress_summary['profit_factor']):.4f} | {float(stress_summary['expectancy_r']):.4f} | {float(stress_summary['max_drawdown_pct']):.4f} |",
        "",
        "## Comparacion H6",
        "",
        f"- H6 PF: `{H6_BENCHMARK['profit_factor']}`",
        f"- H6 Expectancy: `{H6_BENCHMARK['expectancy_r']}R`",
        f"- H6 Drawdown: `{H6_BENCHMARK['drawdown_r']}R`",
        "",
        "- Este cierre no aprueba nada para real.",
        "- H6 permanece intacto como benchmark vigente.",
        "",
    ]
    return "\n".join(lines)


def _final_decision_doc(final_decision: str, final_reason: str, stage2_gate: dict[str, Any], full_summary: dict[str, Any] | None) -> str:
    lines = [
        "# EURUSD ECB Final Decision",
        "",
        f"- Fecha NY: `{_timestamp_text()}`",
        f"- Decision final: `{final_decision}`",
        f"- Motivo principal: {final_reason}",
        f"- Decision Stage-2: `{stage2_gate['decision']}`",
        f"- Gate final Stage-2: `{stage2_gate['gate_triggered'] or 'ninguno'}`",
    ]
    if full_summary is not None:
        lines.extend(
            [
                f"- Full Campaign PF: `{float(full_summary['profit_factor']):.4f}`",
                f"- Full Campaign Expectancy: `{float(full_summary['expectancy_r']):.4f}R`",
                f"- Full Campaign DD %: `{float(full_summary['max_drawdown_pct']):.4f}`",
            ]
        )
    lines.extend(
        [
            "",
            "- No queda aprobado para real.",
            "- El laboratorio mantiene disciplina fail-closed.",
            "",
        ]
    )
    return "\n".join(lines)


def _cleanup_stale_full_outputs(paths: AutopilotPaths) -> None:
    _remove_if_exists(paths.full_dir)
    _remove_if_exists(paths.full_oos_path)


def _update_lab_state(final_decision: str, final_reason: str) -> None:
    lines = [
        "# ESTADO ACTUAL DEL LABORATORIO",
        "",
        "Este documento centraliza la verdad operativa vigente del laboratorio.",
        "",
        "## 1. Mision Principal",
        "",
        "- Fase Actual: `STANDBY`",
        "- Estrategia Benchmark: `H6_SILVER_BULLET_HYBRID` (Frozen)",
        "- Activo Auditado: `EURUSD`",
        "",
        "## 2. Benchmark Vigente (H6_SILVER_BULLET_HYBRID)",
        "",
        "- Sample: `20` senales de backfill",
        "- Profit Factor: `1.29`",
        "- Expectancy: `0.089R / trade`",
        "- Drawdown: `-4.37R`",
        "- Estado: `VIGENTE Y CONGELADO`",
        "",
        "## 3. Campanas y Lineas Cerradas",
        "",
        "### A. Manual-Edge (Subjective ICT)",
        "- Estado: `CLOSED`",
        "- Veredicto: `NOT_TRANSLATABLE`",
        "- Decision Operativa: `STOP_AND_FREEZE`",
        "",
        "### B. Campaign 3B (Logic Expansion)",
        "- Estado: `CLOSED`",
        "- Veredicto: `REJECT`",
        "",
        "### C. Campaign 4 (C4-ICT-ALIGN)",
        "- Estado: `CLOSED`",
        "- Veredicto: `REJECT`",
        "",
        "## 4. Linea ECB Autopilot",
        "",
        "- Linea: `EURUSD_LTF_OBJECTIVE_ENTRY_REPLACEMENT`",
        "- Gatillo unico auditado: `ECB (extreme_candle_break)`",
        f"- Decision final: `{final_decision}`",
        f"- Motivo sintetico: {final_reason}",
        "- Estado maestro del laboratorio: `STANDBY`",
        "- H6 se mantiene intacto como benchmark vigente.",
        "",
        "---",
        f"Ultima Canonizacion: {_now_ny().strftime('%Y-%m-%d')}",
        "Estado Global: `STANDBY`",
    ]
    _write_text(PROJECT_ROOT / "CURRENT_STATE_OF_LAB.md", "\n".join(lines) + "\n")


def run_stage2(
    *,
    paths: AutopilotPaths,
    frame: pd.DataFrame,
    precision_package: dict[str, pd.DataFrame],
    signal_log: pd.DataFrame,
    params: dict[str, Any],
    engine_config: EngineConfig,
    news_result: Any,
    news_config: Any,
) -> dict[str, Any]:
    block_results: dict[str, dict[str, Any]] = {}
    block_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []

    for block_label, start, end in STAGE2_BLOCKS:
        if _checkpoint_exists(paths, block_label):
            payload = _load_checkpoint(paths, block_label)
        else:
            status_payload = _status_payload(
                phase="RUNNING_STAGE2",
                current_segment=block_label,
                last_checkpoint="none",
                next_action=f"evaluar_{block_label}",
                completed_segments=list(block_results.keys()),
                decision=None,
            )
            _write_status(paths, status_payload)
            payload = evaluate_period(
                frame=frame,
                precision_package=precision_package,
                signal_log=signal_log,
                params=params,
                engine_config=engine_config,
                news_result=news_result,
                news_config=news_config,
                start=start,
                end=end,
            )
            _save_checkpoint(paths, block_label, payload)

        block_results[block_label] = payload
        cumulative = _aggregate_blocks("stage2", block_results, params, engine_config)
        gate_snapshot = _stage2_gate_snapshot(cumulative["trades_export"])
        block_rows.append({"block_label": block_label, **gate_snapshot})
        gate_rows.append({"block_label": block_label, **gate_snapshot})

        export_strategy_bundle(
            paths.stage2_dir,
            summary=cumulative["summary"],
            trades_export=cumulative["trades_export"],
            monthly_stats=cumulative["monthly_stats"],
            yearly_stats=cumulative["yearly_stats"],
            equity_export=cumulative["equity_export"],
            optimization_results=pd.DataFrame(block_rows),
            extra_frames={
                "signal_log.csv": cumulative["signal_log"],
                "gate_snapshots.csv": pd.DataFrame(gate_rows),
            },
            extra_json={
                "stage2_gate_evaluation.json": gate_snapshot,
                "stage2_block_order.json": [row["block_label"] for row in block_rows],
            },
        )

        status_payload = _status_payload(
            phase="RUNNING_STAGE2",
            current_segment=block_label,
            last_checkpoint=str(_checkpoint_paths(paths, block_label)["summary"]),
            next_action="evaluar_siguiente_bloque_stage2",
            completed_segments=list(block_results.keys()),
            decision=gate_snapshot["decision"] if gate_snapshot["decision"] != "CONTINUE_STAGE2" else None,
            details=gate_snapshot,
        )
        _write_status(paths, status_payload)

        if gate_snapshot["decision"] in {"REJECT_EARLY", "REJECT", "NEEDS_REDESIGN", "PROMOTE_TO_FULL_CAMPAIGN"}:
            block_table = pd.DataFrame(block_rows)
            _write_text(paths.stage2_decision_path, _stage2_decision_doc(gate_snapshot, block_table, cumulative["summary"]))
            return {
                "block_results": block_results,
                "block_table": block_table,
                "gate_snapshot": gate_snapshot,
                "stage2_bundle": cumulative,
            }

    cumulative = _aggregate_blocks("stage2", block_results, params, engine_config)
    gate_snapshot = _stage2_gate_snapshot(cumulative["trades_export"])
    if gate_snapshot["decision"] == "CONTINUE_STAGE2":
        gate_snapshot["decision"] = "REJECT" if gate_snapshot["profit_factor"] < 1.0 or gate_snapshot["expectancy_r"] <= 0.0 else "NEEDS_REDESIGN"
    block_table = pd.DataFrame(block_rows)
    _write_text(paths.stage2_decision_path, _stage2_decision_doc(gate_snapshot, block_table, cumulative["summary"]))
    return {
        "block_results": block_results,
        "block_table": block_table,
        "gate_snapshot": gate_snapshot,
        "stage2_bundle": cumulative,
    }


def run_full_campaign(
    *,
    paths: AutopilotPaths,
    frame: pd.DataFrame,
    precision_package: dict[str, pd.DataFrame],
    signal_log: pd.DataFrame,
    params: dict[str, Any],
    engine_config: EngineConfig,
    news_result: Any,
    news_config: Any,
) -> dict[str, Any]:
    period_results: dict[str, dict[str, Any]] = {}
    period_rows: list[dict[str, Any]] = []

    for label, start, end in FULL_CAMPAIGN_PERIODS:
        checkpoint_label = f"full_{label}"
        if _checkpoint_exists(paths, checkpoint_label):
            payload = _load_checkpoint(paths, checkpoint_label)
        else:
            status_payload = _status_payload(
                phase="RUNNING_FULL_CAMPAIGN",
                current_segment=label,
                last_checkpoint="none",
                next_action=f"evaluar_{label}",
                completed_segments=list(period_results.keys()),
                decision=None,
            )
            _write_status(paths, status_payload)
            payload = evaluate_period(
                frame=frame,
                precision_package=precision_package,
                signal_log=signal_log,
                params=params,
                engine_config=engine_config,
                news_result=news_result,
                news_config=news_config,
                start=start,
                end=end,
            )
            _save_checkpoint(paths, checkpoint_label, payload)
        period_results[label] = payload
        summary = payload["summary"]
        period_rows.append(
            {
                "period": label,
                "total_trades": int(summary["total_trades"]),
                "profit_factor": float(summary["profit_factor"]),
                "expectancy_r": float(summary["expectancy_r"]),
                "max_drawdown_pct": float(summary["max_drawdown_pct"]),
            }
        )

    stress_config = build_engine_config(execution_mode="conservative_mode")
    stress_result = evaluate_period(
        frame=frame,
        precision_package=precision_package,
        signal_log=signal_log,
        params=params,
        engine_config=stress_config,
        news_result=news_result,
        news_config=news_config,
        start="2020-01-01",
        end="2025-12-31",
    )
    full_payload = period_results["full_2020_2025"]
    export_strategy_bundle(
        paths.full_dir,
        summary=full_payload["summary"],
        trades_export=full_payload["trades_export"],
        monthly_stats=full_payload["monthly_stats"],
        yearly_stats=full_payload["yearly_stats"],
        equity_export=full_payload["equity_export"],
        optimization_results=pd.DataFrame(period_rows),
        extra_frames={
            "signal_log.csv": full_payload["signal_log"],
            "stress_trades.csv": stress_result["trades_export"],
        },
        extra_json={
            "period_summaries.json": {label: payload["summary"] for label, payload in period_results.items()},
            "stress_summary.json": stress_result["summary"],
        },
    )

    dev = period_results["development_2020_2023"]["summary"]
    val = period_results["validation_2024"]["summary"]
    hold = period_results["holdout_2025"]["summary"]
    full = period_results["full_2020_2025"]["summary"]
    stress = stress_result["summary"]

    full_metrics = _stage2_metrics(full_payload["trades_export"])
    beats_h6 = (
        float(full["profit_factor"]) > H6_BENCHMARK["profit_factor"]
        and float(full["expectancy_r"]) > H6_BENCHMARK["expectancy_r"]
        and float(full_metrics["max_drawdown_r"]) > H6_BENCHMARK["drawdown_r"]
    )

    if (
        int(val["total_trades"]) == 0
        or int(hold["total_trades"]) == 0
        or float(val["profit_factor"]) < 1.0
        or float(hold["profit_factor"]) < 1.0
        or float(val["expectancy_r"]) <= 0.0
        or float(hold["expectancy_r"]) <= 0.0
        or float(stress["profit_factor"]) < 1.0
        or float(stress["expectancy_r"]) <= 0.0
        or not beats_h6
    ):
        final_decision = "REJECT"
        final_reason = "La Full Campaign no sostuvo OOS y/o no supero al benchmark H6 con evidencia suficiente."
    else:
        final_decision = "CONTINUE_RESEARCH_ONLY"
        final_reason = "La Full Campaign sobrevivio con evidencia defendible, pero queda solo aprobada para research continuado."

    _write_text(
        paths.full_oos_path,
        _full_campaign_oos_doc(
            {label: payload["summary"] for label, payload in period_results.items()},
            stress_result["summary"],
            final_decision,
            final_reason,
        ),
    )
    return {
        "period_results": period_results,
        "stress_result": stress_result,
        "final_decision": final_decision,
        "final_reason": final_reason,
    }


def _zip_delivery_payload() -> dict[str, Any]:
    if not CANONICAL_ZIP.exists():
        _fail_closed("No existe el zip canonico final despues de la reconstruccion.")
    digest = sha256(CANONICAL_ZIP.read_bytes()).hexdigest().upper()
    import zipfile

    with zipfile.ZipFile(CANONICAL_ZIP, "r") as archive:
        internal_names = archive.namelist()
    return {
        "path": str(CANONICAL_ZIP),
        "size_bytes": int(CANONICAL_ZIP.stat().st_size),
        "sha256": digest,
        "internal_count": int(len(internal_names)),
        "internal_names": internal_names,
        "timestamp_ny": _timestamp_text(),
    }


def _write_zip_delivery_status(payload: dict[str, Any]) -> None:
    lines = [
        "# ZIP Delivery Status",
        "",
        "- Archivo: 000_PARA_CHATGPT.zip",
        f"- Ruta Absoluta: {payload['path']}",
        f"- Tamano: {payload['size_bytes']} bytes",
        f"- Timestamp NY: {payload['timestamp_ny']}",
        f"- SHA256: {payload['sha256']}",
        f"- Cantidad de Archivos Internos: {payload['internal_count']}",
        "- Estado Final: READY_FOR_UPLOAD",
        "",
    ]
    _write_text(PROJECT_ROOT / "ZIP_DELIVERY_STATUS.md", "\n".join(lines))


def _rebuild_bundle() -> dict[str, Any]:
    exit_code = 1
    output = ""
    try:
        import subprocess

        completed = subprocess.run(
            [sys.executable, str(BUILD_BUNDLE_SCRIPT)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        exit_code = completed.returncode
        output = (completed.stdout or "") + (completed.stderr or "")
    except OSError as exc:
        raise RuntimeError(f"No pude ejecutar el builder del zip: {exc}") from exc
    if exit_code != 0:
        raise RuntimeError(f"Fallo la reconstruccion del zip canonico.\n{output}")
    payload = _zip_delivery_payload()
    _write_zip_delivery_status(payload)
    return payload


def main() -> dict[str, Any]:
    _ensure_canonical_root()
    paths = build_paths()
    for directory in (paths.results_root, paths.stage2_dir, paths.failure_reports_dir, paths.checkpoints_dir):
        directory.mkdir(parents=True, exist_ok=True)

    write_runbook(paths)
    verify_preconditions(paths)

    status_payload = _status_payload(
        phase="PRECHECK_COMPLETE",
        current_segment="bootstrap",
        last_checkpoint="none",
        next_action="build_research_frame",
        completed_segments=[],
        decision=None,
    )
    _write_status(paths, status_payload)

    news_config = build_news_config()
    news_result = require_operational_news(PAIR, news_config, context=strategy_module.NAME)
    frame, precision_package, signal_log = build_research_frame("2020-01-01", "2025-12-31")
    params = strategy_module.default_params()
    engine_config = build_engine_config(execution_mode="high_precision_mode")

    stage2_result = run_stage2(
        paths=paths,
        frame=frame,
        precision_package=precision_package,
        signal_log=signal_log,
        params=params,
        engine_config=engine_config,
        news_result=news_result,
        news_config=news_config,
    )
    stage2_gate = stage2_result["gate_snapshot"]

    full_result: dict[str, Any] | None = None
    if stage2_gate["decision"] == "PROMOTE_TO_FULL_CAMPAIGN":
        full_result = run_full_campaign(
            paths=paths,
            frame=frame,
            precision_package=precision_package,
            signal_log=signal_log,
            params=params,
            engine_config=engine_config,
            news_result=news_result,
            news_config=news_config,
        )
        final_decision = str(full_result["final_decision"])
        final_reason = str(full_result["final_reason"])
    else:
        _cleanup_stale_full_outputs(paths)
        final_decision = "REJECT" if stage2_gate["decision"] in {"REJECT_EARLY", "REJECT"} else "NEEDS_REDESIGN"
        final_reason = (
            "ECB no supero las puertas cuantitativas de Stage-2." if final_decision == "REJECT" else "ECB no alcanzo promocion limpia a Full Campaign y requiere redisenio."
        )

    _write_text(
        paths.final_decision_path,
        _final_decision_doc(
            final_decision,
            final_reason,
            stage2_gate,
            full_result["period_results"]["full_2020_2025"]["summary"] if full_result is not None else None,
        ),
    )
    _update_lab_state(final_decision, final_reason)

    bundle_payload = _rebuild_bundle()
    final_status = _status_payload(
        phase="COMPLETED",
        current_segment="done",
        last_checkpoint=str(paths.stage2_decision_path),
        next_action="none",
        completed_segments=list(stage2_result["block_results"].keys()),
        decision=final_decision,
        details={
            "stage2_decision": stage2_gate["decision"],
            "bundle": bundle_payload,
        },
    )
    _write_status(paths, final_status)
    return {
        "final_decision": final_decision,
        "final_reason": final_reason,
        "stage2_gate": stage2_gate,
        "bundle": bundle_payload,
    }


if __name__ == "__main__":
    try:
        payload = main()
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    except Exception as exc:  # pragma: no cover - explicit fail-closed boundary
        _ensure_canonical_root()
        paths = build_paths()
        failure_path = _failure_report(paths, "autopilot", exc)
        _write_status(
            paths,
            _status_payload(
                phase="FAILED",
                current_segment="autopilot",
                last_checkpoint=str(failure_path),
                next_action="manual_audit_required",
                completed_segments=[],
                decision="BLOCKED_FOR_SAFETY",
                details={"error": str(exc)},
            ),
        )
        raise
