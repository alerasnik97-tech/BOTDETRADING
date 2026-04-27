from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(CURRENT_DIR))

from research_lab.config import EngineConfig, NY_TZ, NewsConfig
from research_lab.data_loader import load_high_precision_package, prepare_common_frame, resample_ohlcv_to_timeframe
from research_lab.engine import actual_spread_pips, entry_open_index, run_backtest
from research_lab.ict_primitives import add_ict_primitives
from research_lab.news_filter import build_entry_block, build_news_guard_details, require_operational_news
from research_lab.report import summarize_result
from research_lab.strategies import eurusd_am_post_news_external_liquidity_shift as strategy_module
from research_lab.eurusd_am_post_news_external_liquidity_shift_runner import (
    AM_NEWS_FILE,
    annotate_post_news_external_liquidity_shift_frame,
    schedule_used,
)

PAIR = "EURUSD"
PIP_SIZE = 0.0001
PAPER_STRATEGY_ID = "H6_SILVER_BULLET_HYBRID"
PAPER_WINDOW_START = "10:00"
PAPER_WINDOW_END = "11:00"
PAPER_TARGET_RR = 2.0
PAPER_BREAK_EVEN_AT_R = 1.0
PAPER_RISK_PCT = 0.5
PAPER_SPREAD_PIPS = 1.5
PAPER_SLIPPAGE_PIPS = 0.3
PAPER_LATENCY_MS = 200
PAPER_SPREAD_GUARD_PIPS = 3.0
PAPER_SESSION_CUTOFF = "11:30"
COOLDOWN_LOSS_STREAK = 3
COOLDOWN_HOURS = 48

RESULTS_DIR = Path("results")
LEDGER_OFFICIAL_PATH = RESULTS_DIR / "H6_SHADOW_LEDGER_OFFICIAL.csv"
LEDGER_DIAGNOSTIC_PATH = RESULTS_DIR / "H6_SHADOW_LEDGER_DIAGNOSTIC.csv"
RESEARCH_VS_SHADOW_OFFICIAL_PATH = RESULTS_DIR / "H6_RESEARCH_VS_SHADOW_OFFICIAL.csv"
RESEARCH_VS_SHADOW_DIAGNOSTIC_PATH = RESULTS_DIR / "H6_RESEARCH_VS_SHADOW_OBSERVED.csv"
CALIBRATION_PATH = RESULTS_DIR / "H6_SPREAD_SLIPPAGE_CALIBRATION.csv"
BLOCKED_PATH = RESULTS_DIR / "H6_SHADOW_BLOCKED_SIGNALS_LOG.csv"
DAILY_STATUS_PATH = RESULTS_DIR / "H6_FORWARD_ONLY_DAILY_STATUS.csv"

LEDGER_COLUMNS = [
    "provenance",
    "cost_mode",
    "session_date",
    "event_timestamp",
    "event_type",
    "status",
    "signal_id",
    "strategy_id",
    "pair",
    "direction",
    "setup_tag",
    "signal_time_ny",
    "entry_time_ny",
    "exit_time_ny",
    "block_reason",
    "block_details",
    "research_entry_price",
    "shadow_entry_price",
    "research_exit_price",
    "shadow_exit_price",
    "observed_spread_signal_pips",
    "observed_spread_entry_pips",
    "observed_spread_exit_pips",
    "applied_spread_pips",
    "applied_slippage_pips",
    "latency_ms",
    "fill_variance_pips",
    "pnl_r",
    "result",
    "notes",
]

RESEARCH_VS_SHADOW_COLUMNS = [
    "provenance",
    "cost_mode",
    "signal_id",
    "date",
    "strategy",
    "direction",
    "signal_time_ny",
    "setup_expected",
    "setup_executed",
    "research_entry",
    "shadow_entry",
    "entry_diff_pips",
    "research_exit",
    "shadow_exit",
    "exit_diff_pips",
    "research_pnl_r",
    "shadow_pnl_r",
    "variance_r",
    "edge_retention_pct",
    "primary_divergence_reason",
]

CALIBRATION_COLUMNS = [
    "provenance",
    "signal_id",
    "date",
    "direction",
    "signal_time_ny",
    "entry_time_ny",
    "exit_time_ny",
    "observed_spread_signal_pips",
    "observed_spread_entry_pips",
    "observed_spread_exit_pips",
    "applied_spread_pips",
    "applied_slippage_pips",
    "effective_total_cost_pips",
    "entry_diff_pips",
    "exit_diff_pips",
    "fill_variance_pips",
    "latency_ms",
    "liquidity_regime",
    "notes",
]

BLOCKED_COLUMNS = [
    "provenance",
    "signal_id",
    "date",
    "event_timestamp",
    "direction",
    "setup_tag",
    "block_reason",
    "block_details",
    "observed_spread_entry_pips",
    "latency_ms",
]

DAILY_STATUS_COLUMNS = [
    "date",
    "provenance",
    "status",
    "event_type",
    "signal_id",
    "block_reason",
    "observed_spread_pips",
    "pnl_official",
    "pnl_observed",
    "notes",
]

EVENT_ORDER = {
    "NO_SIGNAL": 0,
    "SIGNAL_DETECTED": 1,
    "SIGNAL_BLOCKED_NEWS": 2,
    "SIGNAL_BLOCKED_SPREAD": 3,
    "SIGNAL_BLOCKED_RISK": 4,
    "SIGNAL_ACCEPTED": 5,
    "PAPER_ENTRY": 6,
    "PAPER_EXIT": 7,
}


@dataclass(frozen=True)
class AuditArtifacts:
    ledger_rows_official: list[dict[str, Any]]
    ledger_rows_diagnostic: list[dict[str, Any]]
    research_vs_shadow_official: dict[str, Any] | None
    research_vs_shadow_diagnostic: dict[str, Any] | None
    calibration_row: dict[str, Any] | None
    blocked_row: dict[str, Any] | None
    daily_status_row: dict[str, Any]


def round_pips(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), 1)


def round_price(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), 5)


def coerce_frame(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path)
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    return frame[columns].copy()


def write_frame(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def reset_outputs() -> None:
    for path in (
        LEDGER_OFFICIAL_PATH,
        LEDGER_DIAGNOSTIC_PATH,
        RESEARCH_VS_SHADOW_OFFICIAL_PATH,
        RESEARCH_VS_SHADOW_DIAGNOSTIC_PATH,
        CALIBRATION_PATH,
        BLOCKED_PATH,
    ):
        if path.exists():
            path.unlink()


def paper_news_config() -> NewsConfig:
    return NewsConfig(enabled=True, file_path=AM_NEWS_FILE, impact_levels=("HIGH",))


def paper_strategy_params() -> dict[str, Any]:
    params = dict(strategy_module.default_params())
    params["target_rr"] = PAPER_TARGET_RR
    params["break_even_at_r"] = PAPER_BREAK_EVEN_AT_R
    return params


def shadow_engine_config_official() -> EngineConfig:
    return EngineConfig(
        pair=PAIR,
        risk_pct=PAPER_RISK_PCT,
        assumed_spread_pips=PAPER_SPREAD_PIPS,
        max_spread_pips=PAPER_SPREAD_GUARD_PIPS,
        slippage_pips=PAPER_SLIPPAGE_PIPS,
        max_trades_per_day=1,
        session_cutoff=PAPER_SESSION_CUTOFF,
        spread_opening_multiplier=1.0,
        spread_high_vol_multiplier=1.0,
        spread_late_session_multiplier=1.0,
        slippage_opening_multiplier=1.0,
        slippage_high_vol_multiplier=1.0,
        slippage_stop_multiplier=1.0,
        slippage_target_multiplier=1.0,
        slippage_late_session_multiplier=1.0,
        slippage_forced_close_multiplier=1.0,
        slippage_final_close_multiplier=1.0,
        stress_spread_multiplier=1.0,
        stress_slippage_multiplier=1.0,
        ambiguity_slippage_multiplier=1.0,
    )


def shadow_engine_config_diagnostic(observed_spread: float) -> EngineConfig:
    # Diagnostico: Spread observado + 0.0 slippage (para aislar impacto de asunciones)
    return EngineConfig(
        pair=PAIR,
        risk_pct=PAPER_RISK_PCT,
        assumed_spread_pips=observed_spread,
        max_spread_pips=PAPER_SPREAD_GUARD_PIPS,
        slippage_pips=0.0,
        max_trades_per_day=1,
        session_cutoff=PAPER_SESSION_CUTOFF,
        spread_opening_multiplier=1.0,
        spread_high_vol_multiplier=1.0,
        spread_late_session_multiplier=1.0,
        slippage_opening_multiplier=1.0,
        slippage_high_vol_multiplier=1.0,
        slippage_stop_multiplier=1.0,
        slippage_target_multiplier=1.0,
        slippage_late_session_multiplier=1.0,
        slippage_forced_close_multiplier=1.0,
        slippage_final_close_multiplier=1.0,
        stress_spread_multiplier=1.0,
        stress_slippage_multiplier=1.0,
        ambiguity_slippage_multiplier=1.0,
    )


def research_engine_config() -> EngineConfig:
    return EngineConfig(
        pair=PAIR,
        risk_pct=PAPER_RISK_PCT,
        assumed_spread_pips=0.000001,
        max_spread_pips=99.0,
        slippage_pips=0.0,
        max_trades_per_day=1,
        session_cutoff=PAPER_SESSION_CUTOFF,
        spread_opening_multiplier=1.0,
        spread_high_vol_multiplier=1.0,
        spread_late_session_multiplier=1.0,
        slippage_opening_multiplier=1.0,
        slippage_high_vol_multiplier=1.0,
        slippage_stop_multiplier=1.0,
        slippage_target_multiplier=1.0,
        slippage_late_session_multiplier=1.0,
        slippage_forced_close_multiplier=1.0,
        slippage_final_close_multiplier=1.0,
        stress_spread_multiplier=1.0,
        stress_slippage_multiplier=1.0,
        ambiguity_slippage_multiplier=1.0,
    )


def liquidity_regime(spread_pips: float | None) -> str:
    if spread_pips is None or not np.isfinite(spread_pips):
        return "UNKNOWN"
    if spread_pips < 1.0:
        return "GREEN"
    if spread_pips <= 1.5:
        return "NORMAL"
    if spread_pips <= 2.5:
        return "TENSE"
    return "PROHIBITED"


def format_setup(signal_row: pd.Series) -> str:
    parts = [
        str(signal_row.get("direction", "")).strip().lower(),
        str(signal_row.get("source_kind", "")).strip().lower(),
        str(signal_row.get("source_level_name", "")).strip().lower(),
    ]
    return "|".join(part for part in parts if part)


def within_paper_window(ts: pd.Timestamp) -> bool:
    minute_value = ts.hour * 60 + ts.minute
    return 10 * 60 <= minute_value < 11 * 60


def filter_paper_signals(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()
    frame = signals.copy()
    frame["signal_time"] = pd.to_datetime(frame["signal_time"], utc=True).dt.tz_convert(NY_TZ)
    return frame.loc[frame["signal_time"].apply(within_paper_window)].reset_index(drop=True)


def apply_paper_window_to_annotated(annotated: pd.DataFrame) -> pd.DataFrame:
    frame = annotated.copy()
    if "els_signal" not in frame.columns:
        return frame
    mask = pd.Series(frame.index.map(within_paper_window), index=frame.index)
    frame.loc[~mask, "els_signal"] = False
    return frame


def load_day_context(target_date_str: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    target_dt = pd.Timestamp(target_date_str, tz=NY_TZ)
    context_start = (target_dt - timedelta(days=3)).normalize()
    context_end = (target_dt + timedelta(days=1)).normalize() - timedelta(minutes=1)

    package = load_high_precision_package(PAIR, Path("data_precision/dukascopy"))
    full_mid = package["mid"].loc[context_start:context_end].copy()
    if full_mid.empty:
        raise RuntimeError(f"No hay data suficiente para {target_date_str}.")

    day_m1 = {side: src.loc[target_date_str].copy() for side, src in package.items()}
    m3_full = add_ict_primitives(prepare_common_frame(full_mid, target_timeframe="M3"))
    m5_full = add_ict_primitives(prepare_common_frame(full_mid, target_timeframe="M5"))
    m3_day = m3_full.loc[target_date_str].copy()
    if m3_day.empty:
        raise RuntimeError(f"No hay frame M3 para {target_date_str}.")
    return m3_day, day_m1, m5_full, full_mid


def build_observed_spread_maps(day_m1: dict[str, pd.DataFrame], index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    bid_exec = resample_ohlcv_to_timeframe(day_m1["bid"], "M3").loc[index].copy()
    ask_exec = resample_ohlcv_to_timeframe(day_m1["ask"], "M3").loc[index].copy()
    open_spreads = pd.Series(
        [actual_spread_pips(PAIR, float(bid_exec["open"].iat[i]), float(ask_exec["open"].iat[i])) for i in range(len(index))],
        index=index,
    )
    close_spreads = pd.Series(
        [actual_spread_pips(PAIR, float(bid_exec["close"].iat[i]), float(ask_exec["close"].iat[i])) for i in range(len(index))],
        index=index,
    )
    return open_spreads, close_spreads


def summarize_trades(
    annotated: pd.DataFrame,
    params: dict[str, Any],
    engine_config: EngineConfig,
    news_result: Any,
) -> pd.DataFrame:
    paper_annotated = apply_paper_window_to_annotated(annotated)
    news_block = build_entry_block(entry_open_index(paper_annotated.index), news_result.events, paper_news_config())
    result = run_backtest(
        strategy_module=strategy_module,
        frame=paper_annotated,
        params=params,
        engine_config=engine_config,
        news_block=news_block,
        news_filter_used=news_result.enabled,
        news_events=news_result.events,
        news_settings=paper_news_config(),
    )
    _summary, trades_export, _monthly, _yearly, _equity = summarize_result(
        strategy_module.NAME,
        result.trades,
        result.equity_curve,
        params,
        news_result.enabled,
        100000,
        None,
        costs_used={"execution_mode": engine_config.execution_mode, "cost_profile": engine_config.cost_profile},
        timeframe="M3",
        schedule_used=schedule_used(),
    )
    return trades_export.copy()


def cooldown_block(target_date_str: str, ledger: pd.DataFrame) -> tuple[bool, str, str]:
    if ledger.empty:
        return False, "", ""
    exits = ledger.loc[ledger["event_type"] == "PAPER_EXIT"].copy()
    if exits.empty:
        return False, "", ""
    exits["event_timestamp"] = pd.to_datetime(exits["event_timestamp"], utc=True, errors="coerce").dt.tz_convert(NY_TZ)
    exits = exits.dropna(subset=["event_timestamp"]).sort_values("event_timestamp")
    target_day = pd.Timestamp(target_date_str, tz=NY_TZ)
    exits = exits.loc[exits["event_timestamp"] < target_day]
    if exits.empty:
        return False, "", ""

    consecutive_losses: list[pd.Series] = []
    for row in reversed(list(exits.itertuples(index=False))):
        if str(row.result).lower() == "loss":
            consecutive_losses.append(pd.Series(row._asdict()))
            if len(consecutive_losses) >= COOLDOWN_LOSS_STREAK:
                break
            continue
        break
    if len(consecutive_losses) < COOLDOWN_LOSS_STREAK:
        return False, "", ""

    last_loss_time = pd.Timestamp(consecutive_losses[0]["event_timestamp"])
    cooldown_until = last_loss_time + pd.Timedelta(hours=COOLDOWN_HOURS)
    if target_day < cooldown_until:
        return True, "cooldown_3_losses_48h", f"cooldown_until={cooldown_until.strftime('%Y-%m-%d %H:%M:%S%z')}"
    return False, "", ""


def remove_date_rows(frame: pd.DataFrame, date_value: str, date_column: str) -> pd.DataFrame:
    if frame.empty or date_column not in frame.columns:
        return frame
    return frame.loc[frame[date_column].astype(str) != date_value].copy()


def make_signal_id(session_date: str, signal_time: pd.Timestamp) -> str:
    return f"H6_{session_date}_{signal_time.strftime('%H%M')}"


def build_signal_row(signals: pd.DataFrame, signal_time: pd.Timestamp) -> pd.Series:
    row = signals.loc[signals["signal_time"] == signal_time]
    if row.empty:
        raise RuntimeError(f"No se pudo localizar la senial {signal_time} en el signal log.")
    return row.iloc[0]


def gate_signal(
    annotated: pd.DataFrame,
    signal_row: pd.Series,
    day_m1: dict[str, pd.DataFrame],
    news_result: Any,
    observed_open_spreads: pd.Series,
    observed_close_spreads: pd.Series,
) -> dict[str, Any]:
    signal_time = pd.Timestamp(signal_row["signal_time"]).tz_convert(NY_TZ)
    signal_index = annotated.index.get_loc(signal_time)
    if isinstance(signal_index, slice):
        signal_index = signal_index.start
    entry_index = int(signal_index) + 1
    signal_spread = round_pips(float(observed_close_spreads.loc[signal_time]))

    if entry_index >= len(annotated):
        return {
            "blocked": True,
            "block_reason": "no_fill_next_bar",
            "block_details": "signal_on_last_bar",
            "signal_spread": signal_spread,
            "entry_spread": None,
            "entry_time": signal_time,
            "exit_spread": None,
        }

    entry_bar_time = annotated.index[entry_index]
    entry_open_times = entry_open_index(annotated.index)
    news_details = build_news_guard_details(entry_open_times, news_result.events, paper_news_config())
    entry_spread = round_pips(float(observed_open_spreads.loc[entry_bar_time]))
    if bool(news_details["entry_blocked"].iat[entry_index]):
        return {
            "blocked": True,
            "block_reason": "high_impact_news_guard",
            "block_details": str(news_details["entry_event_name"].iat[entry_index]),
            "signal_spread": signal_spread,
            "entry_spread": entry_spread,
            "entry_time": entry_open_times[entry_index],
            "exit_spread": None,
        }
    if entry_spread is not None and entry_spread > PAPER_SPREAD_GUARD_PIPS:
        return {
            "blocked": True,
            "block_reason": "spread_guard_gt_3_0",
            "block_details": f"observed_spread={entry_spread}",
            "signal_spread": signal_spread,
            "entry_spread": entry_spread,
            "entry_time": entry_open_times[entry_index],
            "exit_spread": None,
        }
    return {
        "blocked": False,
        "block_reason": "",
        "block_details": "",
        "signal_spread": signal_spread,
        "entry_spread": entry_spread,
        "entry_time": entry_open_times[entry_index],
        "exit_spread": None,
    }


def trade_result_label(pnl_r: float) -> str:
    if pnl_r > 0:
        return "win"
    if pnl_r < 0:
        return "loss"
    return "breakeven"


def build_artifacts_for_date(target_date_str: str, ledger: pd.DataFrame) -> AuditArtifacts:
    news_result = require_operational_news(PAIR, paper_news_config(), context="h6_paper_shadow")
    m3_day, day_m1, m5_full, _full_mid = load_day_context(target_date_str)
    annotated, raw_signals = annotate_post_news_external_liquidity_shift_frame(
        m3_day,
        m5_full,
        news_events=news_result.events,
        news_config=paper_news_config(),
    )
    observed_open_spreads, observed_close_spreads = build_observed_spread_maps(day_m1, annotated.index)
    signals = filter_paper_signals(raw_signals)

    if signals.empty:
        provenance = "FORWARD" if target_date_str >= "2026-04-20" else "BACKFILL"
        
        def build_no_signal_row(mode):
            return {
                "provenance": provenance,
                "cost_mode": mode,
                "date": target_date_str,
                "session_date": target_date_str,
                "event_timestamp": pd.Timestamp(f"{target_date_str} 17:00:00", tz=NY_TZ).strftime("%Y-%m-%d %H:%M:%S%z"),
                "event_type": "NO_SIGNAL",
                "status": "NO_SIGNAL",
                "signal_id": "",
                "strategy_id": PAPER_STRATEGY_ID,
                "pair": PAIR,
                "direction": "",
                "setup_tag": "",
                "signal_time_ny": "",
                "entry_time_ny": "",
                "exit_time_ny": "",
                "block_reason": "",
                "block_details": "",
                "research_entry_price": "",
                "shadow_entry_price": "",
                "research_exit_price": "",
                "shadow_exit_price": "",
                "observed_spread_signal_pips": "",
                "observed_spread_entry_pips": "",
                "observed_spread_exit_pips": "",
                "applied_spread_pips": "",
                "applied_slippage_pips": "",
                "latency_ms": PAPER_LATENCY_MS,
                "fill_variance_pips": "",
                "pnl_r": "",
                "result": "",
                "notes": "paper_window_10_11_no_signal",
            }

        daily_status_row = build_no_signal_row("OFFICIAL")
        daily_status_row["pnl_official"] = 0.0
        daily_status_row["pnl_observed"] = 0.0

        return AuditArtifacts(
            ledger_rows_official=[build_no_signal_row("OFFICIAL")],
            ledger_rows_diagnostic=[build_no_signal_row("OBSERVED")],
            research_vs_shadow_official=None,
            research_vs_shadow_diagnostic=None,
            calibration_row=None,
            blocked_row=None,
            daily_status_row=daily_status_row,
        )

    signal_row = signals.sort_values("signal_time").iloc[0]
    signal_time = pd.Timestamp(signal_row["signal_time"]).tz_convert(NY_TZ)
    signal_id = make_signal_id(target_date_str, signal_time)
    setup_tag = format_setup(signal_row)
    direction = str(signal_row["direction"]).strip().lower()

    cooldown_active, cooldown_reason, cooldown_details = cooldown_block(target_date_str, ledger)
    gate_details = gate_signal(annotated, signal_row, day_m1, news_result, observed_open_spreads, observed_close_spreads)

    provenance = "FORWARD" if target_date_str >= "2026-04-20" else "BACKFILL"

    detected_row = {
        "provenance": provenance,
        "session_date": target_date_str,
        "event_timestamp": signal_time.strftime("%Y-%m-%d %H:%M:%S%z"),
        "event_type": "SIGNAL_DETECTED",
        "status": "DETECTED",
        "signal_id": signal_id,
        "strategy_id": PAPER_STRATEGY_ID,
        "pair": PAIR,
        "direction": direction,
        "setup_tag": setup_tag,
        "signal_time_ny": signal_time.strftime("%Y-%m-%d %H:%M:%S"),
        "entry_time_ny": "",
        "exit_time_ny": "",
        "block_reason": "",
        "block_details": "",
        "research_entry_price": "",
        "shadow_entry_price": "",
        "research_exit_price": "",
        "shadow_exit_price": "",
        "observed_spread_signal_pips": gate_details["signal_spread"],
        "observed_spread_entry_pips": gate_details["entry_spread"],
        "observed_spread_exit_pips": "",
        "applied_spread_pips": PAPER_SPREAD_PIPS,
        "applied_slippage_pips": PAPER_SLIPPAGE_PIPS,
        "latency_ms": PAPER_LATENCY_MS,
        "fill_variance_pips": None,
        "pnl_r": None,
        "result": None,
        "notes": "paper_freeze_10_11_rr2.0_be1.0",
    }
    ledger_rows = [detected_row]

    if cooldown_active:
        blocked_row_official = {
            **detected_row,
            "cost_mode": "OFFICIAL",
            "event_timestamp": signal_time.strftime("%Y-%m-%d %H:%M:%S%z"),
            "event_type": "SIGNAL_BLOCKED_RISK",
            "status": "BLOCKED",
            "block_reason": cooldown_reason,
            "block_details": cooldown_details,
            "notes": "paper_risk_protocol",
        }
        blocked_row_diagnostic = {
            **blocked_row_official,
            "cost_mode": "OBSERVED",
        }
        daily_status_row = {
            "date": target_date_str,
            "provenance": provenance,
            "status": "BLOCKED",
            "event_type": "SIGNAL_BLOCKED_RISK",
            "signal_id": signal_id,
            "block_reason": cooldown_reason,
            "observed_spread_pips": gate_details["entry_spread"],
            "pnl_official": 0.0,
            "pnl_observed": 0.0,
            "notes": "cooldown_active",
        }
        return AuditArtifacts(
            ledger_rows_official=[blocked_row_official],
            ledger_rows_diagnostic=[blocked_row_diagnostic],
            research_vs_shadow_official=None,
            research_vs_shadow_diagnostic=None,
            calibration_row=None,
            blocked_row={
                "provenance": provenance,
                "signal_id": signal_id,
                "date": target_date_str,
                "event_timestamp": signal_time.strftime("%Y-%m-%d %H:%M:%S%z"),
                "direction": direction,
                "setup_tag": setup_tag,
                "block_reason": cooldown_reason,
                "block_details": cooldown_details,
                "observed_spread_entry_pips": gate_details["entry_spread"],
                "latency_ms": PAPER_LATENCY_MS,
            },
            daily_status_row=daily_status_row,
        )

    if gate_details["blocked"]:
        blocked_event = "SIGNAL_BLOCKED_NEWS" if gate_details["block_reason"] == "high_impact_news_guard" else "SIGNAL_BLOCKED_SPREAD"
        blocked_row_official = {
            **detected_row,
            "cost_mode": "OFFICIAL",
            "event_type": blocked_event,
            "status": "BLOCKED",
            "entry_time_ny": pd.Timestamp(gate_details["entry_time"]).strftime("%Y-%m-%d %H:%M:%S"),
            "block_reason": gate_details["block_reason"],
            "block_details": gate_details["block_details"],
            "notes": "paper_gate_block",
        }
        blocked_row_diagnostic = {
            **blocked_row_official,
            "cost_mode": "OBSERVED",
        }
        daily_status_row = {
            "date": target_date_str,
            "provenance": provenance,
            "status": "BLOCKED",
            "event_type": blocked_event,
            "signal_id": signal_id,
            "block_reason": gate_details["block_reason"],
            "observed_spread_pips": gate_details["entry_spread"],
            "pnl_official": 0.0,
            "pnl_observed": 0.0,
            "notes": "gate_blocked",
        }
        return AuditArtifacts(
            ledger_rows_official=[blocked_row_official],
            ledger_rows_diagnostic=[blocked_row_diagnostic],
            research_vs_shadow_official=None,
            research_vs_shadow_diagnostic=None,
            calibration_row=None,
            blocked_row={
                "provenance": provenance,
                "signal_id": signal_id,
                "date": target_date_str,
                "event_timestamp": signal_time.strftime("%Y-%m-%d %H:%M:%S%z"),
                "direction": direction,
                "setup_tag": setup_tag,
                "block_reason": gate_details["block_reason"],
                "block_details": gate_details["block_details"],
                "observed_spread_entry_pips": gate_details["entry_spread"],
                "latency_ms": PAPER_LATENCY_MS,
            },
            daily_status_row=daily_status_row,
        )

    paper_params = paper_strategy_params()
    shadow_trades_official = summarize_trades(annotated, paper_params, shadow_engine_config_official(), news_result)
    shadow_trades_diagnostic = summarize_trades(
        annotated, paper_params, shadow_engine_config_diagnostic(gate_details["entry_spread"]), news_result
    )
    research_trades = summarize_trades(annotated, paper_params, research_engine_config(), news_result)

    if shadow_trades_official.empty or research_trades.empty:
        blocked_row_official = {
            **detected_row,
            "cost_mode": "OFFICIAL",
            "event_type": "SIGNAL_BLOCKED_RISK",
            "status": "BLOCKED",
            "block_reason": "no_fill_after_validation",
            "block_details": "runner_returned_empty_trade_export",
            "notes": "paper_runner_integrity",
        }
        blocked_row_diagnostic = {
            **blocked_row_official,
            "cost_mode": "OBSERVED",
        }
        daily_status_row = {
            "date": target_date_str,
            "provenance": provenance,
            "status": "BLOCKED",
            "event_type": "SIGNAL_BLOCKED_RISK",
            "signal_id": signal_id,
            "block_reason": "no_fill_after_validation",
            "observed_spread_pips": gate_details["entry_spread"],
            "pnl_official": 0.0,
            "pnl_observed": 0.0,
            "notes": "runner_integrity_block",
        }
        return AuditArtifacts(
            ledger_rows_official=[blocked_row_official],
            ledger_rows_diagnostic=[blocked_row_diagnostic],
            research_vs_shadow_official=None,
            research_vs_shadow_diagnostic=None,
            calibration_row=None,
            blocked_row={
                "provenance": provenance,
                "signal_id": signal_id,
                "date": target_date_str,
                "event_timestamp": signal_time.strftime("%Y-%m-%d %H:%M:%S%z"),
                "direction": direction,
                "setup_tag": setup_tag,
                "block_reason": "no_fill_after_validation",
                "block_details": "runner_returned_empty_trade_export",
                "observed_spread_entry_pips": gate_details["entry_spread"],
                "latency_ms": PAPER_LATENCY_MS,
            },
            daily_status_row=daily_status_row,
        )

    shadow_trade_official = shadow_trades_official.iloc[0]
    shadow_trade_diagnostic = shadow_trades_diagnostic.iloc[0]
    research_trade = research_trades.iloc[0]

    shadow_entry_official = float(shadow_trade_official["entry_price"])
    shadow_entry_diagnostic = float(shadow_trade_diagnostic["entry_price"])
    research_entry = float(research_trade["entry_price"])

    shadow_exit_official = float(shadow_trade_official["exit_price"])
    shadow_exit_diagnostic = float(shadow_trade_diagnostic["exit_price"])
    research_exit = float(research_trade["exit_price"])

    entry_diff_pips_official = (
        (shadow_entry_official - research_entry) / PIP_SIZE if direction == "long" else (research_entry - shadow_entry_official) / PIP_SIZE
    )
    entry_diff_pips_diagnostic = (
        (shadow_entry_diagnostic - research_entry) / PIP_SIZE
        if direction == "long"
        else (research_entry - shadow_entry_diagnostic) / PIP_SIZE
    )

    exit_diff_pips_official = (
        (research_exit - shadow_exit_official) / PIP_SIZE if direction == "long" else (shadow_exit_official - research_exit) / PIP_SIZE
    )
    exit_diff_pips_diagnostic = (
        (research_exit - shadow_exit_diagnostic) / PIP_SIZE
        if direction == "long"
        else (shadow_exit_diagnostic - research_exit) / PIP_SIZE
    )

    shadow_pnl_r_official = float(shadow_trade_official["pnl_r"])
    shadow_pnl_r_diagnostic = float(shadow_trade_diagnostic["pnl_r"])
    research_pnl_r = float(research_trade["pnl_r"])

    exit_time = pd.Timestamp(shadow_trade_official["exit_time_ny"], tz=NY_TZ)
    observed_exit_spread = round_pips(float(observed_close_spreads.loc[exit_time]))

    def build_ledger_set(shadow_trade, shadow_entry, shadow_exit, pnl_r, applied_spread, applied_slippage, notes_suffix, cost_mode):
        accepted_row = {
            **detected_row,
            "cost_mode": cost_mode,
            "event_timestamp": signal_time.strftime("%Y-%m-%d %H:%M:%S%z"),
            "event_type": "SIGNAL_ACCEPTED",
            "status": "ACCEPTED",
            "entry_time_ny": str(shadow_trade["entry_time_ny"]),
            "research_entry_price": round_price(research_entry),
            "shadow_entry_price": round_price(shadow_entry),
            "applied_spread_pips": applied_spread,
            "applied_slippage_pips": applied_slippage,
            "notes": f"paper_signal_accepted_{notes_suffix}",
        }
        entry_row = {
            **accepted_row,
            "event_type": "PAPER_ENTRY",
            "status": "FILLED",
            "event_timestamp": signal_time.strftime("%Y-%m-%d %H:%M:%S%z"),
            "observed_spread_entry_pips": gate_details["entry_spread"],
            "notes": f"paper_entry_recorded_{notes_suffix}",
        }
        exit_row = {
            **accepted_row,
            "event_type": "PAPER_EXIT",
            "status": "FILLED",
            "event_timestamp": exit_time.strftime("%Y-%m-%d %H:%M:%S%z"),
            "exit_time_ny": str(shadow_trade["exit_time_ny"]),
            "research_exit_price": round_price(research_exit),
            "shadow_exit_price": round_price(shadow_exit),
            "observed_spread_exit_pips": observed_exit_spread,
            "pnl_r": round(float(pnl_r), 6),
            "result": trade_result_label(pnl_r),
            "notes": f"{shadow_trade['exit_reason']}_{notes_suffix}",
        }
        return [accepted_row, entry_row, exit_row]

    ledger_rows_official = [dict(detected_row, cost_mode="OFFICIAL")] + build_ledger_set(
        shadow_trade_official,
        shadow_entry_official,
        shadow_exit_official,
        shadow_pnl_r_official,
        PAPER_SPREAD_PIPS,
        PAPER_SLIPPAGE_PIPS,
        "official",
        "OFFICIAL",
    )
    ledger_rows_diagnostic = [dict(detected_row, cost_mode="OBSERVED")] + build_ledger_set(
        shadow_trade_diagnostic,
        shadow_entry_diagnostic,
        shadow_exit_diagnostic,
        shadow_pnl_r_diagnostic,
        gate_details["entry_spread"],
        0.0,
        "diagnostic",
        "OBSERVED",
    )

    def build_cvs_row(shadow_trade, shadow_entry, shadow_exit, pnl_r, entry_diff, exit_diff, cost_mode):
        return {
            "provenance": provenance,
            "cost_mode": cost_mode,
            "signal_id": signal_id,
            "date": target_date_str,
            "strategy": PAPER_STRATEGY_ID,
            "direction": direction,
            "signal_time_ny": str(shadow_trade["signal_time_ny"]),
            "setup_expected": setup_tag,
            "setup_executed": setup_tag,
            "research_entry": round_price(research_entry),
            "shadow_entry": round_price(shadow_entry),
            "entry_diff_pips": round_pips(entry_diff),
            "research_exit": round_price(research_exit),
            "shadow_exit": round_price(shadow_exit),
            "exit_diff_pips": round_pips(exit_diff),
            "research_pnl_r": round(float(research_pnl_r), 6),
            "shadow_pnl_r": round(float(pnl_r), 6),
            "variance_r": round(float(pnl_r - research_pnl_r), 6),
            "edge_retention_pct": round(float(pnl_r / research_pnl_r * 100.0), 2) if research_pnl_r > 0 else "",
            "primary_divergence_reason": "friction_regime_impact",
        }

    research_vs_shadow_official = build_cvs_row(
        shadow_trade_official,
        shadow_entry_official,
        shadow_exit_official,
        shadow_pnl_r_official,
        entry_diff_pips_official,
        exit_diff_pips_official,
        "OFFICIAL",
    )
    research_vs_shadow_diagnostic = build_cvs_row(
        shadow_trade_diagnostic,
        shadow_entry_diagnostic,
        shadow_exit_diagnostic,
        shadow_pnl_r_diagnostic,
        entry_diff_pips_diagnostic,
        exit_diff_pips_diagnostic,
        "OBSERVED",
    )

    calibration_row = {
        "provenance": provenance,
        "signal_id": signal_id,
        "date": target_date_str,
        "direction": direction,
        "signal_time_ny": str(shadow_trade_official["signal_time_ny"]),
        "entry_time_ny": str(shadow_trade_official["entry_time_ny"]),
        "exit_time_ny": str(shadow_trade_official["exit_time_ny"]),
        "observed_spread_signal_pips": gate_details["signal_spread"],
        "observed_spread_entry_pips": gate_details["entry_spread"],
        "observed_spread_exit_pips": observed_exit_spread,
        "applied_spread_pips": PAPER_SPREAD_PIPS,
        "applied_slippage_pips": PAPER_SLIPPAGE_PIPS,
        "effective_total_cost_pips": round_pips(PAPER_SPREAD_PIPS + PAPER_SLIPPAGE_PIPS),
        "entry_diff_pips": round_pips(entry_diff_pips_official),
        "exit_diff_pips": round_pips(exit_diff_pips_official),
        "fill_variance_pips": round_pips(entry_diff_pips_official),
        "latency_ms": PAPER_LATENCY_MS,
        "liquidity_regime": liquidity_regime(gate_details["entry_spread"]),
        "notes": "official_calibration_metrics",
    }

    daily_status_row = {
        "date": target_date_str,
        "provenance": provenance,
        "status": "ACCEPTED",
        "event_type": "SIGNAL_ACCEPTED",
        "signal_id": signal_id,
        "block_reason": "",
        "observed_spread_pips": gate_details["signal_spread"],
        "pnl_official": round(float(shadow_pnl_r_official), 6),
        "pnl_observed": round(float(shadow_pnl_r_diagnostic), 6),
        "notes": "",
    }

    return AuditArtifacts(
        ledger_rows_official=ledger_rows_official,
        ledger_rows_diagnostic=ledger_rows_diagnostic,
        research_vs_shadow_official=research_vs_shadow_official,
        research_vs_shadow_diagnostic=research_vs_shadow_diagnostic,
        calibration_row=calibration_row,
        blocked_row=None,
        daily_status_row=daily_status_row,
    )


def persist_artifacts(target_date_str: str, artifacts: AuditArtifacts) -> None:
    def update_ledger(path, rows):
        ledger = remove_date_rows(coerce_frame(path, LEDGER_COLUMNS), target_date_str, "session_date")
        ledger = pd.concat([ledger, pd.DataFrame(rows, columns=LEDGER_COLUMNS)], ignore_index=True)
        ledger["_event_order"] = ledger["event_type"].map(EVENT_ORDER).fillna(99)
        ledger = ledger.sort_values(["session_date", "event_timestamp", "_event_order"], kind="stable").drop(columns="_event_order")
        write_frame(path, ledger.to_dict("records"), LEDGER_COLUMNS)

    update_ledger(LEDGER_OFFICIAL_PATH, artifacts.ledger_rows_official)
    update_ledger(LEDGER_DIAGNOSTIC_PATH, artifacts.ledger_rows_diagnostic)

    def update_comparison(path, row):
        frame = remove_date_rows(coerce_frame(path, RESEARCH_VS_SHADOW_COLUMNS), target_date_str, "date")
        if row is not None:
            frame = pd.concat([frame, pd.DataFrame([row], columns=RESEARCH_VS_SHADOW_COLUMNS)], ignore_index=True)
        frame = frame.sort_values(["date", "signal_time_ny"], kind="stable")
        write_frame(path, frame.to_dict("records"), RESEARCH_VS_SHADOW_COLUMNS)

    update_comparison(RESEARCH_VS_SHADOW_OFFICIAL_PATH, artifacts.research_vs_shadow_official)
    update_comparison(RESEARCH_VS_SHADOW_DIAGNOSTIC_PATH, artifacts.research_vs_shadow_diagnostic)

    calibration = remove_date_rows(coerce_frame(CALIBRATION_PATH, CALIBRATION_COLUMNS), target_date_str, "date")
    if artifacts.calibration_row is not None:
        calibration = pd.concat([calibration, pd.DataFrame([artifacts.calibration_row], columns=CALIBRATION_COLUMNS)], ignore_index=True)
    calibration = calibration.sort_values(["date", "signal_time_ny"], kind="stable")
    write_frame(CALIBRATION_PATH, calibration.to_dict("records"), CALIBRATION_COLUMNS)

    blocked = remove_date_rows(coerce_frame(BLOCKED_PATH, BLOCKED_COLUMNS), target_date_str, "date")
    if artifacts.blocked_row is not None:
        blocked = pd.concat([blocked, pd.DataFrame([artifacts.blocked_row], columns=BLOCKED_COLUMNS)], ignore_index=True)
    blocked = blocked.sort_values(["date", "event_timestamp"], kind="stable")
    if blocked.empty:
        if BLOCKED_PATH.exists():
            BLOCKED_PATH.unlink()
    else:
        write_frame(BLOCKED_PATH, blocked.to_dict("records"), BLOCKED_COLUMNS)

    status_frame = remove_date_rows(coerce_frame(DAILY_STATUS_PATH, DAILY_STATUS_COLUMNS), target_date_str, "date")
    status_frame = pd.concat([status_frame, pd.DataFrame([artifacts.daily_status_row], columns=DAILY_STATUS_COLUMNS)], ignore_index=True)
    status_frame = status_frame.sort_values(["date"], kind="stable")
    write_frame(DAILY_STATUS_PATH, status_frame.to_dict("records"), DAILY_STATUS_COLUMNS)


def audit_day(target_date_str: str) -> None:
    ledger = coerce_frame(LEDGER_OFFICIAL_PATH, LEDGER_COLUMNS)
    artifacts = build_artifacts_for_date(target_date_str, ledger)
    persist_artifacts(target_date_str, artifacts)
    event_types = ",".join(row["event_type"] for row in artifacts.ledger_rows_official)
    print(f"[OK] {target_date_str} -> {event_types}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auditoria paper-shadow congelada para H6.")
    parser.add_argument("dates", nargs="*", help="Fechas NY en formato YYYY-MM-DD.")
    parser.add_argument("--reset", action="store_true", help="Reinicia los CSV canonicos antes de procesar.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.reset:
        reset_outputs()
    for target_date_str in args.dates:
        audit_day(target_date_str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
