from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_lab.config import (
    DEFAULT_DATA_DIRS,
    DEFAULT_HIGH_PRECISION_PREPARED_DIR,
    EngineConfig,
    INITIAL_CAPITAL,
    NY_TZ,
    with_execution_mode,
)
from research_lab.data_loader import (
    adx,
    atr,
    ema,
    fx_market_mask,
    load_high_precision_package,
    load_prepared_ohlcv,
    rsi,
    slice_high_precision_package_to_frame,
    validate_price_frame,
)
from research_lab.engine import run_backtest
from research_lab.report import summarize_result, sync_visible_chatgpt
from research_lab.strategies.common import stratified_sample_combinations


RESULTS_DIR = Path("results") / "edge_search_m5"
MODES = ("normal_mode", "conservative_mode", "high_precision_mode")
PERIODS = {
    "development_2020_2023": ("2020-01-01", "2023-12-31"),
    "validation_2024": ("2024-01-01", "2024-12-31"),
    "holdout_2025": ("2025-01-01", "2025-12-31"),
}
ENTRY_START_MINUTE = 11 * 60
LAST_ENTRY_MINUTE = 18 * 60 + 30
FORCE_CLOSE_MINUTE = 19 * 60
PIP = 0.0001

MIN_TOTAL_TRADES = {
    "development_2020_2023": 80,
    "validation_2024": 18,
    "holdout_2025": 18,
}
MIN_TRADES_PER_MONTH = {
    "development_2020_2023": 2.0,
    "validation_2024": 1.5,
    "holdout_2025": 1.5,
}
MAX_NEGATIVE_YEARS_DEVELOPMENT = 2
MAX_SHARE_BEST_YEAR = 0.60
COLLAPSE_PF_RATIO = 0.80
COLLAPSE_EXPECTANCY_RATIO = 0.70

MANAGEMENT_PACKAGES: dict[str, dict[str, Any]] = {
    "A": {
        "label": "Management A",
        "stop_kind": "atr",
        "stop_atr": 1.0,
        "target_rr": 1.5,
        "break_even_at_r": None,
        "max_hold_bars": 8,
        "spread_max_pips": 1.5,
    },
    "B": {
        "label": "Management B",
        "stop_kind": "atr",
        "stop_atr": 1.2,
        "target_rr": 2.0,
        "break_even_at_r": None,
        "max_hold_bars": 10,
        "spread_max_pips": 1.5,
    },
    "C": {
        "label": "Management C",
        "stop_kind": "atr",
        "stop_atr": 1.0,
        "target_rr": 1.5,
        "break_even_at_r": 1.2,
        "max_hold_bars": 8,
        "spread_max_pips": 1.5,
    },
    "D": {
        "label": "Management D",
        "stop_kind": "structural",
        "stop_atr": None,
        "target_rr": 1.2,
        "break_even_at_r": None,
        "max_hold_bars": 6,
        "spread_max_pips": 1.2,
    },
}

MODEL_DEFINITIONS = {
    "post_spike_mean_reversion": "Shock intradia objetivo contra impulso inmediato: vela previa extrema, ADX bajo y entrada en la apertura siguiente buscando reversión corta.",
    "same_session_ema_pullback_continuation": "Continuacion intradia con EMA20/EMA50: sesgo por EMAs, expansion previa, pullback controlado y cierre de recuperacion sobre EMA20.",
    "vwap_stretch_reversion": "Reversion a VWAP de sesion 11:00 NY cuando el precio se estira multiples ATR y el ADX sigue bajo.",
    "nr7_compression_expansion_continuation": "Continuacion de compresion: NR7 previa, ATR corto comprimido, sesgo por EMA20/EMA50 y ruptura a favor.",
    "rolling_range_reclaim_reversal": "Reversal de extremo corto: nuevo minimo/maximo de ventana corta y cierre de regreso dentro del rango previo.",
    "vwap_trend_pullback_continuation": "Continuacion sobre VWAP de sesion: sesgo alcista/bajista y pullback a VWAP con reclaim objetivo.",
    "open_close_imbalance_reversion": "Fade de desequilibrio secuencial: tres cierres consecutivos direccionales con exceso de rango y ADX bajo.",
    "inside_bar_continuation": "Continuacion despues de inside bar con tendencia EMA20/EMA50 y rango contenido.",
    "atr_exhaustion_fade": "Fade de expansion extrema por ATR con cierre en decil del rango y ADX bajo.",
    "midday_range_breakout_filtered": "Unico breakout de la tanda: ruptura del rango 11:00-13:00 con tendencia EMA y ADX ascendente.",
}


def build_output_root() -> Path:
    timestamp = pd.Timestamp.now(tz=NY_TZ).strftime("%Y%m%d_%H%M%S")
    root = RESULTS_DIR / f"{timestamp}_protocol_search_m5_edge"
    root.mkdir(parents=True, exist_ok=True)
    return root


def minute_of(ts: pd.Timestamp) -> int:
    return ts.hour * 60 + ts.minute


def bar_close_allowed(ts: pd.Timestamp, start: int, end: int) -> bool:
    value = minute_of(ts)
    return start <= value <= min(end, LAST_ENTRY_MINUTE)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _resample_to_5min(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.resample("5min", label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )


def session_vwap_11_19(frame: pd.DataFrame) -> pd.Series:
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    weights = frame["volume"].replace(0.0, 1.0)
    values: list[pd.Series] = []
    for _, chunk in frame.groupby(frame.index.date, sort=True):
        minute_values = chunk.index.hour * 60 + chunk.index.minute
        active = (minute_values >= ENTRY_START_MINUTE) & (minute_values <= FORCE_CLOSE_MINUTE)
        vwap = pd.Series(np.nan, index=chunk.index, dtype=float)
        if np.any(active):
            active_index = chunk.index[active]
            cum_pv = (typical.loc[active_index] * weights.loc[active_index]).cumsum()
            cum_vol = weights.loc[active_index].cumsum().replace(0.0, np.nan)
            vwap.loc[active_index] = cum_pv / cum_vol
        values.append(vwap)
    return pd.concat(values).sort_index()


def fill_midday_range(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["midday_range_high"] = np.nan
    frame["midday_range_low"] = np.nan
    frame["midday_range_ready"] = False
    for _, chunk in frame.groupby(frame.index.date, sort=True):
        minute_values = chunk.index.hour * 60 + chunk.index.minute
        in_window = (minute_values >= ENTRY_START_MINUTE) & (minute_values < 13 * 60)
        if not np.any(in_window):
            continue
        high_value = float(chunk.loc[in_window, "high"].max())
        low_value = float(chunk.loc[in_window, "low"].min())
        ready_mask = minute_values >= 13 * 60
        if np.any(ready_mask):
            ready_index = chunk.index[ready_mask]
            frame.loc[ready_index, "midday_range_high"] = high_value
            frame.loc[ready_index, "midday_range_low"] = low_value
            frame.loc[ready_index, "midday_range_ready"] = True
    return frame


def prepare_common_frame_m5(raw_frame: pd.DataFrame) -> pd.DataFrame:
    frame = raw_frame.copy()
    frame["prev_close"] = frame["close"].shift(1)
    frame["prev_open"] = frame["open"].shift(1)
    frame["prev_high"] = frame["high"].shift(1)
    frame["prev_low"] = frame["low"].shift(1)
    frame["bar_range"] = frame["high"] - frame["low"]
    frame["body_abs"] = (frame["close"] - frame["open"]).abs()
    frame["atr20"] = atr(frame, 20)
    frame["atr14_signal"] = atr(frame, 14)
    frame["atr7"] = atr(frame, 7)
    frame["atr14"] = frame["atr20"]
    frame["adx14"] = adx(frame, 14)
    frame["range_atr"] = frame["bar_range"] / frame["atr20"].replace(0.0, np.nan)
    frame["ema20"] = ema(frame["close"], 20)
    frame["ema50"] = ema(frame["close"], 50)
    frame["rsi14"] = rsi(frame["close"], 14)
    frame["close_location"] = ((frame["close"] - frame["low"]) / (frame["bar_range"].replace(0.0, np.nan))).clip(0.0, 1.0)
    frame["ema50_slope_4"] = frame["ema50"] - frame["ema50"].shift(4)
    frame["ema50_slope_5"] = frame["ema50"] - frame["ema50"].shift(5)
    frame["atr_ratio_7_20"] = frame["atr7"] / frame["atr20"].replace(0.0, np.nan)
    frame["nr7_flag"] = frame["bar_range"] <= frame["bar_range"].rolling(7).min()
    frame["inside_bar_flag"] = (frame["high"] < frame["high"].shift(1)) & (frame["low"] > frame["low"].shift(1))
    frame["inside_range_ratio"] = frame["bar_range"] / frame["atr20"].replace(0.0, np.nan)
    frame["vwap_session_11"] = session_vwap_11_19(frame)
    frame["vwap_dist_atr"] = (frame["close"] - frame["vwap_session_11"]) / frame["atr20"].replace(0.0, np.nan)
    frame["rolling_low_prev_8"] = frame["low"].shift(1).rolling(8).min()
    frame["rolling_low_prev_12"] = frame["low"].shift(1).rolling(12).min()
    frame["rolling_low_prev_16"] = frame["low"].shift(1).rolling(16).min()
    frame["rolling_high_prev_8"] = frame["high"].shift(1).rolling(8).max()
    frame["rolling_high_prev_12"] = frame["high"].shift(1).rolling(12).max()
    frame["rolling_high_prev_16"] = frame["high"].shift(1).rolling(16).max()
    frame["three_bar_range_sum"] = frame["bar_range"] + frame["bar_range"].shift(1) + frame["bar_range"].shift(2)
    frame["three_down_close_seq"] = (
        (frame["close"] < frame["prev_close"])
        & (frame["prev_close"] < frame["close"].shift(2))
        & (frame["close"].shift(2) < frame["close"].shift(3))
    )
    frame["three_up_close_seq"] = (
        (frame["close"] > frame["prev_close"])
        & (frame["prev_close"] > frame["close"].shift(2))
        & (frame["close"].shift(2) > frame["close"].shift(3))
    )
    frame["three_down_body_seq"] = (
        (frame["close"] < frame["open"])
        & (frame["close"].shift(1) < frame["open"].shift(1))
        & (frame["close"].shift(2) < frame["open"].shift(2))
    )
    frame["three_up_body_seq"] = (
        (frame["close"] > frame["open"])
        & (frame["close"].shift(1) > frame["open"].shift(1))
        & (frame["close"].shift(2) > frame["open"].shift(2))
    )
    frame["adx_rising"] = frame["adx14"] > frame["adx14"].shift(1)
    frame = fill_midday_range(frame)
    required = [
        "atr20",
        "atr14",
        "adx14",
        "range_atr",
        "ema20",
        "ema50",
        "ema50_slope_4",
        "ema50_slope_5",
        "vwap_session_11",
        "rolling_low_prev_8",
        "rolling_high_prev_8",
    ]
    frame = frame.dropna(subset=required).copy()
    return frame


def load_mode_context_m5(
    mode: str,
    pair: str,
    start: str,
    end: str,
    data_dirs: tuple[Path, ...] = DEFAULT_DATA_DIRS,
    high_precision_dir: Path = DEFAULT_HIGH_PRECISION_PREPARED_DIR,
) -> dict[str, Any]:
    start_ts = pd.Timestamp(start, tz=NY_TZ)
    end_m5 = pd.Timestamp(end, tz=NY_TZ) + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)
    end_m1 = pd.Timestamp(end, tz=NY_TZ) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)

    if mode != "high_precision_mode":
        raw_m5 = load_prepared_ohlcv(pair, list(data_dirs), "M5")
        raw_m5 = raw_m5[fx_market_mask(raw_m5.index)].copy()
        raw_m5 = raw_m5.loc[(raw_m5.index >= start_ts) & (raw_m5.index <= end_m5)].copy()
        validate_price_frame(raw_m5)
        return {
            "frame": prepare_common_frame_m5(raw_m5),
            "precision_package": None,
            "data_source_used": "prepared_m5_bid",
        }

    package = load_high_precision_package(pair, high_precision_dir)
    filtered_m1: dict[str, pd.DataFrame] = {}
    for side, source in package.items():
        side_frame = source.loc[(source.index >= start_ts) & (source.index <= end_m1)].copy()
        side_frame = side_frame[fx_market_mask(side_frame.index)].copy()
        validate_price_frame(side_frame)
        filtered_m1[f"{side}_m1"] = side_frame

    bid_m5 = _resample_to_5min(filtered_m1["bid_m1"])
    ask_m5 = _resample_to_5min(filtered_m1["ask_m1"])
    mid_m5 = _resample_to_5min(filtered_m1["mid_m1"])
    strategy_frame = prepare_common_frame_m5(mid_m5)
    common_index = strategy_frame.index.intersection(bid_m5.index).intersection(ask_m5.index).intersection(mid_m5.index)
    if common_index.empty:
        raise ValueError("La fuente M1 BID/ASK no pudo alinearse con el frame M5 del protocolo.")

    aligned_package = {
        **filtered_m1,
        "bid_m15": bid_m5.loc[common_index].copy(),
        "ask_m15": ask_m5.loc[common_index].copy(),
        "mid_m15": mid_m5.loc[common_index].copy(),
    }
    return {
        "frame": strategy_frame.loc[common_index].copy(),
        "precision_package": aligned_package,
        "data_source_used": "dukascopy_m1_bid_ask_full",
    }


def period_slice(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz=NY_TZ)
    end_ts = pd.Timestamp(end, tz=NY_TZ) + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)
    return frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)].copy()


def build_engine_config(execution_mode: str, max_spread_pips: float) -> EngineConfig:
    return with_execution_mode(
        EngineConfig(
            pair="EURUSD",
            risk_pct=0.5,
            assumed_spread_pips=1.2,
            max_spread_pips=max_spread_pips,
            slippage_pips=0.2,
            commission_per_lot_roundturn_usd=7.0,
            shock_candle_atr_max=2.8,
            max_trades_per_day=1,
            execution_mode=execution_mode,
        ),
        execution_mode,
    )


def merge_management(management_id: str, **overrides: Any) -> dict[str, Any]:
    payload = dict(MANAGEMENT_PACKAGES[management_id])
    payload["management_id"] = management_id
    payload.update({key: value for key, value in overrides.items() if value is not None})
    return payload


def build_signal_from_management(
    *,
    direction: str,
    management: dict[str, Any],
    structural_stop_price: float | None = None,
    target_price: float | None = None,
    target_rr: float | None = None,
) -> dict[str, Any] | None:
    signal = {
        "direction": direction,
        "break_even_at_r": management.get("break_even_at_r"),
        "trailing_atr": False,
        "session_name": "light_fixed",
        "max_hold_bars": management.get("max_hold_bars"),
        "spread_max_pips": management.get("spread_max_pips"),
    }
    if management["stop_kind"] == "structural":
        if structural_stop_price is None or not np.isfinite(structural_stop_price):
            return None
        signal["stop_mode"] = "price"
        signal["stop_price"] = float(structural_stop_price)
    else:
        signal["stop_mode"] = "atr"
        signal["stop_atr"] = float(management["stop_atr"])

    if target_price is not None and np.isfinite(target_price):
        signal["target_mode"] = "price"
        signal["target_price"] = float(target_price)
    else:
        signal["target_rr"] = float(target_rr if target_rr is not None else management["target_rr"])
    return signal


def share_best_year(yearly_stats: pd.DataFrame) -> float:
    if yearly_stats.empty:
        return 0.0
    yearly = yearly_stats.groupby("year")["total_pnl_r"].sum()
    positive_total = float(yearly[yearly > 0].sum())
    if positive_total <= 0:
        return 0.0
    return float(yearly.max() / positive_total)


def exit_reason_distribution(trades_export: pd.DataFrame) -> dict[str, int]:
    if trades_export.empty or "exit_reason" not in trades_export.columns:
        return {}
    counts = trades_export["exit_reason"].value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def mode_score(summary: dict[str, Any]) -> float:
    trades_pm = float(summary["avg_trades_per_month"])
    return (
        float(summary["profit_factor"]) * 220.0
        + float(summary["expectancy_r"]) * 900.0
        - float(summary["max_drawdown_pct"]) * 3.0
        - float(summary["negative_years"]) * 45.0
        - float(summary["negative_months"]) * 1.5
        - abs(trades_pm - 12.0) * 3.0
    )


def collapse_against_normal(normal_summary: dict[str, Any], other_summary: dict[str, Any]) -> bool:
    if float(normal_summary["profit_factor"]) <= 1.0 or float(normal_summary["expectancy_r"]) <= 0.0:
        return False
    return (
        float(other_summary["profit_factor"]) < float(normal_summary["profit_factor"]) * COLLAPSE_PF_RATIO
        or float(other_summary["expectancy_r"]) < float(normal_summary["expectancy_r"]) * COLLAPSE_EXPECTANCY_RATIO
    )


def period_reasons(
    period_name: str,
    mode_payloads: dict[str, tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
) -> list[str]:
    reasons: list[str] = []
    for mode, payload in mode_payloads.items():
        summary = payload[0]
        yearly_stats = payload[3]
        if int(summary["total_trades"]) < MIN_TOTAL_TRADES[period_name]:
            reasons.append(f"{period_name}:{mode}:sample_too_small")
        if float(summary["avg_trades_per_month"]) < MIN_TRADES_PER_MONTH[period_name]:
            reasons.append(f"{period_name}:{mode}:frequency_too_low")
        if float(summary["profit_factor"]) <= 1.0:
            reasons.append(f"{period_name}:{mode}:pf<=1")
        if float(summary["expectancy_r"]) <= 0.0:
            reasons.append(f"{period_name}:{mode}:expectancy<=0")
        if period_name == "development_2020_2023":
            if int(summary["negative_years"]) > MAX_NEGATIVE_YEARS_DEVELOPMENT:
                reasons.append(f"{period_name}:{mode}:too_many_negative_years")
            if share_best_year(yearly_stats) > MAX_SHARE_BEST_YEAR:
                reasons.append(f"{period_name}:{mode}:year_dependency>0.60")
    normal_summary = mode_payloads["normal_mode"][0]
    if collapse_against_normal(normal_summary, mode_payloads["conservative_mode"][0]):
        reasons.append(f"{period_name}:conservative_collapse")
    if collapse_against_normal(normal_summary, mode_payloads["high_precision_mode"][0]):
        reasons.append(f"{period_name}:high_precision_collapse")
    return reasons


def weighted_mode_score(mode_payloads: dict[str, tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]) -> float:
    weights = {"normal_mode": 0.20, "conservative_mode": 0.35, "high_precision_mode": 0.45}
    return float(sum(mode_score(mode_payloads[mode][0]) * weights[mode] for mode in MODES))
