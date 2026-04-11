"""
Laboratorio rapido de combinaciones de indicadores conocidas sobre un solo par.

Objetivo:
- discovery inicial con una sola serie M5
- buscar setups mas frecuentes sin usar el motor completo
- mantener riesgo fijo, noticias y shock guard

Diseno:
- un par por corrida
- una posicion a la vez
- solo intradia NY
- sesion cerrada a las 18:45 NY
- espacio de busqueda chico y explicable
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import fx_multi_timeframe_backtester as core


SESSION_END = "18:45"
RISK_FRACTION = 0.01
INITIAL_CAPITAL = 100_000.0
SLIPPAGE_PIPS = 0.2
NO_ENTRY_PRE_MIN = 45
NO_ENTRY_POST_MIN = 30
FLATTEN_PRE_MIN = 15
SHOCK_NO_ENTRY_ATR = 2.5
SHOCK_FLATTEN_ATR = 3.0
SHOCK_COOLDOWN_BARS = 6


@dataclass(frozen=True)
class SessionProfile:
    name: str
    start: str
    end: str


@dataclass(frozen=True)
class RegimeProfile:
    name: str
    mode: str
    adx_max: float | None = None
    adx_min: float | None = None
    ema_spread_atr_max: float | None = None
    ema_spread_atr_min: float | None = None


@dataclass(frozen=True)
class StrategySpec:
    label: str
    style: str
    variant: str
    session: SessionProfile
    regime: RegimeProfile
    setup_dist_atr: float
    setup_oscillator: float
    trigger_oscillator: float
    stop_atr: float
    take_profit_rr: float
    max_hold_bars: int
    max_closed_trades_per_day: int = 3
    cooldown_bars: int = 3


SESSIONS: tuple[SessionProfile, ...] = (
    SessionProfile("early", "11:00", "13:30"),
    SessionProfile("core", "13:30", "16:00"),
    SessionProfile("full", "11:00", "16:45"),
)

REGIMES: dict[str, RegimeProfile] = {
    "mr_strict": RegimeProfile("mr_strict", "mr", adx_max=16.0, ema_spread_atr_max=0.80),
    "mr_balanced": RegimeProfile("mr_balanced", "mr", adx_max=20.0, ema_spread_atr_max=1.20),
    "trend_balanced": RegimeProfile("trend_balanced", "trend", adx_min=18.0, ema_spread_atr_min=0.25),
    "trend_active": RegimeProfile("trend_active", "trend", adx_min=22.0, ema_spread_atr_min=0.40),
}

STYLE_VARIANTS: dict[str, tuple[dict[str, Any], ...]] = {
    "mr_vwap_rsi2": (
        {
            "variant": "tight",
            "setup_dist_atr": 0.45,
            "setup_oscillator": 8.0,
            "trigger_oscillator": 18.0,
            "stop_atr": 0.70,
            "take_profit_rr": 0.80,
            "max_hold_bars": 8,
        },
        {
            "variant": "balanced",
            "setup_dist_atr": 0.35,
            "setup_oscillator": 12.0,
            "trigger_oscillator": 22.0,
            "stop_atr": 0.80,
            "take_profit_rr": 0.90,
            "max_hold_bars": 10,
        },
    ),
    "mr_vwap_rsi7": (
        {
            "variant": "tight",
            "setup_dist_atr": 0.50,
            "setup_oscillator": 24.0,
            "trigger_oscillator": 40.0,
            "stop_atr": 0.80,
            "take_profit_rr": 0.80,
            "max_hold_bars": 10,
        },
        {
            "variant": "balanced",
            "setup_dist_atr": 0.40,
            "setup_oscillator": 30.0,
            "trigger_oscillator": 45.0,
            "stop_atr": 0.90,
            "take_profit_rr": 0.90,
            "max_hold_bars": 12,
        },
    ),
    "mr_bb_stoch": (
        {
            "variant": "tight",
            "setup_dist_atr": 0.00,
            "setup_oscillator": 10.0,
            "trigger_oscillator": 20.0,
            "stop_atr": 0.75,
            "take_profit_rr": 0.80,
            "max_hold_bars": 8,
        },
        {
            "variant": "balanced",
            "setup_dist_atr": 0.00,
            "setup_oscillator": 15.0,
            "trigger_oscillator": 25.0,
            "stop_atr": 0.85,
            "take_profit_rr": 0.90,
            "max_hold_bars": 10,
        },
    ),
    "trend_ema_rsi7": (
        {
            "variant": "tight",
            "setup_dist_atr": 0.15,
            "setup_oscillator": 42.0,
            "trigger_oscillator": 52.0,
            "stop_atr": 0.85,
            "take_profit_rr": 1.00,
            "max_hold_bars": 10,
        },
        {
            "variant": "balanced",
            "setup_dist_atr": 0.10,
            "setup_oscillator": 45.0,
            "trigger_oscillator": 50.0,
            "stop_atr": 0.95,
            "take_profit_rr": 1.10,
            "max_hold_bars": 12,
        },
    ),
}

STYLE_REGIMES: dict[str, tuple[str, ...]] = {
    "mr_vwap_rsi2": ("mr_strict", "mr_balanced"),
    "mr_vwap_rsi7": ("mr_strict", "mr_balanced"),
    "mr_bb_stoch": ("mr_strict", "mr_balanced"),
    "trend_ema_rsi7": ("trend_balanced", "trend_active"),
}

STYLE_SESSIONS: dict[str, tuple[str, ...]] = {
    "mr_vwap_rsi2": ("early", "core", "full"),
    "mr_vwap_rsi7": ("early", "core", "full"),
    "mr_bb_stoch": ("early", "core", "full"),
    "trend_ema_rsi7": ("early", "core"),
}


def build_specs() -> tuple[StrategySpec, ...]:
    session_lookup = {session.name: session for session in SESSIONS}
    specs: list[StrategySpec] = []
    for style, variants in STYLE_VARIANTS.items():
        for regime_name in STYLE_REGIMES[style]:
            regime = REGIMES[regime_name]
            for session_name in STYLE_SESSIONS[style]:
                session = session_lookup[session_name]
                for variant in variants:
                    label = f"{style}__{regime.name}__{session.name}__{variant['variant']}"
                    specs.append(
                        StrategySpec(
                            label=label,
                            style=style,
                            variant=variant["variant"],
                            session=session,
                            regime=regime,
                            setup_dist_atr=variant["setup_dist_atr"],
                            setup_oscillator=variant["setup_oscillator"],
                            trigger_oscillator=variant["trigger_oscillator"],
                            stop_atr=variant["stop_atr"],
                            take_profit_rr=variant["take_profit_rr"],
                            max_hold_bars=variant["max_hold_bars"],
                        )
                    )
    return tuple(specs)


SPECS = build_specs()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(frame: pd.DataFrame, period: int) -> pd.Series:
    high = frame["high"]
    low = frame["low"]
    close = frame["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def stochastic_k(frame: pd.DataFrame, period: int) -> pd.Series:
    lowest_low = frame["low"].rolling(period).min()
    highest_high = frame["high"].rolling(period).max()
    return 100 * (frame["close"] - lowest_low) / (highest_high - lowest_low).replace(0.0, np.nan)


def bollinger(frame: pd.DataFrame, period: int, std_mult: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = frame["close"].rolling(period).mean()
    std = frame["close"].rolling(period).std(ddof=0)
    upper = mid + std * std_mult
    lower = mid - std * std_mult
    return mid, upper, lower


def session_vwap(frame: pd.DataFrame) -> pd.Series:
    local_index = frame.index.tz_convert("America/New_York")
    session_key = local_index.date
    typical_price = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    volume = frame["volume"].fillna(0.0)
    numerator = (typical_price * volume).groupby(session_key).cumsum()
    denominator = volume.groupby(session_key).cumsum().replace(0.0, np.nan)
    return numerator / denominator


def load_pair_frame(data_dir: Path, pair: str, start: str, end: str) -> pd.DataFrame:
    path = data_dir / f"{pair}_M5.csv"
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert("America/New_York")
    start_ts = pd.Timestamp(start, tz="America/New_York")
    end_ts = pd.Timestamp(end, tz="America/New_York") + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)
    frame = frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)].copy()
    return frame


def relevant_currencies(pair: str) -> set[str]:
    meta = core.PAIR_META[pair]
    return {meta["base"], meta["quote"]}


def build_news_masks(index: pd.DatetimeIndex, pair: str, news_file: Path | None) -> tuple[np.ndarray, np.ndarray]:
    block = np.zeros(len(index), dtype=bool)
    flatten = np.zeros(len(index), dtype=bool)
    if news_file is None or not news_file.exists():
        return block, flatten

    news = pd.read_csv(news_file)
    if news.empty:
        return block, flatten

    news["DateTime"] = pd.to_datetime(news["DateTime"], utc=True).dt.tz_convert("America/New_York")
    impact = news["Impact"].astype(str).str.lower()
    news = news[impact.str.contains("high")].copy()
    if news.empty:
        return block, flatten

    currencies = relevant_currencies(pair)
    news = news[news["Currency"].isin(currencies)].copy()
    if news.empty:
        return block, flatten

    for event_ts in news["DateTime"]:
        block_start = event_ts - pd.Timedelta(minutes=NO_ENTRY_PRE_MIN)
        block_end = event_ts + pd.Timedelta(minutes=NO_ENTRY_POST_MIN)
        flatten_start = event_ts - pd.Timedelta(minutes=FLATTEN_PRE_MIN)
        left = index.searchsorted(block_start, side="left")
        right = index.searchsorted(block_end, side="right")
        block[left:right] = True
        flat_left = index.searchsorted(flatten_start, side="left")
        flatten[flat_left:right] = True
    return block, flatten


def session_mask(index: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    local = index.tz_convert("America/New_York")
    start_hour, start_minute = (int(part) for part in start.split(":"))
    end_hour, end_minute = (int(part) for part in end.split(":"))
    minutes = local.hour * 60 + local.minute
    start_total = start_hour * 60 + start_minute
    end_total = end_hour * 60 + end_minute
    return (minutes >= start_total) & (minutes < end_total)


def force_close_mask(index: pd.DatetimeIndex) -> np.ndarray:
    local = index.tz_convert("America/New_York")
    end_hour, end_minute = (int(part) for part in SESSION_END.split(":"))
    minutes = local.hour * 60 + local.minute
    return minutes >= end_hour * 60 + end_minute


def build_shock_masks(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    range_atr = frame["range_atr"].fillna(0.0)
    shock_flatten = (range_atr >= SHOCK_FLATTEN_ATR).to_numpy()
    shock_raw = (range_atr >= SHOCK_NO_ENTRY_ATR).astype(int)
    shock_block = shock_raw.rolling(SHOCK_COOLDOWN_BARS, min_periods=1).max().astype(bool).to_numpy()
    return shock_block, shock_flatten


def compute_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["ema8"] = ema(enriched["close"], 8)
    enriched["ema20"] = ema(enriched["close"], 20)
    enriched["ema50"] = ema(enriched["close"], 50)
    enriched["ema200"] = ema(enriched["close"], 200)
    enriched["rsi2"] = rsi(enriched["close"], 2)
    enriched["rsi7"] = rsi(enriched["close"], 7)
    enriched["atr14"] = atr(enriched, 14)
    enriched["adx14"] = core.adx(enriched, 14)
    enriched["stoch14"] = stochastic_k(enriched, 14)
    bb_mid, bb_upper, bb_lower = bollinger(enriched, 20, 2.0)
    enriched["bb_mid"] = bb_mid
    enriched["bb_upper"] = bb_upper
    enriched["bb_lower"] = bb_lower
    enriched["vwap"] = session_vwap(enriched)
    enriched["bar_range"] = enriched["high"] - enriched["low"]
    enriched["range_atr"] = enriched["bar_range"] / enriched["atr14"].replace(0.0, np.nan)
    enriched["ema_spread_atr"] = (enriched["ema50"] - enriched["ema200"]).abs() / enriched["atr14"].replace(0.0, np.nan)
    return enriched


def summarize_stage(trades: pd.DataFrame, equity_curve: pd.DataFrame) -> dict[str, Any]:
    return core.summarize_portfolio(trades, equity_curve, INITIAL_CAPITAL)


def favourable_target(direction: str, entry_price: float, rr_target: float, candidates: list[float]) -> float:
    best = rr_target
    for candidate in candidates:
        if not np.isfinite(candidate):
            continue
        if direction == "long" and candidate > entry_price:
            best = min(best, candidate)
        elif direction == "short" and candidate < entry_price:
            best = max(best, candidate)
    return best


def build_regime_masks(frame: pd.DataFrame, spec: StrategySpec) -> tuple[np.ndarray, np.ndarray]:
    adx_values = frame["adx14"].to_numpy()
    spread_atr = frame["ema_spread_atr"].to_numpy()
    ema50 = frame["ema50"].to_numpy()
    ema200 = frame["ema200"].to_numpy()

    if spec.regime.mode == "mr":
        valid = np.isfinite(adx_values) & np.isfinite(spread_atr)
        long_mask = valid
        short_mask = valid
        if spec.regime.adx_max is not None:
            long_mask &= adx_values <= spec.regime.adx_max
            short_mask &= adx_values <= spec.regime.adx_max
        if spec.regime.ema_spread_atr_max is not None:
            long_mask &= spread_atr <= spec.regime.ema_spread_atr_max
            short_mask &= spread_atr <= spec.regime.ema_spread_atr_max
        return long_mask, short_mask

    valid = np.isfinite(adx_values) & np.isfinite(spread_atr)
    long_mask = valid & (ema50 > ema200)
    short_mask = valid & (ema50 < ema200)
    if spec.regime.adx_min is not None:
        long_mask &= adx_values >= spec.regime.adx_min
        short_mask &= adx_values >= spec.regime.adx_min
    if spec.regime.ema_spread_atr_min is not None:
        long_mask &= spread_atr >= spec.regime.ema_spread_atr_min
        short_mask &= spread_atr >= spec.regime.ema_spread_atr_min
    return long_mask, short_mask


def build_setup_arrays(frame: pd.DataFrame, spec: StrategySpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    close = frame["close"].to_numpy()
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    ema8 = frame["ema8"].to_numpy()
    ema20 = frame["ema20"].to_numpy()
    atr_values = frame["atr14"].to_numpy()
    vwap = frame["vwap"].to_numpy()
    bb_lower = frame["bb_lower"].to_numpy()
    bb_upper = frame["bb_upper"].to_numpy()
    rsi2_values = frame["rsi2"].to_numpy()
    rsi7_values = frame["rsi7"].to_numpy()
    stoch = frame["stoch14"].to_numpy()
    long_regime, short_regime = build_regime_masks(frame, spec)

    prev_close = np.roll(close, 1)
    prev_ema8 = np.roll(ema8, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)

    if spec.style == "mr_vwap_rsi2":
        long_setup_raw = (
            long_regime
            & ((vwap - close) / np.maximum(atr_values, 1e-9) >= spec.setup_dist_atr)
            & (rsi2_values <= spec.setup_oscillator)
            & (close < ema20)
        )
        short_setup_raw = (
            short_regime
            & ((close - vwap) / np.maximum(atr_values, 1e-9) >= spec.setup_dist_atr)
            & (rsi2_values >= 100.0 - spec.setup_oscillator)
            & (close > ema20)
        )
        long_trigger = (prev_close <= prev_ema8) & (close > ema8) & (rsi2_values >= spec.trigger_oscillator)
        short_trigger = (prev_close >= prev_ema8) & (close < ema8) & (rsi2_values <= 100.0 - spec.trigger_oscillator)
    elif spec.style == "mr_vwap_rsi7":
        long_setup_raw = (
            long_regime
            & ((vwap - close) / np.maximum(atr_values, 1e-9) >= spec.setup_dist_atr)
            & (rsi7_values <= spec.setup_oscillator)
            & (close < ema20)
        )
        short_setup_raw = (
            short_regime
            & ((close - vwap) / np.maximum(atr_values, 1e-9) >= spec.setup_dist_atr)
            & (rsi7_values >= 100.0 - spec.setup_oscillator)
            & (close > ema20)
        )
        long_trigger = (prev_close <= prev_ema8) & (close > ema8) & (rsi7_values >= spec.trigger_oscillator)
        short_trigger = (prev_close >= prev_ema8) & (close < ema8) & (rsi7_values <= 100.0 - spec.trigger_oscillator)
    elif spec.style == "mr_bb_stoch":
        long_setup_raw = long_regime & (close <= bb_lower) & (stoch <= spec.setup_oscillator)
        short_setup_raw = short_regime & (close >= bb_upper) & (stoch >= 100.0 - spec.setup_oscillator)
        long_trigger = (close > bb_lower) & (close > ema8) & (stoch >= spec.trigger_oscillator)
        short_trigger = (close < bb_upper) & (close < ema8) & (stoch <= 100.0 - spec.trigger_oscillator)
    elif spec.style == "trend_ema_rsi7":
        long_setup_raw = (
            long_regime
            & ((ema20 - close) / np.maximum(atr_values, 1e-9) >= spec.setup_dist_atr)
            & (close <= ema20)
            & (rsi7_values <= spec.setup_oscillator)
        )
        short_setup_raw = (
            short_regime
            & ((close - ema20) / np.maximum(atr_values, 1e-9) >= spec.setup_dist_atr)
            & (close >= ema20)
            & (rsi7_values >= 100.0 - spec.setup_oscillator)
        )
        long_trigger = (close > ema8) & (close > prev_high) & (rsi7_values >= spec.trigger_oscillator)
        short_trigger = (close < ema8) & (close < prev_low) & (rsi7_values <= 100.0 - spec.trigger_oscillator)
    else:
        raise ValueError(f"Estilo no soportado: {spec.style}")

    long_setup_recent = (
        pd.Series(long_setup_raw, index=frame.index).rolling(3, min_periods=1).max().shift(1).fillna(0).astype(bool).to_numpy()
    )
    short_setup_recent = (
        pd.Series(short_setup_raw, index=frame.index).rolling(3, min_periods=1).max().shift(1).fillna(0).astype(bool).to_numpy()
    )
    long_trigger[0] = False
    short_trigger[0] = False
    return long_setup_recent, short_setup_recent, long_trigger, short_trigger


def backtest_spec(
    frame: pd.DataFrame,
    pair: str,
    spec: StrategySpec,
    news_block: np.ndarray,
    news_flatten: np.ndarray,
    shock_block: np.ndarray,
    shock_flatten: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    index = frame.index
    open_ = frame["open"].to_numpy()
    close = frame["close"].to_numpy()
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    ema20 = frame["ema20"].to_numpy()
    vwap = frame["vwap"].to_numpy()
    bb_mid = frame["bb_mid"].to_numpy()
    atr_values = frame["atr14"].to_numpy()
    entry_allowed = session_mask(index, spec.session.start, spec.session.end)
    force_close = force_close_mask(index)
    long_setup_recent, short_setup_recent, long_trigger, short_trigger = build_setup_arrays(frame, spec)

    pip = core.PAIR_META[pair]["pip_size"]
    spread = (core.DEFAULT_SPREAD_PIPS.get(pair, 0.8) * 0.5 + SLIPPAGE_PIPS) * pip

    position: dict[str, Any] | None = None
    trades: list[dict[str, Any]] = []
    equity_points: list[dict[str, Any]] = []
    equity = INITIAL_CAPITAL
    cooldown_until_bar = -1
    closed_trades_per_day: dict[Any, int] = {}

    for i in range(1, len(frame) - 1):
        ts = index[i]
        session_day = ts.date()

        if position is not None:
            exit_reason = None
            exit_price = None
            if news_flatten[i] or force_close[i]:
                exit_reason = "forced_flatten"
                exit_price = close[i] - spread if position["direction"] == "long" else close[i] + spread
            elif shock_flatten[i]:
                exit_reason = "shock_flatten"
                exit_price = close[i] - spread if position["direction"] == "long" else close[i] + spread
            else:
                if position["direction"] == "long":
                    if low[i] <= position["stop"]:
                        exit_reason = "stop_loss"
                        exit_price = position["stop"]
                    elif high[i] >= position["target"]:
                        exit_reason = "take_profit"
                        exit_price = position["target"]
                else:
                    if high[i] >= position["stop"]:
                        exit_reason = "stop_loss"
                        exit_price = position["stop"]
                    elif low[i] <= position["target"]:
                        exit_reason = "take_profit"
                        exit_price = position["target"]
                if exit_reason is None and i - position["entry_bar"] >= spec.max_hold_bars:
                    exit_reason = "time_stop"
                    exit_price = close[i] - spread if position["direction"] == "long" else close[i] + spread

            if exit_reason is not None and exit_price is not None:
                sign = 1.0 if position["direction"] == "long" else -1.0
                gross_pnl = position["units"] * sign * (exit_price - position["entry_price"])
                commission = position["units"] * position["entry_price"] * 0.00002 + position["units"] * exit_price * 0.00002
                net_pnl = gross_pnl - commission
                r_multiple = net_pnl / max(position["risk_usd"], 1e-9)
                equity += net_pnl
                trades.append(
                    {
                        "pair": pair,
                        "direction": position["direction"],
                        "entry_time": position["entry_time"],
                        "exit_time": ts,
                        "entry_price": position["entry_price"],
                        "exit_price": exit_price,
                        "stop_price": position["stop"],
                        "take_profit_price": position["target"],
                        "initial_risk_usd": position["risk_usd"],
                        "units": position["units"],
                        "gross_pnl_usd": gross_pnl,
                        "net_pnl_usd": net_pnl,
                        "r_multiple": r_multiple,
                        "commission_usd": commission,
                        "exit_reason": exit_reason,
                        "strategy_family": "fast_indicator_combo",
                        "context_key": f"{spec.regime.name}|{spec.session.name}",
                        "setup_tag": spec.style,
                    }
                )
                cooldown_until_bar = i + spec.cooldown_bars
                closed_trades_per_day[session_day] = closed_trades_per_day.get(session_day, 0) + 1
                position = None

        can_enter = (
            position is None
            and i > cooldown_until_bar
            and entry_allowed[i]
            and not news_block[i]
            and not shock_block[i]
            and np.isfinite(atr_values[i])
            and atr_values[i] > 0
            and closed_trades_per_day.get(session_day, 0) < spec.max_closed_trades_per_day
        )
        if can_enter:
            direction = None
            if long_setup_recent[i] and long_trigger[i]:
                direction = "long"
            elif short_setup_recent[i] and short_trigger[i]:
                direction = "short"

            if direction is not None:
                entry_price = open_[i + 1] + spread if direction == "long" else open_[i + 1] - spread
                stop_distance = atr_values[i] * spec.stop_atr
                if np.isfinite(stop_distance) and stop_distance > 0:
                    risk_usd = INITIAL_CAPITAL * RISK_FRACTION
                    units = math.floor(risk_usd / stop_distance)
                    if units > 0:
                        rr_target = entry_price + stop_distance * spec.take_profit_rr if direction == "long" else entry_price - stop_distance * spec.take_profit_rr
                        target = rr_target
                        if spec.style.startswith("mr_"):
                            target = favourable_target(direction, entry_price, rr_target, [vwap[i], ema20[i], bb_mid[i]])
                        stop = entry_price - stop_distance if direction == "long" else entry_price + stop_distance
                        position = {
                            "direction": direction,
                            "entry_time": index[i + 1],
                            "entry_bar": i + 1,
                            "entry_price": entry_price,
                            "stop": stop,
                            "target": target,
                            "units": units,
                            "risk_usd": risk_usd,
                        }

        equity_points.append({"timestamp": ts, "equity": equity})

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
        trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])
    equity_df = pd.DataFrame(equity_points)
    if not equity_df.empty:
        equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"])
    summary = summarize_stage(trades_df, equity_df)
    return trades_df, equity_df, summary


def run_stage(
    *,
    pair: str,
    data_dir: Path,
    start: str,
    end: str,
    news_file: Path | None,
) -> dict[str, tuple[StrategySpec, pd.DataFrame, pd.DataFrame, dict[str, Any]]]:
    results: dict[str, tuple[StrategySpec, pd.DataFrame, pd.DataFrame, dict[str, Any]]] = {}
    frame = compute_features(load_pair_frame(data_dir, pair, start, end))
    news_block, news_flatten = build_news_masks(frame.index, pair, news_file)
    shock_block, shock_flatten = build_shock_masks(frame)
    for spec in SPECS:
        trades, equity, summary = backtest_spec(frame, pair, spec, news_block, news_flatten, shock_block, shock_flatten)
        results[spec.label] = (spec, trades, equity, summary)
    return results


def export_result(root: Path, label: str, trades: pd.DataFrame, equity_curve: pd.DataFrame, summary: dict[str, Any], params: dict[str, Any]) -> None:
    target = root / label
    target.mkdir(parents=True, exist_ok=True)
    core.export_analysis_bundle(
        target,
        trades=trades,
        equity_curve=equity_curve,
        initial_capital=INITIAL_CAPITAL,
        portfolio_summary=summary,
        parameters=params,
        best_score=None,
    )


def build_ranking(
    design_results: dict[str, tuple[StrategySpec, pd.DataFrame, pd.DataFrame, dict[str, Any]]],
    oos_results: dict[str, tuple[StrategySpec, pd.DataFrame, pd.DataFrame, dict[str, Any]]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, (spec, _, _, design_summary) in design_results.items():
        oos_summary = oos_results[label][3]
        rows.append(
            {
                "label": label,
                "style": spec.style,
                "variant": spec.variant,
                "regime": spec.regime.name,
                "session": spec.session.name,
                "design_return_pct": design_summary["total_return_pct"],
                "design_drawdown_pct": design_summary["max_drawdown_pct"],
                "design_profit_factor": design_summary["profit_factor"],
                "design_trades_per_month": design_summary["trades_per_month"],
                "design_total_trades": design_summary["total_trades"],
                "oos_return_pct": oos_summary["total_return_pct"],
                "oos_drawdown_pct": oos_summary["max_drawdown_pct"],
                "oos_profit_factor": oos_summary["profit_factor"],
                "oos_trades_per_month": oos_summary["trades_per_month"],
                "oos_total_trades": oos_summary["total_trades"],
            }
        )
    ranking = pd.DataFrame(rows)
    ranking["accepted"] = (
        (ranking["design_total_trades"] >= 24)
        & (ranking["oos_total_trades"] >= 48)
        & (ranking["design_return_pct"] >= 0.0)
        & (ranking["design_profit_factor"] >= 1.05)
        & (ranking["design_drawdown_pct"] <= 8.0)
        & (ranking["design_trades_per_month"] >= 1.0)
        & (ranking["oos_return_pct"] >= 0.0)
        & (ranking["oos_profit_factor"] >= 1.05)
        & (ranking["oos_drawdown_pct"] <= 10.0)
        & (ranking["oos_trades_per_month"] >= 1.0)
    )
    pf_design = ranking["design_profit_factor"].replace(np.inf, 10.0)
    pf_oos = ranking["oos_profit_factor"].replace(np.inf, 10.0)
    ranking["score"] = (
        ranking["accepted"].astype(int) * 500.0
        + ranking["oos_return_pct"] * 6.0
        + pf_oos * 20.0
        + np.minimum(ranking["oos_trades_per_month"], 6.0) * 40.0
        - ranking["oos_drawdown_pct"] * 10.0
        + ranking["design_return_pct"] * 2.0
        + pf_design * 8.0
        + np.minimum(ranking["design_trades_per_month"], 6.0) * 20.0
        - ranking["design_drawdown_pct"] * 4.0
    )
    return ranking.sort_values(["score", "oos_return_pct", "oos_profit_factor"], ascending=[False, False, False]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast indicator combo lab on one pair.")
    parser.add_argument("--pair", default="EURUSD")
    parser.add_argument("--design-start", default="2020-01-01")
    parser.add_argument("--design-end", default="2021-12-31")
    parser.add_argument("--design-data-dir", default="data_free_2020/prepared")
    parser.add_argument("--oos-start", default="2022-01-01")
    parser.add_argument("--oos-end", default="2025-12-31")
    parser.add_argument("--oos-data-dir", default="data_candidates_2022_2025/prepared")
    parser.add_argument("--news-file", default="data/forex_factory_cache.csv")
    parser.add_argument("--report-dir", default="reports_fast_indicator_combo_lab")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair = args.pair.upper().strip()
    report_root = core.build_report_dir(Path(args.report_dir), "fast_combo_lab")
    news_file = Path(args.news_file) if args.news_file else None

    design_results = run_stage(
        pair=pair,
        data_dir=Path(args.design_data_dir),
        start=args.design_start,
        end=args.design_end,
        news_file=news_file,
    )
    oos_results = run_stage(
        pair=pair,
        data_dir=Path(args.oos_data_dir),
        start=args.oos_start,
        end=args.oos_end,
        news_file=news_file,
    )

    for label, (spec, trades, equity_curve, summary) in design_results.items():
        export_result(report_root / "design", label, trades, equity_curve, summary, params={"label": label, "style": spec.style, "variant": spec.variant, "regime": spec.regime.name, "session": spec.session.name})
    for label, (spec, trades, equity_curve, summary) in oos_results.items():
        export_result(report_root / "oos", label, trades, equity_curve, summary, params={"label": label, "style": spec.style, "variant": spec.variant, "regime": spec.regime.name, "session": spec.session.name})

    ranking = build_ranking(design_results, oos_results)
    ranking.to_csv(report_root / "combo_ranking.csv", index=False)

    decision = {
        "pair": pair,
        "top_label": ranking.iloc[0]["label"] if not ranking.empty else None,
        "accepted_labels": ranking.loc[ranking["accepted"], "label"].tolist(),
        "spec_count": len(SPECS),
    }
    top_label = decision["top_label"]
    if top_label:
        top_spec = design_results[top_label][0]
        top_params = {"label": top_label, "style": top_spec.style, "variant": top_spec.variant, "regime": top_spec.regime.name, "session": top_spec.session.name}
        top_oos_summary = oos_results[top_label][3]
        top_oos_trades = oos_results[top_label][1]
        (report_root / "best_params.json").write_text(
            json.dumps(core.sanitize_for_json({"best_params": top_params, "score": float(ranking.iloc[0]["score"])}), indent=2),
            encoding="utf-8",
        )
        walkforward_rows = [
            {
                "train_period": f"{args.design_start}:{args.design_end}",
                "test_period": f"{args.oos_start}:{args.oos_end}",
                "params_used": json.dumps(core.sanitize_for_json(top_params), ensure_ascii=True),
                "trades": int(top_oos_summary.get("total_trades", 0)),
                "win_rate": float(top_oos_summary.get("win_rate_pct", 0.0)),
                "pnl_r": float(top_oos_trades["r_multiple"].sum()) if not top_oos_trades.empty else 0.0,
                "pnl_usd": float(top_oos_trades["net_pnl_usd"].sum()) if not top_oos_trades.empty else 0.0,
                "max_drawdown_pct": float(top_oos_summary.get("max_drawdown_pct", 0.0)),
                "profit_factor": float(top_oos_summary.get("profit_factor", 0.0)),
            }
        ]
        pd.DataFrame(walkforward_rows).to_csv(report_root / "walkforward_summary.csv", index=False)
    else:
        pd.DataFrame(
            columns=["train_period", "test_period", "params_used", "trades", "win_rate", "pnl_r", "pnl_usd", "max_drawdown_pct", "profit_factor"]
        ).to_csv(report_root / "walkforward_summary.csv", index=False)
        (report_root / "best_params.json").write_text(json.dumps({"best_params": None, "score": None}, indent=2), encoding="utf-8")

    (report_root / "decision.json").write_text(json.dumps(core.sanitize_for_json(decision), indent=2), encoding="utf-8")
    print("\n=== FAST COMBO RANKING ===")
    print(ranking.to_string(index=False))
    print("\n=== FAST COMBO DECISION ===")
    print(json.dumps(core.sanitize_for_json(decision), indent=2))
    print(f"\nReportes exportados en: {report_root}")


if __name__ == "__main__":
    main()
