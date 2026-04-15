from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from research_lab.config import (
    EngineConfig,
    INITIAL_CAPITAL,
    PAIR_META,
    SessionConfig,
    SESSION_VARIANTS,
    resolved_cost_profile,
    resolved_intrabar_policy,
    time_to_minute,
    with_execution_mode,
)


@dataclass
class Position:
    strategy_name: str
    direction: str
    entry_side: str
    signal_time: pd.Timestamp
    signal_price: float
    fill_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    sl: float
    tp: float | None
    units: float
    lots: float
    risk_usd: float
    initial_risk_distance: float
    entry_bar_index: int
    max_hold_bars: int | None
    break_even_at_r: float | None
    trailing_atr: bool
    trail_mult: float
    entry_commission_usd: float
    entry_spread_pips: float
    entry_slippage_pips: float
    execution_mode_used: str
    cost_profile_used: str
    entry_cost_regime: str
    intrabar_policy_used: str
    price_source_used: str
    data_source_used: str


@dataclass
class BacktestResult:
    strategy_name: str
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    params: dict[str, Any]
    news_filter_used: bool


def quote_to_usd(pair: str, pair_price: float) -> float:
    quote = PAIR_META[pair]["quote"]
    if quote == "USD":
        return 1.0
    if pair == "USDJPY":
        return 1.0 / pair_price if pair_price > 0 else np.nan
    raise ValueError(f"No hay conversion a USD implementada para {pair}")


def configured_spread_pips(engine_config: EngineConfig) -> float:
    if engine_config.assumed_spread_pips is not None:
        return float(engine_config.assumed_spread_pips)
    return float(PAIR_META[engine_config.pair]["default_spread_pips"])


def pip_to_price(pair: str, pips: float) -> float:
    return float(pips) * float(PAIR_META[pair]["pip_size"])


def slippage_price(pair: str, slippage_pips: float) -> float:
    return pip_to_price(pair, slippage_pips)


def session_cost_bucket(ts_local: pd.Timestamp, engine_config: EngineConfig) -> str:
    minute_value = ts_local.hour * 60 + ts_local.minute
    if minute_value < time_to_minute(engine_config.opening_session_end):
        return "opening"
    if minute_value >= time_to_minute(engine_config.late_session_start):
        return "late"
    return "core"


def volatility_cost_bucket(range_atr: float, engine_config: EngineConfig) -> str:
    return "high_vol" if np.isfinite(range_atr) and range_atr >= engine_config.high_vol_range_atr else "normal_vol"


def estimate_spread_pips(
    pair: str,
    ts_local: pd.Timestamp,
    range_atr: float,
    engine_config: EngineConfig,
    *,
    fill_kind: str = "entry",
) -> float:
    spread = configured_spread_pips(engine_config)
    profile = resolved_cost_profile(engine_config)
    session_bucket = session_cost_bucket(ts_local, engine_config)
    vol_bucket = volatility_cost_bucket(range_atr, engine_config)

    if profile == "stress":
        spread *= float(engine_config.stress_spread_multiplier)

    if profile == "precision" and session_bucket == "opening":
        spread *= float(engine_config.spread_opening_multiplier)

    if vol_bucket == "high_vol":
        spread *= float(engine_config.spread_high_vol_multiplier)

    if session_bucket == "late":
        spread *= float(engine_config.spread_late_session_multiplier)

    return float(spread if spread > 0 else PAIR_META[pair]["default_spread_pips"])


def spread_guard_allows(spread_pips: float, engine_config: EngineConfig) -> bool:
    if engine_config.max_spread_pips <= 0:
        return True
    return spread_pips <= float(engine_config.max_spread_pips)


def estimate_slippage_pips(
    ts_local: pd.Timestamp,
    range_atr: float,
    engine_config: EngineConfig,
    *,
    fill_kind: str,
) -> float:
    slippage = float(engine_config.slippage_pips)
    profile = resolved_cost_profile(engine_config)
    session_bucket = session_cost_bucket(ts_local, engine_config)
    vol_bucket = volatility_cost_bucket(range_atr, engine_config)

    if profile == "stress":
        slippage *= float(engine_config.stress_slippage_multiplier)

    if vol_bucket == "high_vol":
        slippage *= float(engine_config.slippage_high_vol_multiplier)

    if session_bucket == "late":
        slippage *= float(engine_config.slippage_late_session_multiplier)

    if profile == "precision" and session_bucket == "opening":
        slippage *= float(engine_config.slippage_opening_multiplier)

    if fill_kind == "stop_loss":
        slippage *= float(engine_config.slippage_stop_multiplier)
    elif fill_kind == "take_profit" and profile == "precision":
        slippage *= float(engine_config.slippage_target_multiplier)
    elif fill_kind == "forced_session_close" and profile == "precision":
        slippage *= float(engine_config.slippage_forced_close_multiplier)
    elif fill_kind == "final_bar_close" and profile == "precision":
        slippage *= float(engine_config.slippage_final_close_multiplier)

    return float(slippage)


def infer_bar_delta(index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(index) < 2:
        return pd.Timedelta(minutes=15)
    deltas = pd.Series(index[1:] - index[:-1])
    mode = deltas.mode()
    if mode.empty:
        return pd.Timedelta(minutes=15)
    return pd.Timedelta(mode.iloc[0])


def entry_open_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index - infer_bar_delta(index)


def session_window_from_params(params: dict[str, Any]) -> tuple[int, int]:
    session_name = params.get("session_name")
    if session_name in SESSION_VARIANTS:
        start, end = SESSION_VARIANTS[session_name]
    else:
        base_session = SessionConfig()
        start, end = base_session.entry_start, base_session.entry_end
    return time_to_minute(start), time_to_minute(end)


def entry_execution_price(pair: str, direction: str, bid_open: float, spread_pips: float, slippage_pips: float) -> float:
    spread_price = pip_to_price(pair, spread_pips)
    slip_price = slippage_price(pair, slippage_pips)
    if direction == "long":
        return bid_open + spread_price + slip_price
    return bid_open - slip_price


def exit_execution_price(pair: str, direction: str, bid_price: float, spread_pips: float, slippage_pips: float) -> float:
    spread_price = pip_to_price(pair, spread_pips)
    slip_price = slippage_price(pair, slippage_pips)
    if direction == "long":
        return bid_price - slip_price
    return bid_price + spread_price + slip_price


def stop_trigger_price(pair: str, direction: str, entry_bid_price: float, stop_atr: float, atr_value: float) -> float:
    distance = atr_value * float(stop_atr)
    if direction == "long":
        return entry_bid_price - distance
    return entry_bid_price + distance


def execution_to_trigger_price(pair: str, direction: str, execution_price_value: float, spread_pips: float, slippage_pips: float) -> float:
    spread_price = pip_to_price(pair, spread_pips)
    slip_price = slippage_price(pair, slippage_pips)
    if direction == "long":
        return execution_price_value + slip_price
    return execution_price_value - spread_price - slip_price


def build_fixed_rr_target(
    pair: str,
    direction: str,
    entry_price: float,
    risk_distance: float,
    target_rr: float,
    spread_pips: float,
    slippage_pips: float,
) -> float:
    if direction == "long":
        target_execution = entry_price + risk_distance * target_rr
    else:
        target_execution = entry_price - risk_distance * target_rr
    return execution_to_trigger_price(pair, direction, target_execution, spread_pips, slippage_pips)


def break_even_trigger_price(position: Position, pair: str) -> float:
    return execution_to_trigger_price(
        pair,
        position.direction,
        position.entry_price,
        position.entry_spread_pips,
        0.0,
    )

def mark_to_market_execution_price(pair: str, direction: str, bid_close: float, spread_pips: float) -> float:
    spread_price = pip_to_price(pair, spread_pips)
    if direction == "long":
        return bid_close
    return bid_close + spread_price


def execution_regime_label(ts_local: pd.Timestamp, range_atr: float, engine_config: EngineConfig, *, fill_kind: str) -> str:
    return "|".join(
        [
            resolved_cost_profile(engine_config),
            session_cost_bucket(ts_local, engine_config),
            volatility_cost_bucket(range_atr, engine_config),
            fill_kind,
        ]
    )


def resolve_intrabar_exit(
    *,
    direction: str,
    open_price: float,
    low_price: float,
    high_price: float,
    sl_trigger: float,
    tp_trigger: float | None,
    priority: str,
    intrabar_policy: str,
) -> tuple[str | None, float | None, bool, str | None]:
    if direction == "long":
        if open_price <= sl_trigger:
            return "stop_loss", open_price, False, "gap_stop"
        if tp_trigger is not None and open_price >= tp_trigger:
            return "take_profit", open_price, False, "gap_target"
    else:
        if open_price >= sl_trigger:
            return "stop_loss", open_price, False, "gap_stop"
        if tp_trigger is not None and open_price <= tp_trigger:
            return "take_profit", open_price, False, "gap_target"

    if direction == "long":
        stop_hit = low_price <= sl_trigger
        target_hit = tp_trigger is not None and high_price >= tp_trigger
    else:
        stop_hit = high_price >= sl_trigger
        target_hit = tp_trigger is not None and low_price <= tp_trigger

    if stop_hit and target_hit:
        if intrabar_policy == "conservative":
            return "stop_loss", sl_trigger, True, None
        if priority == "target_first":
            return "take_profit", tp_trigger, True, None
        return "stop_loss", sl_trigger, True, None
    if stop_hit:
        return "stop_loss", sl_trigger, False, None
    if target_hit:
        return "take_profit", tp_trigger, False, None
    return None, None, False, None


def actual_spread_pips(pair: str, bid_price: float, ask_price: float) -> float:
    pip_size = float(PAIR_META[pair]["pip_size"])
    if pip_size <= 0:
        raise ValueError(f"pip_size invalido para {pair}")
    return max(float((ask_price - bid_price) / pip_size), 0.0)


def high_precision_entry_execution_price(pair: str, direction: str, bid_open: float, ask_open: float, slippage_pips: float) -> float:
    slip_price = slippage_price(pair, slippage_pips)
    if direction == "long":
        return ask_open + slip_price
    return bid_open - slip_price


def high_precision_exit_execution_price(pair: str, direction: str, trigger_price: float, slippage_pips: float) -> float:
    slip_price = slippage_price(pair, slippage_pips)
    if direction == "long":
        return trigger_price - slip_price
    return trigger_price + slip_price


def high_precision_stop_trigger_price(direction: str, bid_open: float, ask_open: float, stop_atr: float, atr_value: float) -> float:
    distance = atr_value * float(stop_atr)
    if direction == "long":
        return bid_open - distance
    return ask_open + distance


def high_precision_trigger_from_execution(pair: str, direction: str, execution_price_value: float, slippage_pips: float) -> float:
    slip_price = slippage_price(pair, slippage_pips)
    if direction == "long":
        return execution_price_value + slip_price
    return execution_price_value - slip_price


def build_fixed_rr_target_high_precision(
    pair: str,
    direction: str,
    entry_price: float,
    risk_distance: float,
    target_rr: float,
    slippage_pips: float,
) -> float:
    if direction == "long":
        target_execution = entry_price + risk_distance * target_rr
    else:
        target_execution = entry_price - risk_distance * target_rr
    return high_precision_trigger_from_execution(pair, direction, target_execution, slippage_pips)


def break_even_trigger_price_high_precision(position: Position) -> float:
    return float(position.entry_price)


def mark_to_market_execution_price_high_precision(direction: str, bid_close: float, ask_close: float) -> float:
    if direction == "long":
        return float(bid_close)
    return float(ask_close)


def precision_intrabar_slice(
    precision_package: dict[str, pd.DataFrame],
    *,
    bar_close_time: pd.Timestamp,
    bar_delta: pd.Timedelta,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bar_open_time = bar_close_time - bar_delta
    bid_m1 = precision_package["bid_m1"]
    ask_m1 = precision_package["ask_m1"]
    mask = (bid_m1.index > bar_open_time) & (bid_m1.index <= bar_close_time)
    bid_slice = bid_m1.loc[mask]
    ask_slice = ask_m1.loc[mask]
    return bid_slice, ask_slice


def run_backtest(
    strategy_module: Any,
    frame: pd.DataFrame,
    params: dict[str, Any],
    engine_config: EngineConfig,
    news_block: np.ndarray,
    news_filter_used: bool,
    *,
    precision_package: dict[str, pd.DataFrame] | None = None,
    data_source_used: str | None = None,
) -> BacktestResult:
    engine_config = with_execution_mode(engine_config, engine_config.execution_mode)
    if frame.empty or len(frame) <= getattr(strategy_module, "WARMUP_BARS", 0) + 1:
        return BacktestResult(
            strategy_name=strategy_module.NAME,
            trades=pd.DataFrame(),
            equity_curve=pd.DataFrame(columns=["timestamp", "equity"]),
            params=params,
            news_filter_used=news_filter_used,
        )

    session = SessionConfig()
    force_close_minute = time_to_minute(session.force_close)
    entry_start_minute, entry_end_minute = session_window_from_params(params)
    intrabar_policy_used = resolved_intrabar_policy(engine_config)
    precision_enabled = engine_config.execution_mode == "high_precision_mode"

    pair = engine_config.pair
    lot_size = float(PAIR_META[pair]["lot_size"])
    local_index = frame.index
    timestamp_utc = frame.index.tz_convert("UTC")
    bar_open_local = entry_open_index(local_index)
    bar_open_utc = bar_open_local.tz_convert("UTC")
    minute_values = (local_index.hour * 60 + local_index.minute).to_numpy()
    entry_open_minutes = (bar_open_local.hour * 60 + bar_open_local.minute).to_numpy()
    session_dates = np.array(bar_open_local.date)
    open_ = frame["open"].to_numpy()
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    close = frame["close"].to_numpy()
    atr14 = frame["atr14"].to_numpy()
    range_atr = frame["range_atr"].to_numpy()
    resolved_data_source = data_source_used or ("dukascopy_m1_bid_ask_full" if precision_enabled else "prepared_m5_bid")

    bid_m15_open = ask_m15_open = bid_m15_high = ask_m15_high = bid_m15_low = ask_m15_low = bid_m15_close = ask_m15_close = None
    if precision_enabled:
        if precision_package is None:
            raise ValueError("high_precision_mode requiere precision_package con BID/ASK reales.")
        required_keys = {"bid_m1", "ask_m1", "bid_m15", "ask_m15"}
        missing_keys = sorted(required_keys - set(precision_package))
        if missing_keys:
            raise ValueError(f"precision_package incompleto para high_precision_mode: faltan {missing_keys}")
        if not precision_package["bid_m15"].index.equals(frame.index) or not precision_package["ask_m15"].index.equals(frame.index):
            raise ValueError("El paquete M1 BID/ASK no esta alineado con el frame M15 del backtest.")
        bid_m15_open = precision_package["bid_m15"]["open"].to_numpy()
        ask_m15_open = precision_package["ask_m15"]["open"].to_numpy()
        bid_m15_high = precision_package["bid_m15"]["high"].to_numpy()
        ask_m15_high = precision_package["ask_m15"]["high"].to_numpy()
        bid_m15_low = precision_package["bid_m15"]["low"].to_numpy()
        ask_m15_low = precision_package["ask_m15"]["low"].to_numpy()
        bid_m15_close = precision_package["bid_m15"]["close"].to_numpy()
        ask_m15_close = precision_package["ask_m15"]["close"].to_numpy()

    fill_allowed = (
        (entry_open_minutes >= entry_start_minute)
        & (entry_open_minutes <= entry_end_minute)
        & (entry_open_minutes < force_close_minute)
    )
    force_close_mask = minute_values >= force_close_minute

    cash = INITIAL_CAPITAL
    position: Position | None = None
    pending_signal: dict[str, Any] | None = None
    cooldown_until_index = -1
    opened_total_by_date: dict[Any, int] = {}
    trades: list[dict[str, Any]] = []
    equity_points: list[dict[str, Any]] = [{"timestamp": timestamp_utc[0], "equity": INITIAL_CAPITAL}]

    for i in range(strategy_module.WARMUP_BARS, len(frame)):
        ts_utc = timestamp_utc[i]
        session_date = session_dates[i]

        if pending_signal is not None and i == pending_signal["signal_index"] + 1:
            entry_slippage_pips = estimate_slippage_pips(bar_open_local[i], range_atr[pending_signal["signal_index"]], engine_config, fill_kind="entry")
            entry_cost_regime = execution_regime_label(bar_open_local[i], range_atr[pending_signal["signal_index"]], engine_config, fill_kind="entry")
            if precision_enabled:
                entry_bid_price = float(bid_m15_open[i])
                entry_ask_price = float(ask_m15_open[i])
                entry_spread_pips = actual_spread_pips(pair, entry_bid_price, entry_ask_price)
            else:
                entry_spread_pips = estimate_spread_pips(pair, bar_open_local[i], range_atr[pending_signal["signal_index"]], engine_config, fill_kind="entry")
            if (
                fill_allowed[i]
                and not news_block[i]
                and np.isfinite(atr14[pending_signal["signal_index"]])
                and atr14[pending_signal["signal_index"]] > 0
                and range_atr[pending_signal["signal_index"]] <= engine_config.shock_candle_atr_max
                and spread_guard_allows(entry_spread_pips, engine_config)
            ):
                if not precision_enabled:
                    entry_bid_price = float(open_[i])
                    entry_price = entry_execution_price(pair, pending_signal["direction"], entry_bid_price, entry_spread_pips, entry_slippage_pips)
                else:
                    entry_price = high_precision_entry_execution_price(
                        pair,
                        pending_signal["direction"],
                        entry_bid_price,
                        entry_ask_price,
                        entry_slippage_pips,
                    )
                if pending_signal["stop_mode"] == "price":
                    sl_trigger = float(pending_signal["stop_price"])
                else:
                    if precision_enabled:
                        sl_trigger = high_precision_stop_trigger_price(
                            pending_signal["direction"],
                            entry_bid_price,
                            entry_ask_price,
                            float(pending_signal["stop_atr"]),
                            atr14[pending_signal["signal_index"]],
                        )
                    else:
                        sl_trigger = stop_trigger_price(
                            pair,
                            pending_signal["direction"],
                            entry_bid_price,
                            float(pending_signal["stop_atr"]),
                            atr14[pending_signal["signal_index"]],
                        )
                stop_slippage_pips = estimate_slippage_pips(
                    bar_open_local[i],
                    range_atr[pending_signal["signal_index"]],
                    engine_config,
                    fill_kind="stop_loss",
                )
                if precision_enabled:
                    stop_execution = high_precision_exit_execution_price(pair, pending_signal["direction"], sl_trigger, stop_slippage_pips)
                else:
                    stop_execution = exit_execution_price(
                        pair,
                        pending_signal["direction"],
                        sl_trigger,
                        entry_spread_pips,
                        stop_slippage_pips,
                    )
                stop_distance = abs(entry_price - stop_execution)

                quote_to_usd_rate = quote_to_usd(pair, entry_price)
                risk_usd = cash * (engine_config.risk_pct / 100.0)
                if np.isfinite(stop_distance) and stop_distance > 0 and np.isfinite(quote_to_usd_rate) and quote_to_usd_rate > 0:
                    units = risk_usd / (stop_distance * quote_to_usd_rate)
                    lots = units / lot_size
                    if units > 0 and lots > 0:
                        entry_commission_usd = (engine_config.commission_per_lot_roundturn_usd * lots) / 2.0
                        cash -= entry_commission_usd
                        tp_trigger = None
                        if pending_signal.get("target_mode") == "price":
                            tp_trigger = float(pending_signal["target_price"])
                        else:
                            if precision_enabled:
                                tp_trigger = build_fixed_rr_target_high_precision(
                                    pair,
                                    pending_signal["direction"],
                                    entry_price,
                                    stop_distance,
                                    float(pending_signal["target_rr"]),
                                    entry_slippage_pips,
                                )
                            else:
                                tp_trigger = build_fixed_rr_target(
                                    pair,
                                    pending_signal["direction"],
                                    entry_price,
                                    stop_distance,
                                    float(pending_signal["target_rr"]),
                                    entry_spread_pips,
                                    entry_slippage_pips,
                                )
                        position = Position(
                            strategy_name=strategy_module.NAME,
                            direction=pending_signal["direction"],
                            entry_side="buy" if pending_signal["direction"] == "long" else "sell",
                            signal_time=timestamp_utc[pending_signal["signal_index"]],
                            signal_price=float(pending_signal.get("signal_price", close[pending_signal["signal_index"]])),
                            fill_time=bar_open_utc[i],
                            entry_time=bar_open_utc[i],
                            entry_price=entry_price,
                            sl=sl_trigger,
                            tp=tp_trigger,
                            units=float(units),
                            lots=float(lots),
                            risk_usd=risk_usd,
                            initial_risk_distance=stop_distance,
                            entry_bar_index=i,
                            max_hold_bars=int(pending_signal["max_hold_bars"]) if pending_signal.get("max_hold_bars") is not None else None,
                            break_even_at_r=pending_signal.get("break_even_at_r"),
                            trailing_atr=bool(pending_signal.get("trailing_atr", False)),
                            trail_mult=float(pending_signal.get("stop_atr", 1.0)),
                            entry_commission_usd=entry_commission_usd,
                            entry_spread_pips=entry_spread_pips,
                            entry_slippage_pips=entry_slippage_pips,
                            execution_mode_used=engine_config.execution_mode,
                            cost_profile_used=resolved_cost_profile(engine_config),
                            entry_cost_regime=entry_cost_regime,
                            intrabar_policy_used=intrabar_policy_used,
                            price_source_used="bid_ask_real" if precision_enabled else engine_config.price_source,
                            data_source_used=resolved_data_source,
                        )
                        opened_total_by_date[session_date] = opened_total_by_date.get(session_date, 0) + 1
            pending_signal = None

        if position is not None:
            if position.break_even_at_r is not None and position.initial_risk_distance > 0:
                trigger_distance = position.initial_risk_distance * float(position.break_even_at_r)
                if precision_enabled:
                    if position.direction == "long" and float(bid_m15_high[i]) >= position.entry_price + trigger_distance:
                        position.sl = max(position.sl, break_even_trigger_price_high_precision(position))
                    elif position.direction == "short" and float(ask_m15_low[i]) <= position.entry_price - trigger_distance:
                        position.sl = min(position.sl, break_even_trigger_price_high_precision(position))
                else:
                    if position.direction == "long" and high[i] >= position.entry_price + trigger_distance:
                        position.sl = max(position.sl, break_even_trigger_price(position, pair))
                    elif position.direction == "short" and low[i] <= position.entry_price - trigger_distance:
                        position.sl = min(position.sl, break_even_trigger_price(position, pair))

            if position.trailing_atr and np.isfinite(atr14[i]) and atr14[i] > 0:
                if position.direction == "long":
                    position.sl = max(position.sl, close[i] - atr14[i] * position.trail_mult)
                else:
                    position.sl = min(position.sl, close[i] + atr14[i] * position.trail_mult)

            exit_reason = None
            exit_signal_price = None
            exit_ts_utc = ts_utc
            if precision_enabled:
                exit_spread_pips = actual_spread_pips(pair, float(bid_m15_close[i]), float(ask_m15_close[i]))
            else:
                exit_spread_pips = estimate_spread_pips(pair, local_index[i], range_atr[i], engine_config, fill_kind="mark_to_market")
            exit_cost_regime = execution_regime_label(local_index[i], range_atr[i], engine_config, fill_kind="mark_to_market")
            intrabar_ambiguity_flag = False
            gap_exit_type = None

            if precision_enabled:
                bid_slice, ask_slice = precision_intrabar_slice(
                    precision_package,
                    bar_close_time=local_index[i],
                    bar_delta=infer_bar_delta(local_index),
                )
                common_slice_index = bid_slice.index.intersection(ask_slice.index)
                for precision_ts in common_slice_index:
                    side_frame = bid_slice if position.direction == "long" else ask_slice
                    side_open = float(side_frame.at[precision_ts, "open"])
                    side_low = float(side_frame.at[precision_ts, "low"])
                    side_high = float(side_frame.at[precision_ts, "high"])
                    maybe_reason, maybe_signal_price, maybe_ambiguous, maybe_gap = resolve_intrabar_exit(
                        direction=position.direction,
                        open_price=side_open,
                        low_price=side_low,
                        high_price=side_high,
                        sl_trigger=position.sl,
                        tp_trigger=position.tp,
                        priority=engine_config.intrabar_exit_priority,
                        intrabar_policy=position.intrabar_policy_used,
                    )
                    if maybe_reason is None or maybe_signal_price is None:
                        continue
                    exit_reason = maybe_reason
                    exit_signal_price = float(maybe_signal_price)
                    intrabar_ambiguity_flag = bool(maybe_ambiguous)
                    gap_exit_type = maybe_gap
                    exit_ts_utc = precision_ts.tz_convert("UTC")
                    exit_spread_pips = actual_spread_pips(
                        pair,
                        float(bid_slice.at[precision_ts, "close"]),
                        float(ask_slice.at[precision_ts, "close"]),
                    )
                    break 
            else:
                exit_reason, exit_signal_price, intrabar_ambiguity_flag, gap_exit_type = resolve_intrabar_exit(
                    direction=position.direction,
                    open_price=open_[i],
                    low_price=low[i],
                    high_price=high[i],
                    sl_trigger=position.sl,
                    tp_trigger=position.tp,
                    priority=engine_config.intrabar_exit_priority,
                    intrabar_policy=position.intrabar_policy_used,
                )

            # PRIORIDAD 2: Si no hubo hit de SL/TP/Gap, chequear cierre forzado por horario
            if exit_reason is None and force_close_mask[i]:
                exit_reason = "forced_session_close"
                exit_signal_price = float(bid_m15_close[i] if position.direction == "long" else ask_m15_close[i]) if precision_enabled else close[i]

            # PRIORIDAD 3: Si no hay nada de lo anterior, chequear limite de tiempo
            if exit_reason is None and position.max_hold_bars is not None:
                held_bars = i - position.entry_bar_index
                if held_bars >= position.max_hold_bars:
                    exit_reason = "time_exit"
                    exit_signal_price = float(bid_m15_close[i] if position.direction == "long" else ask_m15_close[i]) if precision_enabled else close[i]

            if exit_reason is not None and exit_signal_price is not None:
                exit_slippage_pips = estimate_slippage_pips(local_index[i], range_atr[i], engine_config, fill_kind=exit_reason)
                if intrabar_ambiguity_flag and position.intrabar_policy_used == "conservative":
                    exit_slippage_pips *= float(engine_config.ambiguity_slippage_multiplier)
                exit_cost_regime = execution_regime_label(local_index[i], range_atr[i], engine_config, fill_kind=exit_reason)
                if precision_enabled:
                    exit_price = high_precision_exit_execution_price(pair, position.direction, exit_signal_price, exit_slippage_pips)
                else:
                    exit_spread_pips = estimate_spread_pips(pair, local_index[i], range_atr[i], engine_config, fill_kind=exit_reason)
                    exit_price = exit_execution_price(pair, position.direction, exit_signal_price, exit_spread_pips, exit_slippage_pips)
                price_delta = exit_price - position.entry_price
                pnl_quote = price_delta * position.units if position.direction == "long" else -price_delta * position.units
                pnl_usd = pnl_quote * quote_to_usd(pair, exit_price)
                exit_commission_usd = (engine_config.commission_per_lot_roundturn_usd * position.lots) / 2.0
                pnl_usd -= exit_commission_usd
                pnl_r = pnl_usd / position.risk_usd if position.risk_usd else 0.0
                cash += pnl_usd
                trades.append(
                    {
                        "pair": pair,
                        "strategy_name": strategy_module.NAME,
                        "entry_side": position.entry_side,
                        "signal_time": position.signal_time,
                        "signal_price": position.signal_price,
                        "fill_time": position.fill_time,
                        "entry_time": position.entry_time,
                        "exit_time": exit_ts_utc,
                        "direction": position.direction,
                        "entry_price": position.entry_price,
                        "fill_price": position.entry_price,
                        "exit_price": exit_price,
                        "exit_signal_price": exit_signal_price,
                        "exit_fill_price": exit_price,
                        "sl": position.sl,
                        "tp": position.tp,
                        "spread_applied": position.entry_spread_pips + exit_spread_pips,
                        "slippage_applied": position.entry_slippage_pips + exit_slippage_pips,
                        "commission_applied": position.entry_commission_usd + exit_commission_usd,
                        "entry_spread_pips": position.entry_spread_pips,
                        "exit_spread_pips": exit_spread_pips,
                        "entry_slippage_pips": position.entry_slippage_pips,
                        "exit_slippage_pips": exit_slippage_pips,
                        "entry_commission_usd": position.entry_commission_usd,
                        "exit_commission_usd": exit_commission_usd,
                        "price_source_used": position.price_source_used,
                        "data_source_used": position.data_source_used,
                        "pnl_r": pnl_r,
                        "pnl_usd": pnl_usd,
                        "exit_reason": exit_reason,
                        "commission_usd": position.entry_commission_usd + exit_commission_usd,
                        "forced_close_flag": bool(exit_reason == "forced_session_close"),
                        "intrabar_ambiguity_flag": bool(intrabar_ambiguity_flag),
                        "gap_exit_flag": bool(gap_exit_type is not None),
                        "gap_exit_type": gap_exit_type if gap_exit_type else "no_gap",
                        "execution_mode_used": position.execution_mode_used,
                        "intrabar_policy_used": position.intrabar_policy_used,
                        "cost_profile_used": position.cost_profile_used,
                        "entry_cost_regime": position.entry_cost_regime,
                        "exit_cost_regime": exit_cost_regime,
                        "blocked_by_news": False,
                        "blocking_event_name": "",
                        "blocking_event_time_ny": "",
                        "blocking_rule_used": "",
                    }
                )
                cooldown_until_index = i + int(params.get("cooldown_bars", 0))
                position = None

        if position is None and pending_signal is None and i < len(frame) - 1:
            if i <= cooldown_until_index or not np.isfinite(range_atr[i]) or range_atr[i] > engine_config.shock_candle_atr_max:
                pass
            elif opened_total_by_date.get(session_date, 0) >= engine_config.max_trades_per_day:
                pass
            elif not fill_allowed[i + 1]:
                pass
            elif precision_enabled and not spread_guard_allows(actual_spread_pips(pair, float(bid_m15_open[i + 1]), float(ask_m15_open[i + 1])), engine_config):
                pass
            elif not precision_enabled and not spread_guard_allows(estimate_spread_pips(pair, bar_open_local[i + 1], range_atr[i], engine_config, fill_kind="entry"), engine_config):
                pass
            else:
                signal = strategy_module.signal(frame, i, params)
                if signal is not None:
                    signal["signal_index"] = i
                    signal["signal_price"] = float(close[i])
                    pending_signal = signal

        mark_equity = cash
        if position is not None:
            if precision_enabled:
                mark_execution = mark_to_market_execution_price_high_precision(position.direction, float(bid_m15_close[i]), float(ask_m15_close[i]))
            else:
                mark_execution = mark_to_market_execution_price(pair, position.direction, close[i], estimate_spread_pips(pair, local_index[i], range_atr[i], engine_config, fill_kind="mark_to_market"))
            unrealized_quote = (mark_execution - position.entry_price) * position.units if position.direction == "long" else (position.entry_price - mark_execution) * position.units
            unrealized_usd = unrealized_quote * quote_to_usd(pair, mark_execution)
            unrealized_usd -= (engine_config.commission_per_lot_roundturn_usd * position.lots) / 2.0
            mark_equity += unrealized_usd
        equity_points.append({"timestamp": ts_utc, "equity": mark_equity})

    if position is not None:
        final_ts = timestamp_utc[-1]
        final_cost_regime = execution_regime_label(local_index[-1], range_atr[-1], engine_config, fill_kind="final_bar_close")
        final_slippage_pips = estimate_slippage_pips(local_index[-1], range_atr[-1], engine_config, fill_kind="final_bar_close")
        final_gap_exit_flag = False
        final_gap_exit_type = "no_gap"
        
        if precision_enabled:
            final_signal = float(bid_m15_close[-1] if position.direction == "long" else ask_m15_close[-1])
            final_spread_pips = actual_spread_pips(pair, float(bid_m15_close[-1]), float(ask_m15_close[-1]))
            final_exit = high_precision_exit_execution_price(pair, position.direction, final_signal, final_slippage_pips)
            final_exit_reason = "final_bar_close"
        else:
            maybe_reason, maybe_price, _, maybe_gap = resolve_intrabar_exit(
                direction=position.direction,
                open_price=open_[-1],
                low_price=low[-1],
                high_price=high[-1],
                sl_trigger=position.sl,
                tp_trigger=position.tp,
                priority=engine_config.intrabar_exit_priority,
                intrabar_policy=position.intrabar_policy_used,
            )
            if maybe_reason is not None:
                final_exit_reason = maybe_reason
                final_signal = maybe_price
                final_gap_exit_flag = bool(maybe_gap is not None)
                final_gap_exit_type = maybe_gap if maybe_gap else "no_gap"
            else:
                final_exit_reason = "final_bar_close"
                final_signal = float(close[-1])
            
            final_spread_pips = estimate_spread_pips(pair, local_index[-1], range_atr[-1], engine_config, fill_kind=final_exit_reason)
            final_exit = exit_execution_price(pair, position.direction, final_signal, final_spread_pips, final_slippage_pips)
        pnl_quote = (final_exit - position.entry_price) * position.units if position.direction == "long" else (position.entry_price - final_exit) * position.units
        pnl_usd = pnl_quote * quote_to_usd(pair, final_exit)
        exit_commission_usd = (engine_config.commission_per_lot_roundturn_usd * position.lots) / 2.0
        pnl_usd -= exit_commission_usd
        pnl_r = pnl_usd / position.risk_usd if position.risk_usd else 0.0
        cash += pnl_usd
        trades.append(
            {
                "pair": pair,
                "strategy_name": strategy_module.NAME,
                "entry_side": position.entry_side,
                "signal_time": position.signal_time,
                "signal_price": position.signal_price,
                "fill_time": position.fill_time,
                "entry_time": position.entry_time,
                "exit_time": final_ts,
                "direction": position.direction,
                "entry_price": position.entry_price,
                "fill_price": position.entry_price,
                "exit_price": final_exit,
                "exit_signal_price": final_signal,
                "exit_fill_price": final_exit,
                "sl": position.sl,
                "tp": position.tp,
                "spread_applied": position.entry_spread_pips + final_spread_pips,
                "slippage_applied": position.entry_slippage_pips + final_slippage_pips,
                "commission_applied": position.entry_commission_usd + exit_commission_usd,
                "entry_spread_pips": position.entry_spread_pips,
                "exit_spread_pips": final_spread_pips,
                "entry_slippage_pips": position.entry_slippage_pips,
                "exit_slippage_pips": final_slippage_pips,
                "entry_commission_usd": position.entry_commission_usd,
                "exit_commission_usd": exit_commission_usd,
                "price_source_used": position.price_source_used,
                "data_source_used": position.data_source_used,
                "pnl_r": pnl_r,
                "pnl_usd": pnl_usd,
                "exit_reason": final_exit_reason,
                "commission_usd": position.entry_commission_usd + exit_commission_usd,
                "forced_close_flag": False,
                "intrabar_ambiguity_flag": False,
                "gap_exit_flag": final_gap_exit_flag,
                "gap_exit_type": final_gap_exit_type,
                "execution_mode_used": position.execution_mode_used,
                "intrabar_policy_used": position.intrabar_policy_used,
                "cost_profile_used": position.cost_profile_used,
                "entry_cost_regime": position.entry_cost_regime,
                "exit_cost_regime": final_cost_regime,
                "blocked_by_news": False,
                "blocking_event_name": "",
                "blocking_event_time_ny": "",
                "blocking_rule_used": "",
            }
        )
        equity_points.append({"timestamp": final_ts, "equity": cash})

    return BacktestResult(
        strategy_name=strategy_module.NAME,
        trades=pd.DataFrame(trades),
        equity_curve=pd.DataFrame(equity_points),
        params=params,
        news_filter_used=news_filter_used,
    )
