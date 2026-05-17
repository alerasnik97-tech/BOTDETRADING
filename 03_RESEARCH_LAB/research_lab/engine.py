from __future__ import annotations

from dataclasses import dataclass, replace
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
    NY_TZ,
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


D5_TELEMETRY_VERSION = "d5_core_telemetry_v1"


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
    elif fill_kind == "stop_entry":
        slippage *= float(engine_config.slippage_stop_entry_multiplier)
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


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _empty_news_details(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "blocked": np.zeros(len(index), dtype=bool),
            "entry_blocked": np.zeros(len(index), dtype=bool),
            "cooldown_blocked": np.zeros(len(index), dtype=bool),
            "pending_kill": np.zeros(len(index), dtype=bool),
            "force_flat": np.zeros(len(index), dtype=bool),
            "blocking_event_name": [""] * len(index),
            "blocking_event_time_ny": [""] * len(index),
            "blocking_rule_used": [""] * len(index),
            "entry_event_name": [""] * len(index),
            "entry_event_time_ny": [""] * len(index),
            "entry_rule_used": [""] * len(index),
            "pending_event_name": [""] * len(index),
            "pending_event_time_ny": [""] * len(index),
            "pending_rule_used": [""] * len(index),
            "force_flat_event_name": [""] * len(index),
            "force_flat_event_time_ny": [""] * len(index),
            "force_flat_rule_used": [""] * len(index),
        },
        index=index,
    )


def validate_signal_risk_contract(signal: dict[str, Any], *, signal_price: float, engine_config: EngineConfig) -> dict[str, Any]:
    validated = dict(signal)
    direction = str(validated.get("direction", "")).strip().lower()
    if direction not in {"long", "short"}:
        raise ValueError(f"Signal invalida: direction debe ser 'long' o 'short', recibida={validated.get('direction')!r}")
    validated["direction"] = direction

    entry_mode = str(validated.get("entry_mode", "market")).strip().lower()
    validated["entry_mode"] = entry_mode
    entry_reference_price = float(signal_price)
    if entry_mode == "limit":
        limit_price = _safe_float(validated.get("limit_price"))
        if limit_price is None:
            raise ValueError("Signal invalida: entry_mode='limit' requiere limit_price numerico.")
        validated["limit_price"] = limit_price
        entry_reference_price = limit_price
    elif entry_mode == "stop":
        stop_entry_price = _safe_float(validated.get("stop_entry_price"))
        if stop_entry_price is None:
            raise ValueError("Signal invalida: entry_mode='stop' requiere stop_entry_price numerico.")
        if direction == "long" and stop_entry_price <= signal_price:
            raise ValueError("Signal invalida: en long el stop entry debe quedar por encima del precio de senal.")
        if direction == "short" and stop_entry_price >= signal_price:
            raise ValueError("Signal invalida: en short el stop entry debe quedar por debajo del precio de senal.")
        validated["stop_entry_price"] = stop_entry_price
        entry_reference_price = stop_entry_price
    else:
        # Market entry
        entry_reference_price = signal_price

    if not getattr(engine_config, "enforce_hard_stop", True):
        return validated

    stop_mode = str(validated.get("stop_mode", "")).strip().lower()
    validated["stop_mode"] = stop_mode
    if stop_mode == "price":
        stop_price = _safe_float(validated.get("stop_price"))
        if stop_price is None:
            raise ValueError("Signal invalida: stop_mode='price' requiere stop_price numerico finito.")
        if direction == "long" and stop_price >= entry_reference_price:
            raise ValueError("Signal invalida: en long el stop hard debe quedar por debajo del precio de señal.")
        if direction == "short" and stop_price <= entry_reference_price:
            raise ValueError("Signal invalida: en short el stop hard debe quedar por encima del precio de señal.")
        validated["stop_price"] = stop_price
    elif stop_mode == "atr":
        stop_atr = _safe_float(validated.get("stop_atr"))
        if stop_atr is None or stop_atr <= 0:
            raise ValueError("Signal invalida: stop_mode='atr' requiere stop_atr > 0.")
        validated["stop_atr"] = stop_atr
    else:
        raise ValueError("Signal invalida: falta un hard stop valido (stop_mode debe ser 'price' o 'atr').")

    target_mode = str(validated.get("target_mode", "rr")).strip().lower()
    validated["target_mode"] = target_mode
    if target_mode == "price":
        target_price = _safe_float(validated.get("target_price"))
        if target_price is None:
            raise ValueError("Signal invalida: target_mode='price' requiere target_price numerico finito.")
        if direction == "long" and target_price <= entry_reference_price:
            raise ValueError("Signal invalida: en long el target por precio debe quedar por encima del precio de señal.")
        if direction == "short" and target_price >= entry_reference_price:
            raise ValueError("Signal invalida: en short el target por precio debe quedar por debajo del precio de señal.")
        validated["target_price"] = target_price
    elif target_mode == "rr":
        target_rr = _safe_float(validated.get("target_rr"))
        if target_rr is None or target_rr <= 0:
            raise ValueError("Signal invalida: target_mode='rr' requiere target_rr > 0.")
        validated["target_rr"] = target_rr
    else:
        raise ValueError(f"Signal invalida: target_mode no soportado: {target_mode!r}")

    max_hold_bars = validated.get("max_hold_bars")
    if max_hold_bars is not None and int(max_hold_bars) <= 0:
        raise ValueError("Signal invalida: max_hold_bars debe ser positivo si se informa.")

    break_even_at_r = validated.get("break_even_at_r")
    if break_even_at_r is not None:
        break_even_numeric = _safe_float(break_even_at_r)
        if break_even_numeric is None or break_even_numeric <= 0:
            raise ValueError("Signal invalida: break_even_at_r debe ser > 0 si se informa.")
        validated["break_even_at_r"] = break_even_numeric

    return validated


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


def directional_pnl_usd(
    direction: str,
    entry_price: float,
    exit_price: float,
    units: float,
    quote_to_usd_rate: float,
) -> float:
    # units is unsigned (risk_usd / abs(stop_distance)); a short profits when price falls.
    direction_sign = 1.0 if direction == "long" else -1.0
    return direction_sign * (exit_price - entry_price) * units * quote_to_usd_rate


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


def resolve_stop_entry_fill(
    pair: str,
    direction: str,
    *,
    open_price: float,
    high_price: float,
    low_price: float,
    stop_entry_price: float,
    spread_pips: float,
    slippage_pips: float,
) -> tuple[float, float] | None:
    if direction == "long":
        if open_price >= stop_entry_price:
            entry_bid_price = float(open_price)
        elif high_price >= stop_entry_price:
            entry_bid_price = float(stop_entry_price)
        else:
            return None
    else:
        if open_price <= stop_entry_price:
            entry_bid_price = float(open_price)
        elif low_price <= stop_entry_price:
            entry_bid_price = float(stop_entry_price)
        else:
            return None
    entry_price = entry_execution_price(pair, direction, entry_bid_price, spread_pips, slippage_pips)
    return entry_bid_price, entry_price


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


def resolve_high_precision_stop_entry_fill(
    pair: str,
    direction: str,
    *,
    stop_entry_price: float,
    bid_slice: pd.DataFrame,
    ask_slice: pd.DataFrame,
    slippage_pips: float,
) -> tuple[float, float, float, float] | None:
    pip_size = float(PAIR_META[pair]["pip_size"])
    common_slice_index = bid_slice.index.intersection(ask_slice.index)
    for precision_ts in common_slice_index:
        bid_open = float(bid_slice.at[precision_ts, "open"])
        ask_open = float(ask_slice.at[precision_ts, "open"])
        minute_spread_pips = actual_spread_pips(pair, bid_open, ask_open)

        if direction == "long":
            if ask_open >= stop_entry_price:
                bid_fill_ref = bid_open
                ask_fill_ref = ask_open
            elif float(ask_slice.at[precision_ts, "high"]) >= stop_entry_price:
                ask_fill_ref = float(stop_entry_price)
                bid_fill_ref = ask_fill_ref - (minute_spread_pips * pip_size)
            else:
                continue
        else:
            if bid_open <= stop_entry_price:
                bid_fill_ref = bid_open
                ask_fill_ref = ask_open
            elif float(bid_slice.at[precision_ts, "low"]) <= stop_entry_price:
                bid_fill_ref = float(stop_entry_price)
                ask_fill_ref = bid_fill_ref + (minute_spread_pips * pip_size)
            else:
                continue
        entry_spread_pips = actual_spread_pips(pair, bid_fill_ref, ask_fill_ref)
        entry_price = high_precision_entry_execution_price(pair, direction, bid_fill_ref, ask_fill_ref, slippage_pips)
        return bid_fill_ref, ask_fill_ref, entry_spread_pips, entry_price

    return None


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
    news_events: pd.DataFrame | None = None,
    news_settings: Any | None = None,
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
    
    # Global override for window studies
    if engine_config.session_cutoff:
        cutoff_min = time_to_minute(engine_config.session_cutoff)
        force_close_minute = cutoff_min
        entry_end_minute = min(entry_end_minute, cutoff_min)
        
    intrabar_policy_used = resolved_intrabar_policy(engine_config)
    precision_enabled = engine_config.execution_mode == "high_precision_mode"

    pair = engine_config.pair
    lot_size = float(PAIR_META[pair]["lot_size"])
    # HARDENED: Explicit NY Timezone awareness to handle DST properly
    local_index = frame.index.tz_convert(NY_TZ) if frame.index.tz is not None else frame.index.tz_localize("UTC").tz_convert(NY_TZ)
    bar_delta = infer_bar_delta(local_index)
    timestamp_utc = local_index.tz_convert("UTC")
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

    from research_lab.news_filter import build_news_guard_details
    if news_events is not None and news_settings is not None and not news_events.empty:
        news_details = build_news_guard_details(bar_open_local, news_events, news_settings)
        entry_news_block = news_details["entry_blocked"].to_numpy(dtype=bool)
        pending_kill_mask = news_details["pending_kill"].to_numpy(dtype=bool)
        force_flat_mask = news_details["force_flat"].to_numpy(dtype=bool)
    else:
        news_details = _empty_news_details(bar_open_local)
        entry_news_block = np.asarray(news_block, dtype=bool)
        pending_kill_mask = np.asarray(news_block, dtype=bool)
        force_flat_mask = np.asarray(news_block, dtype=bool)

    cancel_pending_pre_news = bool(
        getattr(
            news_settings if news_settings is not None else engine_config,
            "cancel_pending_pre_news",
            getattr(engine_config, "cancel_pending_pre_news", True),
        )
    )
    forced_exit_pre_news = bool(
        getattr(
            news_settings if news_settings is not None else engine_config,
            "forced_exit_pre_news",
            getattr(engine_config, "forced_exit_pre_news", True),
        )
    )

    for i in range(strategy_module.WARMUP_BARS, len(frame)):
        ts_utc = timestamp_utc[i]
        ts_local = bar_open_local[i]
        session_date = session_dates[i]

        # --- SIGNAL GENERATION ---
        if not position and not pending_signal and i >= cooldown_until_index:
            if opened_total_by_date.get(session_date, 0) < engine_config.max_trades_per_day:
                # CORE_REMEDIATION: Fallback to legacy .signal() if .generate_signal() is missing
                if hasattr(strategy_module, "generate_signal"):
                    raw_signal = strategy_module.generate_signal(frame, i, params)
                else:
                    raw_signal = strategy_module.signal(frame, i, params)
                if raw_signal:
                    pending_signal = validate_signal_risk_contract(raw_signal, signal_price=close[i], engine_config=engine_config)
                    pending_signal["signal_index"] = i
                    pending_signal["signal_time"] = ts_utc
                    pending_signal["signal_price"] = close[i]

        # --- NEWS FORTRESS: PENDING SIGNAL PROTECTION ---
        if pending_signal is not None and cancel_pending_pre_news:
            if pending_kill_mask[i]:
                 # Si entramos en zona de bloqueo de noticias, matamos cualquier orden pendiente
                 pending_signal = None

        # --- ENTRY PROCESSING (T+1 from Signal) ---
        if pending_signal is not None and i == pending_signal["signal_index"] + 1:
            entry_fill_kind = "stop_entry" if pending_signal.get("entry_mode") == "stop" else "entry"
            entry_slippage_pips = estimate_slippage_pips(bar_open_local[i], range_atr[pending_signal["signal_index"]], engine_config, fill_kind=entry_fill_kind)
            entry_cost_regime = execution_regime_label(bar_open_local[i], range_atr[pending_signal["signal_index"]], engine_config, fill_kind=entry_fill_kind)
            if precision_enabled:
                entry_bid_price = float(bid_m15_open[i])
                entry_ask_price = float(ask_m15_open[i])
                entry_spread_pips = actual_spread_pips(pair, entry_bid_price, entry_ask_price)
            else:
                entry_spread_pips = estimate_spread_pips(pair, bar_open_local[i], range_atr[pending_signal["signal_index"]], engine_config, fill_kind="entry")
            if (
                fill_allowed[i]
                and not entry_news_block[i]
                and np.isfinite(atr14[pending_signal["signal_index"]])
                and atr14[pending_signal["signal_index"]] > 0
                and range_atr[pending_signal["signal_index"]] <= engine_config.shock_candle_atr_max
                and spread_guard_allows(entry_spread_pips, engine_config)
            ):
                if pending_signal.get("entry_mode") == "limit":
                    limit_target = float(pending_signal["limit_price"])
                    if precision_enabled:
                        if pending_signal["direction"] == "long":
                            can_fill_limit = float(ask_m15_low[i]) <= limit_target <= float(ask_m15_high[i])
                        else:
                            can_fill_limit = float(bid_m15_low[i]) <= limit_target <= float(bid_m15_high[i])
                    else:
                        can_fill_limit = float(low[i]) <= limit_target <= float(high[i])
                    
                    if can_fill_limit:
                        entry_bid_price = limit_target
                        entry_price = entry_execution_price(pair, pending_signal["direction"], entry_bid_price, entry_spread_pips, entry_slippage_pips)
                    else:
                        pending_signal = None
                        continue
                elif pending_signal.get("entry_mode") == "stop":
                    stop_entry_target = float(pending_signal["stop_entry_price"])
                    if precision_enabled:
                        bid_slice, ask_slice = precision_intrabar_slice(
                            precision_package,
                            bar_close_time=local_index[i],
                            bar_delta=bar_delta,
                        )
                        stop_fill = resolve_high_precision_stop_entry_fill(
                            pair,
                            pending_signal["direction"],
                            stop_entry_price=stop_entry_target,
                            bid_slice=bid_slice,
                            ask_slice=ask_slice,
                            slippage_pips=entry_slippage_pips,
                        )
                        if stop_fill is None:
                            pending_signal = None
                            continue
                        entry_bid_price, entry_ask_price, entry_spread_pips, entry_price = stop_fill
                    else:
                        stop_fill = resolve_stop_entry_fill(
                            pair,
                            pending_signal["direction"],
                            open_price=float(open_[i]),
                            high_price=float(high[i]),
                            low_price=float(low[i]),
                            stop_entry_price=stop_entry_target,
                            spread_pips=entry_spread_pips,
                            slippage_pips=entry_slippage_pips,
                        )
                        if stop_fill is None:
                            pending_signal = None
                            continue
                        entry_bid_price, entry_price = stop_fill
                else:
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
                if not spread_guard_allows(entry_spread_pips, engine_config):
                    pending_signal = None
                    continue
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
                            entry_side=pending_signal["direction"],
                            signal_time=pending_signal["signal_time"],
                            signal_price=pending_signal["signal_price"],
                            fill_time=ts_local.tz_convert("UTC"),
                            entry_time=ts_local.tz_convert("UTC"),
                            entry_price=entry_price,
                            sl=sl_trigger,
                            tp=tp_trigger,
                            units=units,
                            lots=lots,
                            risk_usd=risk_usd,
                            initial_risk_distance=stop_distance,
                            entry_bar_index=i,
                            max_hold_bars=pending_signal.get("max_hold_bars"),
                            break_even_at_r=pending_signal.get("break_even_at_r"),
                            trailing_atr=bool(pending_signal.get("trailing_atr", False)),
                            trail_mult=float(pending_signal.get("trail_mult", 0.0)),
                            entry_commission_usd=entry_commission_usd,
                            entry_spread_pips=entry_spread_pips,
                            entry_slippage_pips=entry_slippage_pips,
                            execution_mode_used=engine_config.execution_mode,
                            cost_profile_used=resolved_cost_profile(engine_config),
                            entry_cost_regime=entry_cost_regime,
                            intrabar_policy_used=intrabar_policy_used,
                            price_source_used=engine_config.price_source,
                            data_source_used=resolved_data_source,
                        )
                        opened_total_by_date[session_date] = opened_total_by_date.get(session_date, 0) + 1
                pending_signal = None

        if position:
            exit_reason = None
            exit_price_signal = None
            gap_reason = None
            
            # --- NEWS FORTRESS: ACTIVE POSITION PROTECTION ---
            if forced_exit_pre_news and force_flat_mask[i]:
                exit_reason = "news_fortress_kill"
                exit_price_signal = open_[i]
            
            if not exit_reason:
                if precision_enabled:
                    bid_slice, ask_slice = precision_intrabar_slice(
                        precision_package,
                        bar_close_time=local_index[i],
                        bar_delta=bar_delta,
                    )
                    common_index = bid_slice.index.intersection(ask_slice.index)
                    for p_ts in common_index:
                        p_bid_o = float(bid_slice.at[p_ts, "open"])
                        p_ask_o = float(ask_slice.at[p_ts, "open"])
                        p_bid_h = float(bid_slice.at[p_ts, "high"])
                        p_ask_h = float(ask_slice.at[p_ts, "high"])
                        p_bid_l = float(bid_slice.at[p_ts, "low"])
                        p_ask_l = float(ask_slice.at[p_ts, "low"])
                        
                        p_low_ref = p_bid_l if position.direction == "long" else p_ask_l
                        p_high_ref = p_ask_h if position.direction == "long" else p_bid_h
                        p_open_ref = p_bid_o if position.direction == "long" else p_ask_o
                        
                        kind, trigger, _, gap_reason = resolve_intrabar_exit(
                            direction=position.direction,
                            open_price=p_open_ref,
                            low_price=p_low_ref,
                            high_price=p_high_ref,
                            sl_trigger=position.sl,
                            tp_trigger=position.tp,
                            priority=engine_config.intrabar_exit_priority,
                            intrabar_policy=intrabar_policy_used,
                        )
                        if kind:
                            exit_reason = kind
                            exit_price_signal = trigger
                            break
                else:
                    kind, trigger, _, gap_reason = resolve_intrabar_exit(
                        direction=position.direction,
                        open_price=float(open_[i]),
                        low_price=float(low[i]),
                        high_price=float(high[i]),
                        sl_trigger=position.sl,
                        tp_trigger=position.tp,
                        priority=engine_config.intrabar_exit_priority,
                        intrabar_policy=intrabar_policy_used,
                    )
                    if kind:
                        exit_reason = kind
                        exit_price_signal = trigger

            if not exit_reason and force_close_mask[i]:
                exit_reason = "forced_session_close"
                exit_price_signal = close[i]
            
            if not exit_reason and position.max_hold_bars is not None:
                bars_held = i - position.entry_bar_index
                if bars_held >= int(position.max_hold_bars):
                    exit_reason = "time_exit"
                    exit_price_signal = close[i]

            if exit_reason:
                exit_slippage_pips = estimate_slippage_pips(bar_open_local[i], range_atr[i], engine_config, fill_kind=exit_reason)
                if precision_enabled:
                    exit_price = high_precision_exit_execution_price(pair, position.direction, exit_price_signal, exit_slippage_pips)
                else:
                    exit_price = exit_execution_price(pair, position.direction, exit_price_signal, position.entry_spread_pips, exit_slippage_pips)
                
                pnl_usd = directional_pnl_usd(position.direction, position.entry_price, exit_price, position.units, quote_to_usd(pair, exit_price))
                exit_commission_usd = (engine_config.commission_per_lot_roundturn_usd * position.lots) / 2.0
                pnl_usd -= exit_commission_usd
                # risk_usd is never reserved from cash at entry, so only realized pnl is applied here.
                cash += pnl_usd
                pnl_r = pnl_usd / position.risk_usd
                pip_size = float(PAIR_META[pair]["pip_size"])
                sl_pips = position.initial_risk_distance / pip_size if pip_size > 0 else np.nan
                commission_total_usd = position.entry_commission_usd + exit_commission_usd
                
                trades.append({
                    "pair": pair,
                    "strategy_name": position.strategy_name,
                    "direction": position.direction,
                    "signal_time": position.signal_time,
                    "signal_price": position.signal_price,
                    "entry_time": position.entry_time,
                    "entry_price": position.entry_price,
                    "exit_time": ts_local.tz_convert("UTC"),
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "pnl_usd": pnl_usd,
                    "pnl_r": pnl_r,
                    "lots": position.lots,
                    "session_date": session_date,
                    "telemetry_version": D5_TELEMETRY_VERSION,
                    "telemetry_behavior_neutral": True,
                    "net_r": pnl_r,
                    "gross_r": None,
                    "gross_r_available": False,
                    "gross_r_reason": "not_available_without_explicit_pre_cost_pnl_source",
                    "sl_pips": sl_pips,
                    "sl_pips_available": True,
                    "risk_pips": sl_pips,
                    "risk_distance_price": position.initial_risk_distance,
                    "initial_risk_distance": position.initial_risk_distance,
                    "risk_usd": position.risk_usd,
                    "stop_price": position.sl,
                    "initial_stop_price": position.sl,
                    "final_stop_price": position.sl,
                    "sl": position.sl,
                    "tp": position.tp,
                    "fill_time": position.fill_time,
                    "fill_price": position.entry_price,
                    "exit_signal_price": exit_price_signal,
                    "exit_fill_price": exit_price,
                    "gap_exit_flag": bool(gap_reason),
                    "gap_exit_type": gap_reason if gap_reason else "no_gap",
                    "entry_spread_pips": position.entry_spread_pips,
                    "entry_slippage_pips": position.entry_slippage_pips,
                    "exit_slippage_pips": exit_slippage_pips,
                    "slippage_applied": exit_slippage_pips,
                    "entry_commission_usd": position.entry_commission_usd,
                    "exit_commission_usd": exit_commission_usd,
                    "commission_total_usd": commission_total_usd,
                    "spread_cost_r": None,
                    "slippage_cost_r": None,
                    "commission_cost_r": None,
                    "cost_total_r": None,
                    "cost_breakdown_r_available": False,
                    "cost_breakdown_r_reason": "not_available_without_explicit_per_component_pnl_source",
                    "execution_mode_used": position.execution_mode_used,
                    "cost_profile_used": position.cost_profile_used,
                    "entry_cost_regime": position.entry_cost_regime,
                    "intrabar_policy_used": position.intrabar_policy_used,
                    "price_source_used": position.price_source_used,
                    "data_source_used": position.data_source_used,
                    "blocking_rule_used": news_details.iloc[i]["blocking_rule_used"],
                    "entry_rule_used": news_details.iloc[i]["entry_rule_used"],
                    "pending_rule_used": news_details.iloc[i]["pending_rule_used"],
                    "force_flat_rule_used": news_details.iloc[i]["force_flat_rule_used"],
                })
                position = None
                cooldown_until_index = i + 1

        equity_points.append({"timestamp": ts_utc, "equity": cash + (0 if not position else 0)}) # Simplificado

    # --- FINAL CLOSE ---
    if position:
        last_idx = len(frame) - 1
        exit_reason = "final_bar_close"
        exit_price_signal = open_[last_idx]
        exit_slippage_pips = estimate_slippage_pips(bar_open_local[last_idx], range_atr[last_idx], engine_config, fill_kind=exit_reason)
        if precision_enabled:
            exit_price = high_precision_exit_execution_price(pair, position.direction, exit_price_signal, exit_slippage_pips)
        else:
            exit_price = exit_execution_price(pair, position.direction, exit_price_signal, position.entry_spread_pips, exit_slippage_pips)
        
        pnl_usd = directional_pnl_usd(position.direction, position.entry_price, exit_price, position.units, quote_to_usd(pair, exit_price))
        exit_commission_usd = (engine_config.commission_per_lot_roundturn_usd * position.lots) / 2.0
        pnl_usd -= exit_commission_usd
        # risk_usd is never reserved from cash at entry, so only realized pnl is applied here.
        cash += pnl_usd
        pnl_r = pnl_usd / position.risk_usd
        pip_size = float(PAIR_META[pair]["pip_size"])
        sl_pips = position.initial_risk_distance / pip_size if pip_size > 0 else np.nan
        commission_total_usd = position.entry_commission_usd + exit_commission_usd
        
        trades.append({
            "pair": pair,
            "strategy_name": position.strategy_name,
            "direction": position.direction,
            "signal_time": position.signal_time,
            "signal_price": position.signal_price,
            "entry_time": position.entry_time,
            "entry_price": position.entry_price,
            "exit_time": timestamp_utc[last_idx],
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "pnl_usd": pnl_usd,
            "pnl_r": pnl_r,
            "lots": position.lots,
            "session_date": session_dates[last_idx],
            "telemetry_version": D5_TELEMETRY_VERSION,
            "telemetry_behavior_neutral": True,
            "net_r": pnl_r,
            "gross_r": None,
            "gross_r_available": False,
            "gross_r_reason": "not_available_without_explicit_pre_cost_pnl_source",
            "sl_pips": sl_pips,
            "sl_pips_available": True,
            "risk_pips": sl_pips,
            "risk_distance_price": position.initial_risk_distance,
            "initial_risk_distance": position.initial_risk_distance,
            "risk_usd": position.risk_usd,
            "stop_price": position.sl,
            "initial_stop_price": position.sl,
            "final_stop_price": position.sl,
            "sl": position.sl,
            "tp": position.tp,
            "fill_time": position.fill_time,
            "fill_price": position.entry_price,
            "exit_signal_price": exit_price_signal,
            "exit_fill_price": exit_price,
            "gap_exit_flag": False,
            "gap_exit_type": "no_gap",
            "entry_spread_pips": position.entry_spread_pips,
            "entry_slippage_pips": position.entry_slippage_pips,
            "exit_slippage_pips": exit_slippage_pips,
            "slippage_applied": exit_slippage_pips,
            "entry_commission_usd": position.entry_commission_usd,
            "exit_commission_usd": exit_commission_usd,
            "commission_total_usd": commission_total_usd,
            "spread_cost_r": None,
            "slippage_cost_r": None,
            "commission_cost_r": None,
            "cost_total_r": None,
            "cost_breakdown_r_available": False,
            "cost_breakdown_r_reason": "not_available_without_explicit_pre_cost_pnl_source",
            "execution_mode_used": position.execution_mode_used,
            "cost_profile_used": position.cost_profile_used,
            "entry_cost_regime": position.entry_cost_regime,
            "intrabar_policy_used": position.intrabar_policy_used,
            "price_source_used": position.price_source_used,
            "data_source_used": position.data_source_used,
            "blocking_rule_used": news_details.iloc[last_idx]["blocking_rule_used"],
            "entry_rule_used": news_details.iloc[last_idx]["entry_rule_used"],
            "pending_rule_used": news_details.iloc[last_idx]["pending_rule_used"],
            "force_flat_rule_used": news_details.iloc[last_idx]["force_flat_rule_used"],
        })

    return BacktestResult(
        strategy_name=strategy_module.NAME,
        trades=pd.DataFrame(trades),
        equity_curve=pd.DataFrame(equity_points),
        params=params,
        news_filter_used=news_filter_used,
    )
