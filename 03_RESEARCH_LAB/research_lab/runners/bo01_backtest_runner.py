# -*- coding: utf-8 -*-
"""
BO01 BACKTEST RUNNER SYNTHETIC V1
This module provides a pure, dependency-free structural backtesting engine
for candidate strategy BO01 (London Breakout) with strict quant controls.
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd

# Required constants
RUNNER_ID = "BO01_BACKTEST_RUNNER_SYNTHETIC_V1"
STRATEGY_ID = "BO01"
ENTRY_POLICY = "ENTRY_NEXT_CANDLE_OPEN"
SAME_BAR_POLICY = "STOP_FIRST"
MAX_TRADES_PER_DAY = 1
MAX_ACTIVE_POSITIONS = 1

def validate_backtest_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """
    Validates the structure of the input backtesting DataFrame.
    Returns a status dictionary with validation results.
    """
    res = {
        "ok": True,
        "errors": [],
        "warnings": [],
        "row_count": 0,
        "min_timestamp": None,
        "max_timestamp": None
    }
    
    if frame is None or not isinstance(frame, pd.DataFrame):
        res["ok"] = False
        res["errors"].append("Input frame must be a non-None pandas DataFrame.")
        return res

    res["row_count"] = len(frame)
    if len(frame) == 0:
        res["ok"] = False
        res["errors"].append("DataFrame is empty.")
        return res

    # Check columns
    required_cols = ("open", "high", "low", "close")
    for col in required_cols:
        if col not in frame.columns:
            res["ok"] = False
            res["errors"].append(f"Missing required column: '{col}'.")

    # Temporal index validation
    if not isinstance(frame.index, pd.DatetimeIndex):
        res["ok"] = False
        res["errors"].append("DataFrame index must be a pandas DatetimeIndex.")
    else:
        # Check order
        if not frame.index.is_monotonic_increasing:
            res["ok"] = False
            res["errors"].append("DataFrame index must be monotonically increasing.")
        
        # Check timezone
        if frame.index.tz is None:
            res["ok"] = False
            res["errors"].append("DataFrame index must have an active timezone (UTC).")
        
        # Range bounds
        if len(frame) > 0:
            min_ts = frame.index[0]
            max_ts = frame.index[-1]
            res["min_timestamp"] = str(min_ts)
            res["max_timestamp"] = str(max_ts)

            # Block future dates (2025/2026)
            for ts in (min_ts, max_ts):
                if ts.year in (2025, 2026):
                    res["ok"] = False
                    res["errors"].append(f"Unauthorized date detected: {ts.year}. Access to 2025/2026 is strictly blocked.")

    # Check for NaNs in critical columns
    if res["ok"]:
        for col in required_cols:
            if frame[col].isna().any():
                res["ok"] = False
                res["errors"].append(f"NaN value detected in critical column: '{col}'.")

    # Check for partition/split attributes
    for attr in ("partition", "split", "dataset_split", "data_split"):
        if attr in frame.columns:
            vals = frame[attr].dropna().unique()
            for v in vals:
                if str(v).lower() in ("validation", "holdout"):
                    res["ok"] = False
                    res["errors"].append(f"Unauthorized data partition '{v}' detected. Access strictly blocked.")

    return res


def validate_signal_contract(signal: dict[str, Any]) -> dict[str, Any]:
    """
    Validates a strategy signal against the BO01 strategy contract.
    """
    res = {"ok": True, "errors": []}
    if signal is None or not isinstance(signal, dict):
        raise TypeError("Signal must be a non-None dictionary.")

    required_keys = ("signal", "direction", "stop_price", "target_rr")
    for key in required_keys:
        if key not in signal:
            res["ok"] = False
            res["errors"].append(f"Missing required contract key: '{key}'.")

    if not res["ok"]:
        raise ValueError(f"Signal contract validation failed: {res['errors']}")

    # Values validation
    sig_val = signal.get("signal")
    if sig_val not in (1, -1):
        res["ok"] = False
        res["errors"].append(f"Signal value must be 1 or -1, got {sig_val}.")

    direction = signal.get("direction")
    if direction not in ("long", "short"):
        res["ok"] = False
        res["errors"].append(f"Direction must be 'long' or 'short', got '{direction}'.")

    stop_price = signal.get("stop_price")
    try:
        float(stop_price)
    except (ValueError, TypeError):
        res["ok"] = False
        res["errors"].append(f"Stop price must be a valid float, got {stop_price}.")

    target_rr = signal.get("target_rr")
    try:
        float(target_rr)
    except (ValueError, TypeError):
        res["ok"] = False
        res["errors"].append(f"Target risk-reward must be a valid float, got {target_rr}.")

    if not res["ok"]:
        raise ValueError(f"Signal contract validation failed: {res['errors']}")

    return res


def compute_cost_r(
    entry_price: float,
    stop_price: float,
    cost_profile: dict[str, Any],
    pip_size: float = 0.0001
) -> float:
    """
    Computes total friction costs (spread, slippage, commission) in terms of R-multiples.
    1R is defined as the distance between the entry price and the stop-loss price.
    For EURUSD standard sizing, 1 Lot has a pip value of $10.
    """
    stop_dist = abs(entry_price - stop_price)
    if stop_dist <= 1e-8:
        raise ValueError("Entry price and Stop-loss price cannot be equal.")

    stop_dist_pips = stop_dist / pip_size

    # Spreads and Slippage (already in pips)
    spread_pips = float(cost_profile.get("spread", 0.0))
    slippage_pips = float(cost_profile.get("slippage", 0.0))
    spread_slippage_r = (spread_pips + slippage_pips) / stop_dist_pips

    # Commissions (defined in USD per standard lot round-turn)
    # Commission R conversion: USD_commission / (Stop_Distance_Pips * Pip_Value)
    # Assuming standard Lot sizing where 1 pip = $10.
    commission_usd = float(cost_profile.get("commission", 0.0))
    commission_r = commission_usd / (stop_dist_pips * 10.0) if stop_dist_pips > 0 else 0.0

    return spread_slippage_r + commission_r


def resolve_trade_exit(
    frame: pd.DataFrame,
    entry_idx: int,
    side: str,
    entry_price: float,
    stop_price: float,
    target_price: float,
    timeout_bars: int | None = None
) -> dict[str, Any]:
    """
    Simulates chronological exits row-by-row starting from the entry index t+1.
    If stop and target are both hit in the same candle, STOP_FIRST resolution applies.
    """
    n_rows = len(frame)
    
    for j in range(entry_idx, n_rows):
        high = float(frame["high"].iat[j])
        low = float(frame["low"].iat[j])
        close = float(frame["close"].iat[j])

        # Timeout condition check
        is_timeout = False
        if timeout_bars is not None and (j - entry_idx) >= timeout_bars:
            is_timeout = True

        if side == "long":
            stop_hit = (low <= stop_price)
            target_hit = (high >= target_price)

            if stop_hit and target_hit:
                # Same-bar resolution: STOP-FIRST conservative policy
                return {
                    "exit_idx": j,
                    "exit_type": "same_bar_stop_first",
                    "gross_r": -1.0,
                    "exit_price": stop_price
                }
            elif stop_hit:
                return {
                    "exit_idx": j,
                    "exit_type": "stop",
                    "gross_r": -1.0,
                    "exit_price": stop_price
                }
            elif target_hit:
                target_rr = (target_price - entry_price) / (entry_price - stop_price)
                return {
                    "exit_idx": j,
                    "exit_type": "target",
                    "gross_r": float(target_rr),
                    "exit_price": target_price
                }
            elif is_timeout:
                timeout_r = (close - entry_price) / (entry_price - stop_price)
                return {
                    "exit_idx": j,
                    "exit_type": "timeout",
                    "gross_r": float(timeout_r),
                    "exit_price": close
                }

        elif side == "short":
            stop_hit = (high >= stop_price)
            target_hit = (low <= target_price)

            if stop_hit and target_hit:
                # Same-bar resolution: STOP-FIRST conservative policy
                return {
                    "exit_idx": j,
                    "exit_type": "same_bar_stop_first",
                    "gross_r": -1.0,
                    "exit_price": stop_price
                }
            elif stop_hit:
                return {
                    "exit_idx": j,
                    "exit_type": "stop",
                    "gross_r": -1.0,
                    "exit_price": stop_price
                }
            elif target_hit:
                target_rr = (entry_price - target_price) / (stop_price - entry_price)
                return {
                    "exit_idx": j,
                    "exit_type": "target",
                    "gross_r": float(target_rr),
                    "exit_price": target_price
                }
            elif is_timeout:
                timeout_r = (entry_price - close) / (stop_price - entry_price)
                return {
                    "exit_idx": j,
                    "exit_type": "timeout",
                    "gross_r": float(timeout_r),
                    "exit_price": close
                }

    # Open position at end of frame
    last_idx = n_rows - 1
    last_close = float(frame["close"].iat[last_idx])
    if side == "long":
        unrealized_r = (last_close - entry_price) / (entry_price - stop_price)
    else:
        unrealized_r = (entry_price - last_close) / (stop_price - entry_price)

    return {
        "exit_idx": last_idx,
        "exit_type": "open_at_end",
        "gross_r": float(unrealized_r),
        "exit_price": last_close
    }


def run_bo01_backtest_on_frame(
    strategy_cls: Any,
    frame: pd.DataFrame,
    params: dict[str, Any] | None = None,
    cost_profile: dict[str, Any] | None = None,
    max_trades_per_day: int = 1
) -> dict[str, Any]:
    """
    Runs the backtest simulation over the input Datetime-indexed DataFrame.
    """
    validation_res = validate_backtest_frame(frame)
    if not validation_res["ok"]:
        raise ValueError(f"DataFrame validation failed: {validation_res['errors']}")

    if params is None:
        params = {}
    if cost_profile is None:
        cost_profile = {"spread": 0.0, "slippage": 0.0, "commission": 0.0}

    n_rows = len(frame)
    trades = []
    
    # Tracking variables
    active_trade = None
    trades_by_day = {}
    skipped_same_day = 0
    skipped_active_pos = 0
    invalid_sig_count = 0
    exception_count = 0
    same_bar_stop_first_count = 0
    stop_count = 0
    target_count = 0
    timeout_count = 0
    open_end_count = 0

    idx = 0
    while idx < n_rows:
        current_time = frame.index[idx]
        current_date = current_time.date()

        # Update active position status in params for the strategy
        session_params = {
            **params,
            "daily_trade_count": trades_by_day.get(current_date, 0),
            "has_active_position": active_trade is not None
        }

        # Check for exits first
        if active_trade is not None:
            # We are holding a position; skip processing strategy signals until closed
            # In row-by-row backtesting, we skip past the exit index of the active trade
            exit_info = active_trade["exit_info"]
            if idx == exit_info["exit_idx"]:
                # The position closed on this candle
                active_trade = None
            idx += 1
            continue

        # Evaluate strategy signals at index t (closed bar)
        try:
            sig = strategy_cls.signal(frame, idx, session_params)
        except Exception:
            exception_count += 1
            idx += 1
            continue

        if sig is not None:
            # Validate signal contract
            try:
                validate_signal_contract(sig)
            except ValueError:
                invalid_sig_count += 1
                idx += 1
                continue

            # Day limit check
            day_trades = trades_by_day.get(current_date, 0)
            if day_trades >= max_trades_per_day:
                skipped_same_day += 1
                idx += 1
                continue

            # Check next candle t+1 exists for entry open fill
            entry_idx = idx + 1
            if entry_idx >= n_rows:
                # No next candle available; abort entry signal
                idx += 1
                continue

            # Execute entry at Open of t+1
            entry_price = float(frame["open"].iat[entry_idx])
            stop_price = float(sig["stop_price"])
            direction = sig["direction"]

            # Compute target price based on target_rr
            target_rr = float(sig["target_rr"])
            stop_distance = abs(entry_price - stop_price)
            if direction == "long":
                target_price = entry_price + (stop_distance * target_rr)
            else:
                target_price = entry_price - (stop_distance * target_rr)

            # Resolve exit chronologically starting from t+1
            exit_res = resolve_trade_exit(
                frame=frame,
                entry_idx=entry_idx,
                side=direction,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price
            )

            # Calculate costs in R
            pip_size = params.get("pip_size", 0.0001)
            cost_r = compute_cost_r(
                entry_price=entry_price,
                stop_price=stop_price,
                cost_profile=cost_profile,
                pip_size=pip_size
            )

            gross_r = exit_res["gross_r"]
            net_r = gross_r - cost_r

            trade_record = {
                "entry_idx": entry_idx,
                "entry_time": str(frame.index[entry_idx]),
                "direction": direction,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "exit_idx": exit_res["exit_idx"],
                "exit_time": str(frame.index[exit_res["exit_idx"]]),
                "exit_type": exit_res["exit_type"],
                "exit_price": exit_res["exit_price"],
                "gross_r": gross_r,
                "cost_r": cost_r,
                "net_r": net_r
            }

            trades.append(trade_record)
            trades_by_day[current_date] = day_trades + 1

            # Exit type counts
            etype = exit_res["exit_type"]
            if etype == "same_bar_stop_first":
                same_bar_stop_first_count += 1
                stop_count += 1
            elif etype == "stop":
                stop_count += 1
            elif etype == "target":
                target_count += 1
            elif etype == "timeout":
                timeout_count += 1
            elif etype == "open_at_end":
                open_end_count += 1

            # Set as active position so that subsequent candles are skipped until exit
            active_trade = {
                "info": trade_record,
                "exit_info": exit_res
            }

        idx += 1

    # Summarize metrics
    trade_count = len(trades)
    gross_R = float(sum(t["gross_r"] for t in trades))
    net_R = float(sum(t["net_r"] for t in trades))
    
    avg_R = gross_R / trade_count if trade_count > 0 else 0.0
    median_R = float(np.median([t["gross_r"] for t in trades])) if trade_count > 0 else 0.0

    wins = sum(1 for t in trades if t["gross_r"] > 0)
    winrate = (wins / trade_count) if trade_count > 0 else 0.0

    loss_r_sum = sum(abs(t["gross_r"]) for t in trades if t["gross_r"] < 0)
    win_r_sum = sum(t["gross_r"] for t in trades if t["gross_r"] > 0)
    profit_factor_R = (win_r_sum / loss_r_sum) if loss_r_sum > 0 else (float("inf") if win_r_sum > 0 else 0.0)

    # Max Drawdown in terms of net cumulative R
    cum_R = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        cum_R += t["net_r"]
        if cum_R > peak:
            peak = cum_R
        dd = peak - cum_R
        if dd > max_dd:
            max_dd = dd

    # Winning / Losing streaks
    max_losing = 0
    max_winning = 0
    curr_losing = 0
    curr_winning = 0
    for t in trades:
        if t["net_r"] > 0:
            curr_winning += 1
            curr_losing = 0
            if curr_winning > max_winning:
                max_winning = curr_winning
        else:
            curr_losing += 1
            curr_winning = 0
            if curr_losing > max_losing:
                max_losing = curr_losing

    expectancy_R = net_R / trade_count if trade_count > 0 else 0.0

    return {
        "runner_id": RUNNER_ID,
        "strategy_id": STRATEGY_ID,
        "entry_policy": ENTRY_POLICY,
        "same_bar_policy": SAME_BAR_POLICY,
        "trade_count": trade_count,
        "gross_R": gross_R,
        "net_R": net_R,
        "average_R": avg_R,
        "median_R": median_R,
        "winrate": winrate,
        "profit_factor_R": profit_factor_R,
        "max_drawdown_R": max_dd,
        "max_losing_streak": max_losing,
        "max_winning_streak": max_winning,
        "expectancy_R": expectancy_R,
        "stop_count": stop_count,
        "target_count": target_count,
        "timeout_count": timeout_count,
        "same_bar_stop_first_count": same_bar_stop_first_count,
        "open_end_count": open_end_count,
        "skipped_signals_same_day": skipped_same_day,
        "skipped_signals_active_position": skipped_active_pos,
        "invalid_signal_count": invalid_sig_count,
        "exception_count": exception_count,
        "trades": trades
    }
