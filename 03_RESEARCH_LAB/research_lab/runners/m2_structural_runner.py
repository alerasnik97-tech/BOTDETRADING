from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

RUNNER_ID = "M2_STRUCTURAL_RUNNER_BO01_MR02_V1"
ALLOWED_STRATEGY_IDS = {"BO01", "MR02"}

# Strictly negative declarations for compliance checks
FORBIDDEN_PERFORMANCE_TERMS = (
    "pnl",
    "profit",
    "profit_factor",
    "pf",
    "winrate",
    "drawdown",
    "sharpe",
    "sortino",
    "expectancy",
    "equity_curve",
    "r_multiple",
    "average_winner",
    "average_loser",
    "optimization",
    "sweep",
    "grid_search",
    "walk_forward",
    "backtest",
    "train",
    "trades.csv",
    "equity_curve.csv",
    "pnl.csv",
    "performance_report"
)


def validate_frame_for_m2(frame: pd.DataFrame) -> dict[str, Any]:
    """Validates the input DataFrame to ensure compliance with train-only structural evaluation bounds.
    
    This function strictly does NOT compute returns, return variances, or profitability metrics.
    """
    if frame is None or frame.empty:
        raise ValueError("Frame is None or empty")
    
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Frame index must be a DatetimeIndex")
        
    if frame.index.tz is None:
        raise ValueError("DatetimeIndex must be timezone-aware (UTC expected)")
        
    # Check forbidden years
    years = frame.index.year
    if any(y == 2025 for y in years) or any(y == 2026 for y in years):
        raise ValueError("Data contains forbidden dates (2025 or 2026)")
        
    # Check validation/holdout flags
    for col in ("partition", "split", "dataset_split", "data_split"):
        if col in frame.columns:
            vals = frame[col].astype(str).str.lower().values
            if any("val" in v for v in vals) or any("hold" in v for v in vals):
                raise ValueError("Data contains validation or holdout partition records")
                
    # Check columns
    required_cols = {"open", "high", "low", "close"}
    missing = required_cols - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
        
    # Verify approximate M5 cadence (difference between consecutive timestamps is ~5 minutes)
    if len(frame) > 1:
        diffs = pd.Series(frame.index).diff().dropna()
        median_diff = diffs.median()
        if median_diff < pd.Timedelta(minutes=1) or median_diff > pd.Timedelta(minutes=10):
            raise ValueError(f"Cadence is not M5 approximate. Median delta: {median_diff}")
            
    # Check critical NaNs in OHLC
    for col in required_cols:
        if frame[col].isna().any():
            raise ValueError(f"OHLC column '{col}' contains NaN values")
            
    return {
        "valid": True,
        "row_count": len(frame),
        "min_timestamp": str(frame.index.min()),
        "max_timestamp": str(frame.index.max()),
    }


def run_structural_counts(
    strategy_cls: Any,
    frame: pd.DataFrame,
    params: dict[str, Any] | None = None,
    strategy_id: str | None = None
) -> dict[str, Any]:
    """Runs a pure structural evaluation of the strategy over the given DataFrame.
    
    This function counts candle processing, signal attempts, valid signal structure,
    exceptions, temporal distribution, and safety violations. It strictly does not
    simulate positions, calculate profitability, or simulate execution paths.
    """
    str_id = getattr(strategy_cls, "ID", strategy_id)
    if str_id not in ALLOWED_STRATEGY_IDS:
        raise ValueError(f"Strategy {str_id} not allowed. Supported: {ALLOWED_STRATEGY_IDS}")
        
    if params is None:
        params = strategy_cls.default_params()
        
    counts = {
        "row_count": len(frame),
        "signal_call_count": 0,
        "valid_signal_count": 0,
        "none_count": 0,
        "exception_count": 0,
        "days_with_signal": 0,
        "max_signals_per_day": 0,
        "signals_by_hour": {h: 0 for h in range(24)},
        "signals_by_month": {m: 0 for m in range(1, 13)},
        "missing_column_failure_count": 0,
        "fail_closed_count": 0,
        "contract_valid_count": 0,
        "timestamp_anomaly_count": 0,
        "cadence_anomaly_count": 0,
        "forbidden_date_count": 0,
        "validation_holdout_access_count": 0,
    }
    
    signals_per_day: dict[Any, int] = {}
    
    for i in range(len(frame)):
        ts = frame.index[i]
        
        # Safe inline check for forbidden dates/partitions
        if ts.year == 2025 or ts.year == 2026:
            counts["forbidden_date_count"] += 1
            counts["fail_closed_count"] += 1
            continue
            
        partition_failed = False
        for col in ("partition", "split", "dataset_split", "data_split"):
            if col in frame.columns:
                val = str(frame[col].iat[i]).lower()
                if "val" in val or "hold" in val:
                    counts["validation_holdout_access_count"] += 1
                    counts["fail_closed_count"] += 1
                    partition_failed = True
                    break
        if partition_failed:
            continue
            
        # Verify OHLC finite values
        try:
            current_vals = [
                float(frame["open"].iat[i]),
                float(frame["high"].iat[i]),
                float(frame["low"].iat[i]),
                float(frame["close"].iat[i]),
            ]
            if not all(np.isfinite(val) for val in current_vals):
                counts["missing_column_failure_count"] += 1
                counts["fail_closed_count"] += 1
                continue
        except Exception:
            counts["missing_column_failure_count"] += 1
            counts["fail_closed_count"] += 1
            continue
            
        # Safe call to signal function
        counts["signal_call_count"] += 1
        try:
            sig = strategy_cls.signal(frame, i, params)
        except Exception:
            counts["exception_count"] += 1
            counts["fail_closed_count"] += 1
            continue
            
        if sig is None:
            counts["none_count"] += 1
            continue
            
        # Verify signal structure meets minimum contract
        try:
            if not isinstance(sig, dict):
                raise ValueError("Signal result is not a dictionary")
            if "signal" not in sig or sig["signal"] not in (1, -1):
                raise ValueError("Signal result 'signal' must be 1 or -1")
            if "direction" not in sig or sig["direction"] not in ("long", "short"):
                raise ValueError("Signal result 'direction' must be 'long' or 'short'")
            
            # The signal structure is contract-valid
            counts["valid_signal_count"] += 1
            counts["contract_valid_count"] += 1
            
            # Record temporal statistics
            day = ts.date()
            signals_per_day[day] = signals_per_day.get(day, 0) + 1
            counts["signals_by_hour"][ts.hour] += 1
            counts["signals_by_month"][ts.month] += 1
            
        except Exception:
            counts["exception_count"] += 1
            counts["fail_closed_count"] += 1
            
    if signals_per_day:
        counts["days_with_signal"] = len(signals_per_day)
        counts["max_signals_per_day"] = max(signals_per_day.values())
        
    return counts


def run_m2_structural_evaluation(
    strategies: list[Any],
    frame: pd.DataFrame,
    window_start: str,
    window_end: str
) -> dict[str, Any]:
    """Filters the input DataFrame by the declared window, validates it, and runs the structural counts.
    
    This function strictly operates in-memory and does not read/write to the filesystem.
    """
    start_ts = pd.Timestamp(window_start)
    end_ts = pd.Timestamp(window_end)
    
    if frame is not None and isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is not None:
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize(frame.index.tz)
        else:
            start_ts = start_ts.tz_convert(frame.index.tz)
            
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize(frame.index.tz)
        else:
            end_ts = end_ts.tz_convert(frame.index.tz)
            
    # Safe slice of frame
    sliced = frame.loc[start_ts:end_ts].copy()
    
    # Run validation
    diag = validate_frame_for_m2(sliced)
    
    summary = {
        "status": "COMPLETED",
        "runner_id": RUNNER_ID,
        "window_start": str(start_ts),
        "window_end": str(end_ts),
        "sliced_rows": len(sliced),
        "frame_validation": diag,
        "results": {},
    }
    
    for strategy in strategies:
        str_id = getattr(strategy, "ID", None)
        counts = run_structural_counts(strategy, sliced)
        summary["results"][str_id] = counts
        
    return summary
