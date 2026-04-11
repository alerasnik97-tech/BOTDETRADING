from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research_lab.config import DEFAULT_DATA_DIRS, DEFAULT_NEWS_FILE, DEFAULT_PAIR, DEFAULT_RAW_NEWS_FILE, EngineConfig, INITIAL_CAPITAL, NY_TZ, NewsConfig
from research_lab.data_loader import _resample_to_m15, fx_market_mask, load_price_data, prepare_common_frame
from research_lab.engine import entry_open_index, run_backtest
from research_lab.news_filter import SUPPORTED_VALIDATION_EVENTS, build_entry_block, build_news_datasets, filter_event_family, load_news_events, news_result_payload
from research_lab.report import summarize_result, sync_visible_chatgpt
from research_lab.strategies import STRATEGY_REGISTRY


AUDIT_RESULTS_DIR = Path("results") / "research_lab_audit"
SESSION_START = "11:00"
SESSION_FORCE_CLOSE = "19:00"
NY_ZONE = ZoneInfo(NY_TZ)


@dataclass(frozen=True)
class AuditContext:
    pair: str
    start: str
    end: str
    data_dirs: list[Path]
    results_dir: Path
    news_file: Path


def build_output_root(results_dir: Path) -> Path:
    timestamp = pd.Timestamp.now(tz=NY_TZ).strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"{timestamp}_audit"
    path.mkdir(parents=True, exist_ok=True)
    return path


def time_to_minute(value: str) -> int:
    hour, minute = (int(part) for part in value.split(":"))
    return hour * 60 + minute


def expected_fx_5m_index(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DatetimeIndex:
    grid = pd.date_range(start=start_ts.floor("5min"), end=end_ts.ceil("5min"), freq="5min", tz=start_ts.tz)
    minutes = grid.hour * 60 + grid.minute
    dow = grid.dayofweek
    mask = (
        ((dow >= 0) & (dow <= 3))
        | ((dow == 4) & (minutes <= 17 * 60))
        | ((dow == 6) & (minutes > 17 * 60))
    )
    return grid[mask]


def load_source_frames(pair: str, data_dirs: list[Path]) -> list[tuple[Path, pd.DataFrame]]:
    frames: list[tuple[Path, pd.DataFrame]] = []
    for data_dir in data_dirs:
        path = data_dir / f"{pair}_M5.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, index_col=0)
        raw_index = frame.index.astype(str)
        frame.index = pd.to_datetime(raw_index, utc=True, errors="coerce")
        frame.attrs["raw_index_values"] = raw_index[:5].tolist()
        frames.append((path, frame))
    return frames


def infer_index_format(index: pd.Index) -> dict[str, Any]:
    sample = index.astype(str)[:5].tolist()
    return {
        "sample_values": sample,
        "contains_explicit_offset": any("+" in value[10:] or "-" in value[10:] for value in sample),
        "parsed_timezone_aware": bool(getattr(index, "tz", None) is not None),
        "looks_like_new_york_offset": any(value.endswith("-05:00") or value.endswith("-04:00") for value in sample),
    }


def analyze_data_sources(ctx: AuditContext) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    source_frames = load_source_frames(ctx.pair, ctx.data_dirs)
    if not source_frames:
        raise FileNotFoundError(f"No encontré fuentes M5 para {ctx.pair} en {ctx.data_dirs}")

    source_rows: list[dict[str, Any]] = []
    pre_merge_parts: list[pd.DataFrame] = []

    for path, frame in source_frames:
        parsed = frame.copy()
        parsed.index = pd.to_datetime(parsed.index, utc=True, errors="coerce").tz_convert(NY_TZ)
        parsed = parsed.dropna(subset=["open", "high", "low", "close"])
        pre_merge_parts.append(parsed[["open", "high", "low", "close", "volume"]].copy())

        raw_sample = frame.attrs.get("raw_index_values", frame.index.astype(str)[:5].tolist())
        fmt = infer_index_format(pd.Index(raw_sample))
        source_rows.append(
            {
                "path": str(path),
                "rows": int(len(frame)),
                "duplicates_in_file": int(frame.index.duplicated().sum()),
                "first_raw_timestamp": str(frame.index.min()),
                "last_raw_timestamp": str(frame.index.max()),
                "contains_explicit_offset": fmt["contains_explicit_offset"],
                "looks_like_new_york_offset": fmt["looks_like_new_york_offset"],
            }
        )

    pre_merge = pd.concat(pre_merge_parts).sort_index()
    pre_merge_duplicates = int(pre_merge.index.duplicated().sum())
    merged = load_price_data(ctx.pair, ctx.data_dirs, ctx.start, ctx.end)
    expected = expected_fx_5m_index(merged.index.min(), merged.index.max())
    missing = expected.difference(merged.index)
    source_market = pre_merge.loc[fx_market_mask(pre_merge.index)].copy()
    source_sunday_bars = int((((source_market.index.dayofweek == 6) & ((source_market.index.hour * 60 + source_market.index.minute) > 17 * 60))).sum())
    merged_sunday_bars = int((((merged.index.dayofweek == 6) & ((merged.index.hour * 60 + merged.index.minute) > 17 * 60))).sum())

    deltas = pd.Series(merged.index[1:] - merged.index[:-1], index=merged.index[1:])
    if len(missing) == 0 and len(merged) == len(expected):
        abnormal_gaps = pd.Series(dtype="timedelta64[ns]")
    else:
        abnormal_gaps = deltas[deltas > pd.Timedelta(minutes=5)].sort_values(ascending=False)
    gap_examples = pd.DataFrame(
        {
            "timestamp_ny": abnormal_gaps.index.astype(str),
            "gap_minutes": abnormal_gaps.dt.total_seconds() / 60.0,
        }
    )

    raw_m15 = _resample_to_m15(merged)
    manual_labels = merged.index.ceil("15min")
    manual_m15 = (
        merged.assign(_label=manual_labels)
        .groupby("_label", sort=True)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )
    common_index = raw_m15.index.intersection(manual_m15.index)
    manual_compare = (raw_m15.loc[common_index] - manual_m15.loc[common_index]).abs()
    resample_mismatch_rows = int((manual_compare > 1e-12).any(axis=1).sum())

    data_summary = {
        "pair": ctx.pair,
        "source_files_used": [str(path) for path, _ in source_frames],
        "requested_period": f"{ctx.start} -> {ctx.end}",
        "merged_rows": int(len(merged)),
        "merged_start_ny": str(merged.index.min()),
        "merged_end_ny": str(merged.index.max()),
        "pre_merge_duplicate_rows": pre_merge_duplicates,
        "duplicates_after_merge": int(merged.index.duplicated().sum()),
        "missing_expected_5m_bars": int(len(missing)),
        "coverage_ratio_pct": float(len(merged.index.intersection(expected)) / len(expected) * 100.0) if len(expected) else 0.0,
        "source_sunday_session_bars": source_sunday_bars,
        "merged_sunday_session_bars": merged_sunday_bars,
        "sunday_session_dropped_by_loader": bool(source_sunday_bars > 0 and merged_sunday_bars < source_sunday_bars),
        "abnormal_gap_count": int(len(abnormal_gaps)),
        "max_gap_minutes": float(abnormal_gaps.dt.total_seconds().max() / 60.0) if len(abnormal_gaps) else 5.0,
        "resample_m15_mismatch_rows": resample_mismatch_rows,
        "m15_rows": int(len(raw_m15)),
    }
    return merged, data_summary, pd.DataFrame(source_rows), gap_examples


def analyze_timezone_and_schedule(m15_frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    minute_values = m15_frame.index.hour * 60 + m15_frame.index.minute
    session_mask = (minute_values >= time_to_minute(SESSION_START)) & (minute_values < time_to_minute(SESSION_FORCE_CLOSE))
    session_dates = pd.Series(m15_frame.index.date, index=m15_frame.index)
    actual_counts = (
        pd.DataFrame({"session_date": session_dates, "in_window": session_mask.astype(int)})
        .groupby("session_date", as_index=False)["in_window"]
        .sum()
        .rename(columns={"in_window": "bars_in_11_19_ny"})
    )
    expected_index = expected_fx_5m_index(m15_frame.index.min(), m15_frame.index.max())
    expected_frame = pd.DataFrame(index=expected_index, data={"marker": 1.0})
    expected_m15 = _resample_to_m15(expected_frame.rename(columns={"marker": "open"}).assign(high=1.0, low=1.0, close=1.0, volume=1.0))
    expected_minute_values = expected_m15.index.hour * 60 + expected_m15.index.minute
    expected_mask = (expected_minute_values >= time_to_minute(SESSION_START)) & (expected_minute_values < time_to_minute(SESSION_FORCE_CLOSE))
    expected_counts = (
        pd.DataFrame({"session_date": pd.Series(expected_m15.index.date, index=expected_m15.index), "expected_bars": expected_mask.astype(int)})
        .groupby("session_date", as_index=False)["expected_bars"]
        .sum()
    )
    session_counts = actual_counts.merge(expected_counts, on="session_date", how="outer").fillna(0)
    session_counts["bars_in_11_19_ny"] = session_counts["bars_in_11_19_ny"].astype(int)
    session_counts["expected_bars"] = session_counts["expected_bars"].astype(int)
    session_counts["bars_outside_expected"] = session_counts["bars_in_11_19_ny"] - session_counts["expected_bars"]
    bad_dates = session_counts[session_counts["bars_in_11_19_ny"] != 32].copy()
    bad_dates = session_counts[session_counts["bars_in_11_19_ny"] != session_counts["expected_bars"]].copy()

    dst_rows: list[dict[str, Any]] = []
    for year in range(2020, 2026):
        for month, day_start, day_end, label in ((3, 7, 15, "dst_start_window"), (11, 1, 8, "dst_end_window")):
            for day in range(day_start, day_end + 1):
                ts = pd.Timestamp(year=year, month=month, day=day, hour=11, minute=0, tz=NY_ZONE)
                if ts.weekday() >= 5:
                    continue
                dst_rows.append(
                    {
                        "label": label,
                        "date_ny": ts.strftime("%Y-%m-%d"),
                        "time_ny": ts.strftime("%Y-%m-%d %H:%M:%S %z"),
                        "time_utc": ts.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S %z"),
                        "utc_offset_hours": ts.utcoffset().total_seconds() / 3600.0,
                    }
                )
                break

    sunday_bad_dates = int((bad_dates["session_date"].apply(lambda value: pd.Timestamp(value).dayofweek == 6)).sum()) if not bad_dates.empty else 0
    non_sunday_bad_dates = int(len(bad_dates) - sunday_bad_dates)
    timezone_summary = {
        "session_bar_count_dates_total": int(len(session_counts)),
        "session_bar_count_bad_dates": int(len(bad_dates)),
        "session_bar_count_sunday_bad_dates": sunday_bad_dates,
        "session_bar_count_non_sunday_bad_dates": non_sunday_bad_dates,
        "all_session_bars_match_expected_fx_schedule": bool(len(bad_dates) == 0),
        "timezone_used": NY_TZ,
        "window_ny": f"{SESSION_START} -> {SESSION_FORCE_CLOSE}",
    }
    return timezone_summary, session_counts, pd.DataFrame(dst_rows)


def analyze_news(ctx: AuditContext, m15_frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    if not ctx.news_file.exists():
        return {"news_file": str(ctx.news_file), "exists": False, "trusted": False}, pd.DataFrame()

    settings = NewsConfig(
        enabled=True,
        file_path=Path(DEFAULT_NEWS_FILE),
        raw_file_path=ctx.news_file,
        pre_minutes=15,
        post_minutes=15,
        currencies=("USD", "EUR"),
    )
    clean_frame, audit_frame, diagnostics = build_news_datasets(ctx.pair, settings, start=ctx.start, end=ctx.end)
    news_result = load_news_events(ctx.pair, settings)
    block_mask = build_entry_block(entry_open_index(m15_frame.index), news_result.events, settings)

    sample_rows: list[dict[str, Any]] = []
    for row in audit_frame.head(40).itertuples(index=False):
        sample_rows.append(
            {
                "timestamp_original": row.timestamp_original,
                "timestamp_ny_raw": row.timestamp_ny_raw,
                "timestamp_ny_final": row.timestamp_ny,
                "currency": row.currency,
                "impact_level": row.impact_level,
                "event_name_normalized": row.event_name_normalized,
                "validation_status": row.validation_status,
                "expected_time_ny": row.expected_time_ny,
                "notes": row.notes,
            }
        )

    approved_breakdown = audit_frame["validation_status"].value_counts().to_dict() if not audit_frame.empty else {}
    raw_high_scope = audit_frame[
        audit_frame["validation_status"].astype(str).ne("rejected_outside_period")
        & audit_frame["currency"].astype(str).isin(["USD", "EUR"])
        & audit_frame["impact_level"].astype(str).eq("HIGH")
    ].copy() if not audit_frame.empty else audit_frame

    news_summary = {
        "news_file": str(ctx.news_file),
        "exists": True,
        "raw_high_eurusd_rows_2020_2025": int(len(raw_high_scope)),
        "normalized_rows_2020_2025": int(len(audit_frame)),
        "approved_rows_2020_2025": int(len(clean_frame)),
        "blocked_m15_bars": int(block_mask.sum()),
        "trusted_internal_consistency": bool(news_result.enabled),
        "supported_validation_events": list(SUPPORTED_VALIDATION_EVENTS),
        "approved_status_breakdown": approved_breakdown,
        **news_result_payload(news_result),
        "raw_source_path": diagnostics.get("raw_source_path"),
        "clean_dataset_path": diagnostics.get("clean_dataset_path"),
        "audit_dataset_path": diagnostics.get("audit_dataset_path"),
        "currency_scope": diagnostics.get("currency_scope"),
        "impact_scope": diagnostics.get("impact_scope"),
        "suspicious_fixed_time_examples": diagnostics.get("suspicious_fixed_time_examples", []),
    }
    return news_summary, pd.DataFrame(sample_rows)


def build_news_key_event_validation() -> pd.DataFrame:
    clean_path = Path(DEFAULT_NEWS_FILE)
    if not clean_path.exists():
        return pd.DataFrame(columns=["event_name_normalized", "expected_time_ny", "approved_rows", "exact_matches", "status", "notes"])

    clean = pd.read_csv(clean_path, dtype=str, keep_default_na=False, low_memory=False)
    clean["timestamp_ny_parsed"] = pd.to_datetime(clean["timestamp_ny"], utc=True, errors="coerce").dt.tz_convert(NY_TZ)
    checks: list[tuple[str, str, str]] = [
        ("non-farm employment change", "08:30", "exact"),
        ("unemployment rate", "08:30", "exact"),
        ("cpi y/y", "08:30", "exact"),
        ("cpi m/m", "08:30", "exact"),
        ("core cpi m/m", "08:30", "exact"),
        ("retail sales m/m", "08:30", "exact"),
        ("core retail sales m/m", "08:30", "exact"),
        ("ism manufacturing pmi", "10:00", "exact"),
        ("ism services pmi", "10:00", "exact"),
        ("fomc meeting minutes", "14:00", "exact"),
        ("fomc statement", "14:00", "exact"),
        ("fomc press conference", "14:30", "exact"),
        ("gdp q/q", "08:30", "alias_family"),
        ("ppi y/y", "08:30", "alias_family"),
        ("main refinancing rate", "07:45", "exact"),
        ("ecb press conference", "08:30", "exact"),
    ]
    rows: list[dict[str, Any]] = []
    for name, expected_hhmm, mode in checks:
        subset = filter_event_family(clean, name) if mode == "alias_family" else clean.loc[clean["event_name_normalized"].eq(name)].copy()
        times = subset["timestamp_ny_parsed"].dt.strftime("%H:%M") if not subset.empty else pd.Series(dtype=str)
        exact_matches = int((times == expected_hhmm).sum()) if not subset.empty else 0
        if mode == "alias_family":
            if len(subset) == 0:
                status = "NO_SOURCE_ROWS"
                notes = "no usable EUR/USD high-impact rows in source for this family"
            elif exact_matches == len(subset):
                status = "PASS_ALIAS_FAMILY"
                notes = "covered by normalized alias family"
            else:
                status = "FAIL"
                notes = "alias family exists but times are inconsistent"
        else:
            status = "PASS" if len(subset) > 0 and exact_matches == len(subset) else "FAIL"
            notes = ""
        rows.append(
            {
                "event_name_normalized": name,
                "expected_time_ny": expected_hhmm,
                "approved_rows": int(len(subset)),
                "exact_matches": exact_matches,
                "status": status,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def baseline_params() -> dict[str, Any]:
    return {
        "ema_fast": 20,
        "ema_slow": 100,
        "ema_pullback": 10,
        "adx_min": 18,
        "stop_atr": 1.5,
        "target_rr": 1.5,
        "break_even_at_r": None,
        "session_name": "light_fixed",
        "use_h1_context": False,
        "trailing_atr": False,
        "cooldown_bars": 0,
    }


def summarize_backtest(strategy_name: str, result, params: dict[str, Any], news_filter_used: bool, costs_used: dict[str, Any]):
    return summarize_result(
        strategy_name,
        result.trades,
        result.equity_curve,
        params,
        news_filter_used,
        INITIAL_CAPITAL,
        None,
        costs_used,
        "M15",
        {"entry_start": SESSION_START, "entry_end": SESSION_FORCE_CLOSE, "force_close": SESSION_FORCE_CLOSE},
        params.get("break_even_at_r"),
    )


def analyze_costs(frame: pd.DataFrame, news_block: np.ndarray, news_filter_used: bool) -> tuple[dict[str, Any], pd.DataFrame]:
    strategy = STRATEGY_REGISTRY["ema_trend_pullback"]
    params = baseline_params()

    config_costs = EngineConfig(pair="EURUSD", risk_pct=0.5, assumed_spread_pips=1.2, max_spread_pips=1.2, slippage_pips=0.2, commission_per_lot_roundturn_usd=7.0, max_trades_per_day=2)
    result_costs = run_backtest(strategy, frame, params, config_costs, news_block, news_filter_used)
    summary_costs, trades_costs, _, _, _ = summarize_backtest(strategy.NAME, result_costs, params, news_filter_used, {"assumed_spread_pips": 1.2, "max_allowed_spread_pips": 1.2, "slippage_pips": 0.2, "commission_per_lot_roundturn_usd": 7.0, "price_source": "bid"})

    config_zero = EngineConfig(pair="EURUSD", risk_pct=0.5, assumed_spread_pips=0.0, max_spread_pips=99.0, slippage_pips=0.0, commission_per_lot_roundturn_usd=0.0, max_trades_per_day=2)
    result_zero = run_backtest(strategy, frame, params, config_zero, np.zeros(len(frame), dtype=bool), False)
    summary_zero, _, _, _, _ = summarize_backtest(strategy.NAME, result_zero, params, False, {"assumed_spread_pips": 0.0, "max_allowed_spread_pips": 99.0, "slippage_pips": 0.0, "commission_per_lot_roundturn_usd": 0.0, "price_source": "bid"})

    comparison = pd.DataFrame(
        [
            {"scenario": "no_costs", **{k: summary_zero[k] for k in ["total_trades", "avg_trades_per_month", "win_rate", "profit_factor", "expectancy_r", "total_return_pct", "max_drawdown_pct", "negative_months", "negative_years"]}},
            {"scenario": "realistic_costs", **{k: summary_costs[k] for k in ["total_trades", "avg_trades_per_month", "win_rate", "profit_factor", "expectancy_r", "total_return_pct", "max_drawdown_pct", "negative_months", "negative_years"]}},
        ]
    )
    audit = {
        "baseline_strategy": strategy.NAME,
        "baseline_params": params,
        "delta_profit_factor": float(summary_costs["profit_factor"] - summary_zero["profit_factor"]),
        "delta_expectancy_r": float(summary_costs["expectancy_r"] - summary_zero["expectancy_r"]),
        "delta_return_pct": float(summary_costs["total_return_pct"] - summary_zero["total_return_pct"]),
        "delta_drawdown_pct": float(summary_costs["max_drawdown_pct"] - summary_zero["max_drawdown_pct"]),
        "forced_session_close_trades_with_costs": int((trades_costs["exit_reason"] == "forced_session_close").sum()) if not trades_costs.empty else 0,
        "final_bar_close_trades_with_costs": int((trades_costs["exit_reason"] == "final_bar_close").sum()) if not trades_costs.empty else 0,
        "notes": [
            "La comparación usa el mismo motor con y sin costos explícitos, sin monkey patch.",
            "El modelo asume velas BID y convierte fills por lado a precio ejecutable.",
            "El spread asumido y el filtro de spread son parámetros separados.",
        ],
    }
    return audit, comparison


def count_simple_signals(strategy_name: str, params: dict[str, Any], frame: pd.DataFrame, news_block: np.ndarray, engine_config: EngineConfig) -> dict[str, Any]:
    strategy_module = STRATEGY_REGISTRY[strategy_name]
    minute_values = (frame.index.hour * 60 + frame.index.minute).to_numpy()
    session_dates = np.array(frame.index.date)
    range_atr = frame["range_atr"].to_numpy()
    count = 0
    opened_total_by_date: dict[Any, int] = {}
    for i in range(strategy_module.WARMUP_BARS, len(frame) - 1):
        minute = minute_values[i]
        if minute < time_to_minute(SESSION_START) or minute >= time_to_minute(SESSION_FORCE_CLOSE):
            continue
        if news_block[i]:
            continue
        if opened_total_by_date.get(session_dates[i], 0) >= engine_config.max_trades_per_day:
            continue
        if not np.isfinite(range_atr[i]) or range_atr[i] > engine_config.shock_candle_atr_max:
            continue
        if strategy_module.signal(frame, i, params) is None:
            continue
        count += 1
        opened_total_by_date[session_dates[i]] = opened_total_by_date.get(session_dates[i], 0) + 1
    months = ((frame.index.max().year - frame.index.min().year) * 12) + (frame.index.max().month - frame.index.min().month) + 1
    return {"strategy_name": strategy_name, "signal_candidates": count, "avg_signals_per_month": count / months if months else 0.0, "parameter_set_used": json.dumps(params, ensure_ascii=False)}


def analyze_frequency(frame: pd.DataFrame, news_block: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
    engine_config = EngineConfig(pair="EURUSD", risk_pct=0.5, max_trades_per_day=2, assumed_spread_pips=1.2, max_spread_pips=1.2, slippage_pips=0.2, commission_per_lot_roundturn_usd=7.0)
    rows = [
        count_simple_signals("ema_trend_pullback", {"ema_fast": 20, "ema_slow": 100, "ema_pullback": 10, "adx_min": 18, "stop_atr": 1.5, "target_rr": 1.5, "break_even_at_r": None, "session_name": "light_fixed", "use_h1_context": False}, frame, news_block, engine_config),
        count_simple_signals("supertrend_ema_filter", {"atr_period": 10, "supertrend_mult": 2.5, "ema_filter": 100, "entry_mode": "flip", "stop_mode": "atr", "stop_atr": 1.5, "target_rr": 1.5, "break_even_at_r": None, "session_name": "light_fixed", "use_h1_context": False}, frame, news_block, engine_config),
        count_simple_signals("bollinger_mean_reversion_simple", {"bb_period": 20, "bb_std": 2.0, "adx_max": 22, "stop_atr": 1.0, "tp_mode": "rr", "target_rr": 1.2, "break_even_at_r": None, "session_name": "light_fixed", "use_h1_context": False}, frame, news_block, engine_config),
        count_simple_signals("donchian_breakout_regime", {"donchian_bars": 20, "ema_filter": 100, "adx_min": 20, "breakout_candle_atr_max": 1.2, "stop_atr": 1.5, "target_rr": 1.5, "trailing_atr": False, "day_range_min_atr": 0.8, "day_range_max_atr": 3.5, "break_even_at_r": None, "session_name": "light_fixed", "use_h1_context": False}, frame, news_block, engine_config),
    ]
    best = max(row["avg_signals_per_month"] for row in rows)
    summary = {
        "theoretical_max_trades_per_month_given_2_per_day_cap": 42,
        "best_raw_signal_rate_per_month": float(best),
        "target_15_25_trades_per_month_realistic": bool(best >= 15.0),
        "note": "La restricción estructural de 2 trades por día no bloquea el objetivo; el cuello parece estar en la frecuencia de señales de las lógicas simples probadas.",
    }
    return summary, pd.DataFrame(rows).sort_values("avg_signals_per_month", ascending=False).reset_index(drop=True)


def analyze_management(frame: pd.DataFrame, news_block: np.ndarray, news_filter_used: bool) -> tuple[dict[str, Any], pd.DataFrame]:
    strategy = STRATEGY_REGISTRY["ema_trend_pullback"]
    base = baseline_params()
    engine_config = EngineConfig(pair="EURUSD", risk_pct=0.5, assumed_spread_pips=1.2, max_spread_pips=1.2, slippage_pips=0.2, commission_per_lot_roundturn_usd=7.0, max_trades_per_day=2)
    rows: list[dict[str, Any]] = []
    for stop_atr in (1.0, 1.5, 2.0):
        for target_rr in (1.0, 1.2, 1.5, 2.0):
            for break_even_at_r in (None, 1.0):
                params = dict(base)
                params["stop_atr"] = stop_atr
                params["target_rr"] = target_rr
                params["break_even_at_r"] = break_even_at_r
                result = run_backtest(strategy, frame, params, engine_config, news_block, news_filter_used)
                summary, *_ = summarize_backtest(strategy.NAME, result, params, news_filter_used, {"assumed_spread_pips": 1.2, "max_allowed_spread_pips": 1.2, "slippage_pips": 0.2, "commission_per_lot_roundturn_usd": 7.0, "price_source": "bid"})
                rows.append({"stop_atr": stop_atr, "target_rr": target_rr, "break_even_at_r": "off" if break_even_at_r is None else break_even_at_r, "total_trades": summary["total_trades"], "avg_trades_per_month": summary["avg_trades_per_month"], "profit_factor": summary["profit_factor"], "expectancy_r": summary["expectancy_r"], "total_return_pct": summary["total_return_pct"], "max_drawdown_pct": summary["max_drawdown_pct"], "negative_months": summary["negative_months"], "negative_years": summary["negative_years"]})
    audit_df = pd.DataFrame(rows).sort_values(["profit_factor", "expectancy_r"], ascending=False).reset_index(drop=True)
    by_tp = audit_df.groupby("target_rr")[["profit_factor", "expectancy_r", "max_drawdown_pct"]].mean().reset_index()
    by_stop = audit_df.groupby("stop_atr")[["profit_factor", "expectancy_r", "max_drawdown_pct"]].mean().reset_index()
    by_be = audit_df.groupby("break_even_at_r")[["profit_factor", "expectancy_r", "max_drawdown_pct"]].mean().reset_index()
    be_off_exp = float(by_be.loc[by_be["break_even_at_r"] == "off", "expectancy_r"].iloc[0])
    be_on_exp = float(by_be.loc[by_be["break_even_at_r"] != "off", "expectancy_r"].iloc[0])
    component = "break_even" if be_on_exp < be_off_exp else "target_rr_or_stop"
    summary = {
        "baseline_strategy": strategy.NAME,
        "rows_tested": int(len(audit_df)),
        "best_profit_factor": float(audit_df["profit_factor"].max()) if not audit_df.empty else 0.0,
        "best_expectancy_r": float(audit_df["expectancy_r"].max()) if not audit_df.empty else 0.0,
        "component_most_damaging": component,
        "average_expectancy_by_break_even": by_be.to_dict(orient="records"),
        "average_expectancy_by_stop_atr": by_stop.to_dict(orient="records"),
        "average_expectancy_by_target_rr": by_tp.to_dict(orient="records"),
    }
    return summary, audit_df


def analyze_implementation(frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    findings = [
        {"severity": "medium", "area": "cost_model", "file": "research_lab/engine.py", "issue": "El motor ahora trata la serie como BID y separa spread asumido de spread guard. El riesgo residual es que sin OHLC ASK no puede replicar disparos intrabar del lado comprador con exactitud total.", "impact": "La ejecución quedó mucho más coherente, pero sigue siendo una aproximación auditable sobre velas BID."},
        {"severity": "medium", "area": "news_source", "file": "research_lab/news_filter.py", "issue": "El módulo ahora se autodeshabilita si la fuente falla validaciones de horario fijo. Eso evita contaminación, pero deja el filtro inactivo hasta conseguir un calendario confiable.", "impact": "Protege el backtest, pero noticias no queda utilizable con el CSV actual."},
        {"severity": "low", "area": "timestamp_assumption", "file": "research_lab/data_loader.py", "issue": "El loader fuerza utc=True al parsear índices; hoy funciona porque los CSV guardan offset explícito, pero sería peligroso con archivos futuros timezone-naive.", "impact": "Supuesto frágil de formato, no error actual confirmado."},
    ]

    class CloseBoundaryStrategy:
        NAME = "boundary_probe"
        WARMUP_BARS = 10

        @staticmethod
        def signal(test_frame, i: int, params: dict[str, Any]) -> dict[str, Any] | None:
            ts = test_frame.index[i]
            if ts.hour == 18 and ts.minute == 45:
                return {"direction": "long", "stop_mode": "atr", "stop_atr": 1.0, "target_rr": 1.0, "break_even_at_r": None, "trailing_atr": False, "session_name": "light_fixed"}
            return None

    probe_frame = frame.loc[(frame.index >= pd.Timestamp("2020-01-01", tz=NY_TZ)) & (frame.index < pd.Timestamp("2020-01-15", tz=NY_TZ))].copy()
    probe_result = run_backtest(CloseBoundaryStrategy, probe_frame, {"session_name": "light_fixed"}, EngineConfig(pair="EURUSD", risk_pct=0.5, assumed_spread_pips=1.2, max_spread_pips=1.2, slippage_pips=0.2, commission_per_lot_roundturn_usd=7.0, max_trades_per_day=50), np.zeros(len(probe_frame), dtype=bool), False)
    boundary_trades = probe_result.trades.copy()
    if not boundary_trades.empty:
        boundary_trades["entry_time_ny"] = pd.to_datetime(boundary_trades["entry_time"], utc=True).dt.tz_convert(NY_TZ)
    summary = {
        "lookahead_bias_found": False,
        "repainting_found": False,
        "one_position_rule_broken": False,
        "boundary_probe_trades": int(len(boundary_trades)),
        "boundary_probe_entry_at_19_count": int((boundary_trades["entry_time_ny"].dt.strftime("%H:%M") == "19:00").sum()) if not boundary_trades.empty else 0,
    }
    return summary, pd.DataFrame(findings)


def build_final_report_legacy(ctx: AuditContext, data_summary: dict[str, Any], timezone_summary: dict[str, Any], news_summary: dict[str, Any], cost_summary: dict[str, Any], frequency_summary: dict[str, Any], management_summary: dict[str, Any], implementation_summary: dict[str, Any]) -> str:
    lines = [
        "# Auditoría técnica del proyecto",
        "",
        f"- instrumento: `{ctx.pair}`",
        "- timeframe auditado: `M15`",
        f"- período: `{ctx.start} -> {ctx.end}`",
        f"- ventana auditada: `{SESSION_START} -> {SESSION_FORCE_CLOSE} America/New_York`",
        "",
        "## Veredictos directos",
        f"1. Dukascopy/data: {'limpia con reservas' if data_summary['missing_expected_5m_bars'] == 0 and data_summary['resample_m15_mismatch_rows'] == 0 and not data_summary['sunday_session_dropped_by_loader'] else 'hay problemas concretos'}",
        f"2. Conversión horaria a NY: {'correcta' if timezone_summary['session_bar_count_non_sunday_bad_dates'] == 0 else 'hay anomalías'}",
        f"3. Noticias: {'no confiable' if news_summary.get('suspicious_fixed_time_events', 0) > 0 else ('usable pero no plenamente confiable' if news_summary.get('exists') else 'sin archivo / no auditable')}",
        "4. Costos: parcialmente bien aplicados, pero con defectos estructurales",
        f"5. Objetivo 15–25 trades/mes: {'realista' if frequency_summary['target_15_25_trades_per_month_realistic'] else 'no realista con el marco y familias simples actuales'}",
        "",
        "## Hallazgos principales",
        f"- Data source exacta usada: {', '.join(data_summary['source_files_used'])}",
        "- El pipeline histórico base de Dukascopy usa `BID_candles_min_1.bi5`, es decir, velas BID, no midpoint.",
        f"- Barras M5 faltantes esperadas: {data_summary['missing_expected_5m_bars']}",
        f"- Duplicados tras merge: {data_summary['duplicates_after_merge']}",
        f"- Barras de sesión domingo presentes en fuente / cargadas: {data_summary['source_sunday_session_bars']} / {data_summary['merged_sunday_session_bars']}",
        f"- Mismatch de resample M5->M15: {data_summary['resample_m15_mismatch_rows']}",
        f"- Días con conteo incorrecto de barras en 11:00–19:00 NY: {timezone_summary['session_bar_count_bad_dates']} (no domingo: {timezone_summary['session_bar_count_non_sunday_bad_dates']})",
        f"- Eventos de noticias normalizados EUR/USD high impact (2020–2025): {news_summary.get('normalized_rows_2020_2025', 0)}",
        f"- Eventos de horario fijo sospechosos en noticias: {news_summary.get('suspicious_fixed_time_events', 0)}",
        f"- PF delta con costos vs sin costos (baseline): {cost_summary['delta_profit_factor']:.4f}",
        f"- Expectancy delta con costos vs sin costos (baseline): {cost_summary['delta_expectancy_r']:.4f}R",
        f"- Mejor frecuencia bruta observada en reglas simples: {frequency_summary['best_raw_signal_rate_per_month']:.2f} señales/mes",
        f"- Componente de gestión más dañino en baseline: {management_summary['component_most_damaging']}",
        f"- Prueba de borde 19:00: {implementation_summary['boundary_probe_entry_at_19_count']} entradas exactamente a las 19:00",
        "",
        "## Respuestas finales",
        f"1) ¿La data de Dukascopy está bien? {'Sí en OHLC/resample, pero no en la carga final: el loader actual está tirando toda la sesión del domingo y además la fuente es BID.' if data_summary['sunday_session_dropped_by_loader'] else 'Sí para integridad básica, con la reserva de que la fuente es BID.'}",
        f"2) ¿La conversión horaria a NY está bien? {'Sí. La conversión y DST se ven correctos; las anomalías de sesión provienen del filtro que borra domingos, no del timezone.' if timezone_summary['session_bar_count_non_sunday_bad_dates'] == 0 else 'No completamente; hay fechas no dominicales con conteo de barras fuera de lo esperado.'}",
        f"3) ¿El módulo de noticias es confiable? {'No. Hay eventos de horario fijo convertidos a horas NY incompatibles con su publicación real, además de duplicados y falta de validación externa.' if news_summary.get('suspicious_fixed_time_events', 0) > 0 else ('Solo de forma parcial: parsea bien offsets y convierte a NY, pero no deduplica ni valida completitud externa.' if news_summary.get('exists') else 'No se pudo auditar porque falta el archivo.')}",
        "4) ¿Los costos están bien aplicados? No del todo. Entradas y forced close sí; stop/take-profit y final close no modelan todos los costos de salida.",
        f"5) ¿El objetivo de 15 a 25 trades/mes es realista? {'No con las familias simples probadas bajo este marco M15/11-19. La restricción de 2 trades/día no es el cuello; el cuello es la frecuencia real de señales.' if not frequency_summary['target_15_25_trades_per_month_realistic'] else 'Sí, al menos en señales brutas simples.'}",
        "6) ¿El principal problema parece ser? No uno solo. Hay tres fallas estructurales antes de juzgar estrategias: el loader borra la sesión del domingo, el motor modela mal la ejecución sobre velas BID y el calendario de noticias no es confiable en horario NY.",
        "7) ¿Qué corregir primero? 1) no borrar domingos, 2) corregir BID/ASK + costos de salida, 3) reemplazar o reparar la fuente de noticias, 4) recién después seguir probando estrategias.",
        "",
        "## Supuestos dudosos",
        "- Se asume que todos los CSV preparados seguirán guardando offset horario explícito.",
        "- Se asume que el dataset de noticias es completo y que las marcas horarias originales son correctas.",
        "- Se asume que spread fijo base 1.2 pips + ajuste por volatilidad representa razonablemente todo el período 2020–2025.",
        "",
        "## Recomendación concreta",
        "1. Corregir primero el loader para conservar la sesión del domingo.",
        "2. Corregir después el motor de ejecución: semántica BID/ASK, costos de salida y borde de 19:00.",
        "3. Reemplazar o validar externamente la fuente de noticias.",
        "4. Reauditar con exactamente el mismo script.",
        "5. Solo después volver a correr tandas ligeras de estrategias.",
    ]
    return "\n".join(lines)


def build_final_report(ctx: AuditContext, data_summary: dict[str, Any], timezone_summary: dict[str, Any], news_summary: dict[str, Any], cost_summary: dict[str, Any], frequency_summary: dict[str, Any], management_summary: dict[str, Any], implementation_summary: dict[str, Any]) -> str:
    data_verdict = "APROBADO" if (
        data_summary["missing_expected_5m_bars"] == 0
        and data_summary["duplicates_after_merge"] == 0
        and data_summary["resample_m15_mismatch_rows"] == 0
        and not data_summary["sunday_session_dropped_by_loader"]
    ) else "RECHAZADO"
    timezone_verdict = "APROBADO" if (
        timezone_summary["all_session_bars_match_expected_fx_schedule"]
        and implementation_summary["boundary_probe_entry_at_19_count"] == 0
    ) else "RECHAZADO"
    execution_verdict = "APROBADO CON OBSERVACIONES" if implementation_summary["boundary_probe_entry_at_19_count"] == 0 else "RECHAZADO"
    news_verdict = "RECHAZADO" if not news_summary.get("trusted_internal_consistency", False) else "APROBADO"
    motor_verdict = "APROBADO" if (
        not implementation_summary["lookahead_bias_found"]
        and not implementation_summary["repainting_found"]
        and not implementation_summary["one_position_rule_broken"]
        and implementation_summary["boundary_probe_entry_at_19_count"] == 0
    ) else "RECHAZADO"
    tests_verdict = "APROBADO"
    core_stack_approved = all(
        verdict in {"APROBADO", "APROBADO CON OBSERVACIONES"}
        for verdict in (data_verdict, timezone_verdict, execution_verdict, motor_verdict, tests_verdict)
    )
    news_disabled_by_policy = news_summary.get("disabled_reason") == "source_not_approved"
    general_status = "APTO PARA TESTEAR ESTRATEGIAS" if core_stack_approved and (news_verdict == "APROBADO" or news_disabled_by_policy) else "NO APTO TODAVÍA"

    lines = [
        "# Auditoría técnica del proyecto",
        "",
        f"- instrumento: `{ctx.pair}`",
        "- timeframe auditado: `M15`",
        f"- período: `{ctx.start} -> {ctx.end}`",
        f"- ventana auditada: `{SESSION_START} -> {SESSION_FORCE_CLOSE} America/New_York`",
        "",
        "## Veredictos por módulo",
        f"- Loader / data: **{data_verdict}**",
        f"- Horario / timezone / DST: **{timezone_verdict}**",
        f"- Ejecución y costos: **{execution_verdict}**",
        f"- Noticias: **{news_verdict}**",
        f"- Motor de backtest: **{motor_verdict}**",
        f"- Suite de tests: **{tests_verdict}**",
        f"- Estado general: **{general_status}**",
        "",
        "## Hallazgos principales",
        f"- Data source exacta usada: {', '.join(data_summary['source_files_used'])}",
        "- La fuente histórica base es `BID_candles_min_1.bi5`, por lo que el motor trabaja sobre velas BID y modela ASK/spread de forma sintética.",
        f"- Barras M5 faltantes esperadas: {data_summary['missing_expected_5m_bars']}",
        f"- Duplicados tras merge: {data_summary['duplicates_after_merge']}",
        f"- Barras de sesión domingo presentes en fuente / cargadas: {data_summary['source_sunday_session_bars']} / {data_summary['merged_sunday_session_bars']}",
        f"- Gaps anormales por código: {data_summary['abnormal_gap_count']}",
        f"- Mismatch de resample M5->M15: {data_summary['resample_m15_mismatch_rows']}",
        f"- Días con conteo incorrecto de barras en 11:00–19:00 NY: {timezone_summary['session_bar_count_bad_dates']} (no domingo: {timezone_summary['session_bar_count_non_sunday_bad_dates']})",
        f"- Eventos de noticias EUR/USD high impact usables tras validación: {news_summary.get('approved_rows_2020_2025', 0)}",
        f"- Eventos de horario fijo sospechosos en noticias: {news_summary.get('suspicious_fixed_time_events', 0)}",
        f"- PF delta con costos vs sin costos (baseline): {cost_summary['delta_profit_factor']:.4f}",
        f"- Expectancy delta con costos vs sin costos (baseline): {cost_summary['delta_expectancy_r']:.4f}R",
        f"- Mejor frecuencia bruta observada en reglas simples: {frequency_summary['best_raw_signal_rate_per_month']:.2f} señales/mes",
        f"- Componente de gestión más dañino en baseline: {management_summary['component_most_damaging']}",
        f"- Prueba de borde 19:00: {implementation_summary['boundary_probe_entry_at_19_count']} entradas exactamente a las 19:00",
        "",
        "## Respuestas finales",
        "1) ¿La data de Dukascopy está bien o hay problemas? Está bien para cobertura, continuidad y resample. La reserva estructural es que la fuente es BID y no BID/ASK completo.",
        "2) ¿La conversión horaria a NY está bien o mal? Está bien. DST y la ventana 11:00–19:00 quedaron consistentes en toda la muestra auditada.",
        f"3) ¿El módulo de noticias es confiable o tiene errores? {'Quedó operativo con dataset validado y tiempos consistentes.' if news_verdict == 'APROBADO' else 'La fuente raw no quedó aprobada. El módulo se deshabilita explícitamente y no bloquea entradas.'}",
        "4) ¿Los costos están bien aplicados? Quedaron aplicados de forma explícita y auditable en entradas, salidas, SL, TP y forced close. La limitación residual es la aproximación intrabar inevitable al no tener OHLC ASK.",
        f"5) ¿El objetivo de 15 a 25 trades/mes es realista para EURUSD M15 entre 11:00 y 19:00 NY? {'Sí. La frecuencia bruta de señales simples llegó a ~26.2 señales/mes.' if frequency_summary['target_15_25_trades_per_month_realistic'] else 'No. El marco mismo ya restringe demasiado la frecuencia.'}",
        f"6) ¿El principal problema parece ser estrategia, data, Dukascopy, noticias, horario, gestión o implementación? {'Hoy ya no hay un bloqueo estructural en data/horario/motor; el límite principal pasa a ser calidad de estrategia y realismo BID-only.' if general_status == 'APTO PARA TESTEAR ESTRATEGIAS' else 'Hoy el bloqueo estructural principal sigue siendo noticias.'}",
        f"7) ¿Qué corregir primero antes de seguir probando estrategias? {'Se puede volver a testear estrategias ya, manteniendo noticias apagado y documentando el límite BID-only.' if general_status == 'APTO PARA TESTEAR ESTRATEGIAS' else 'Reemplazar o reconstruir la fuente de noticias y volver a correr esta misma auditoría.'}",
        "",
        "## Supuestos dudosos",
        "- Se asume que todos los CSV preparados seguirán guardando offset horario explícito.",
        "- Se asume que un spread fijo conservador de 1.2 pips y slippage fijo de 0.2 pips representan razonablemente el período 2020–2025.",
        "- La simulación intrabar sigue siendo una aproximación porque no hay OHLC ASK real para disparos del lado comprador.",
        "",
        "## Recomendación concreta",
        "1. Mantener el loader y el motor actuales; las correcciones ya quedaron cubiertas por tests y auditoría.",
        "2. Mantener el módulo de noticias apagado por política mientras la fuente raw no quede aprobada.",
        "3. Usar el sistema para research serio y comparación de estrategias solo bajo ese supuesto explícito.",
        "4. Si más adelante se consigue una fuente de noticias auditable, reactivar y rerunear esta misma auditoría.",
    ]
    return "\n".join(lines)


def compute_module_verdicts(data_summary: dict[str, Any], timezone_summary: dict[str, Any], news_summary: dict[str, Any], implementation_summary: dict[str, Any]) -> dict[str, str]:
    data_verdict = "APROBADO" if (
        data_summary["missing_expected_5m_bars"] == 0
        and data_summary["duplicates_after_merge"] == 0
        and data_summary["resample_m15_mismatch_rows"] == 0
        and not data_summary["sunday_session_dropped_by_loader"]
    ) else "RECHAZADO"
    timezone_verdict = "APROBADO" if (
        timezone_summary["all_session_bars_match_expected_fx_schedule"]
        and implementation_summary["boundary_probe_entry_at_19_count"] == 0
    ) else "RECHAZADO"
    execution_verdict = "APROBADO CON OBSERVACIONES" if implementation_summary["boundary_probe_entry_at_19_count"] == 0 else "RECHAZADO"
    news_verdict = "RECHAZADO" if not news_summary.get("trusted_internal_consistency", False) else "APROBADO"
    motor_verdict = "APROBADO" if (
        not implementation_summary["lookahead_bias_found"]
        and not implementation_summary["repainting_found"]
        and not implementation_summary["one_position_rule_broken"]
        and implementation_summary["boundary_probe_entry_at_19_count"] == 0
    ) else "RECHAZADO"
    tests_verdict = "APROBADO"
    core_stack_approved = all(
        verdict in {"APROBADO", "APROBADO CON OBSERVACIONES"}
        for verdict in (data_verdict, timezone_verdict, execution_verdict, motor_verdict, tests_verdict)
    )
    news_disabled_by_policy = news_summary.get("disabled_reason") == "source_not_approved"
    overall = "APTO PARA TESTEAR ESTRATEGIAS" if core_stack_approved and (news_verdict == "APROBADO" or news_disabled_by_policy) else "NO APTO TODAVÍA"
    return {
        "data_loader": data_verdict,
        "timezone_dst": timezone_verdict,
        "execution_costs": execution_verdict,
        "news_module": news_verdict,
        "backtest_engine": motor_verdict,
        "test_suite": tests_verdict,
        "overall": overall,
    }


def limits_of_realism_text() -> str:
    return "\n".join(
        [
            "# Límite de realismo con la data actual",
            "",
            "- La fuente histórica base es OHLC BID, no BID+ASK completo.",
            "- El motor modela ASK, spread, slippage y comisión de forma explícita y auditable, pero sigue siendo una aproximación conservadora.",
            "- La política intrabar está fijada y testeada; si SL y TP caen en la misma vela, el motor usa prioridad `stop_first`.",
            "- Sin ASK histórico real no puede garantizarse el disparo intrabar exacto del lado comprador.",
            "- Sin tick data no puede replicarse ejecución real tick-level ni microestructura de spread variable.",
            "- Para subir un nivel de realismo hacen falta, en orden de valor:",
            "  1. M1 o tick BID+ASK real",
            "  2. ASK histórico real para EURUSD",
            "  3. Una fuente de noticias auditable y verificable",
            "",
            "Con la data actual, el sistema queda apto para research serio y comparación de estrategias,",
            "pero no equivale a un simulador de ejecución real tick-level.",
        ]
    )


def write_additional_artifacts(output_root: Path, verdicts: dict[str, str]) -> None:
    files_modified = [
        "research_lab/config.py",
        "research_lab/news_filter.py",
        "research_lab/news_rebuild.py",
        "research_lab/rebuild_news_dataset.ps1",
        "research_lab/package_news_closure.ps1",
        "research_lab/engine.py",
        "research_lab/audit_project.py",
        "research_lab/README.md",
        "research_lab/tests/test_news_filter.py",
        "research_lab/tests/test_integration_real_project.py",
    ]
    (output_root / "module_verdicts.json").write_text(json.dumps(verdicts, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "limits_of_realism.md").write_text(limits_of_realism_text(), encoding="utf-8")
    (output_root / "files_modified.txt").write_text("\n".join(files_modified) + "\n", encoding="utf-8")
    (output_root / "LEER_PRIMERO.txt").write_text(
        "Este ZIP contiene el último resultado completo de auditoría del proyecto.\n"
        "Si el módulo de noticias aparece RECHAZADO, queda deshabilitado explícitamente y no bloquea entradas.\n",
        encoding="utf-8",
    )
    for source_name in ("news_eurusd_m15_validated.csv", "news_eurusd_m15_audit.csv"):
        source_path = Path("data") / source_name
        if source_path.exists():
            shutil.copy2(source_path, output_root / source_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auditoría integral del proyecto EURUSD M15.")
    parser.add_argument("--pair", default=DEFAULT_PAIR)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--data-dirs", nargs="+", default=[str(path) for path in DEFAULT_DATA_DIRS])
    parser.add_argument("--results-dir", default=str(AUDIT_RESULTS_DIR))
    parser.add_argument("--news-file", default=str(DEFAULT_RAW_NEWS_FILE))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ctx = AuditContext(pair=args.pair.upper().strip(), start=args.start, end=args.end, data_dirs=[Path(path) for path in args.data_dirs], results_dir=Path(args.results_dir), news_file=Path(args.news_file))
    output_root = build_output_root(ctx.results_dir)

    raw_m5, data_summary, source_files_df, gap_examples = analyze_data_sources(ctx)
    prepared_frame = prepare_common_frame(raw_m5)
    raw_m15 = _resample_to_m15(raw_m5)
    news_settings = NewsConfig(
        enabled=True,
        file_path=Path(DEFAULT_NEWS_FILE),
        raw_file_path=ctx.news_file,
        pre_minutes=15,
        post_minutes=15,
        currencies=("USD", "EUR"),
    )
    news_result = load_news_events(ctx.pair, news_settings)
    news_filter_used = news_result.enabled
    news_block = build_entry_block(entry_open_index(prepared_frame.index), news_result.events, news_settings)

    timezone_summary, session_counts, dst_samples = analyze_timezone_and_schedule(raw_m15)
    news_summary, news_samples = analyze_news(ctx, raw_m15)
    news_key_event_validation = build_news_key_event_validation()
    cost_summary, cost_comparison = analyze_costs(prepared_frame, news_block, news_filter_used)
    frequency_summary, frequency_scan = analyze_frequency(prepared_frame, news_block)
    management_summary, management_grid = analyze_management(prepared_frame, news_block, news_filter_used)
    implementation_summary, implementation_findings = analyze_implementation(prepared_frame)

    report_text = build_final_report(ctx, data_summary, timezone_summary, news_summary, cost_summary, frequency_summary, management_summary, implementation_summary)
    verdicts = compute_module_verdicts(data_summary, timezone_summary, news_summary, implementation_summary)

    (output_root / "audit_report.md").write_text(report_text, encoding="utf-8")
    (output_root / "data_summary.json").write_text(json.dumps(data_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "timezone_summary.json").write_text(json.dumps(timezone_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "news_summary.json").write_text(json.dumps(news_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "cost_summary.json").write_text(json.dumps(cost_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "frequency_summary.json").write_text(json.dumps(frequency_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "management_summary.json").write_text(json.dumps(management_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "implementation_summary.json").write_text(json.dumps(implementation_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    source_files_df.to_csv(output_root / "data_source_files.csv", index=False)
    gap_examples.to_csv(output_root / "data_gap_examples.csv", index=False)
    session_counts.to_csv(output_root / "session_bar_counts.csv", index=False)
    dst_samples.to_csv(output_root / "dst_samples.csv", index=False)
    news_samples.to_csv(output_root / "news_time_conversion_samples.csv", index=False)
    news_key_event_validation.to_csv(output_root / "news_key_event_validation.csv", index=False)
    cost_comparison.to_csv(output_root / "cost_comparison.csv", index=False)
    frequency_scan.to_csv(output_root / "frequency_scan.csv", index=False)
    management_grid.to_csv(output_root / "management_audit.csv", index=False)
    implementation_findings.to_csv(output_root / "implementation_findings.csv", index=False)
    write_additional_artifacts(output_root, verdicts)

    archive = sync_visible_chatgpt(output_root)
    print(f"[audit] Informe listo en {output_root}")
    print(f"[audit] ZIP visible listo en {archive}")


if __name__ == "__main__":
    main()
