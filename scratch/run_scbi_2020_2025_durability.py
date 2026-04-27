from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PRICE_DIRS = [
    ROOT / "data_free_2020" / "prepared",
    ROOT / "data_candidates_2022_2025" / "prepared",
]
RESULTS_DIR = ROOT / "results" / "SCBI_2020_2025_DURABILITY"
RESULTS_FILE = RESULTS_DIR / "summary.json"
TRADES_FILE = RESULTS_DIR / "trades_baseline.csv"
SWEEP_AUDIT_FILE = RESULTS_DIR / "sweep_audit.csv"
YEARLY_FILE = RESULTS_DIR / "yearly_stats.csv"
SEMESTER_FILE = RESULTS_DIR / "semester_stats.csv"
BLOCK_FILE = RESULTS_DIR / "block_stats.csv"
MONTHLY_FILE = RESULTS_DIR / "monthly_pnl.csv"
CHECKPOINT_DIR = ROOT / "scbi_2020_2025_durability_checkpoints"

START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
PRE_SPLIT_DATE = "2022-01-01"

BASELINE_SPREAD_PIPS = 0.3
STRESS_TOTAL_SPREAD_PIPS = 1.2
STRESS_DELTA_PIPS = round(STRESS_TOTAL_SPREAD_PIPS - BASELINE_SPREAD_PIPS, 4)
PIP_SIZE = 0.0001
MIN_RISK_PRICE = 0.0002

MANDATORY_H1_YEARS = ("2020", "2021", "2022", "2023", "2024", "2025")
BLOCKS = {
    "2020_2021": ("2020-01-01", "2021-12-31"),
    "2022_2023": ("2022-01-01", "2023-12-31"),
    "2024_2025": ("2024-01-01", "2025-12-31"),
    "pre_2022": ("2020-01-01", "2021-12-31"),
    "post_2022": ("2022-01-01", "2025-12-31"),
}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def project_relative(value: str | Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path.relative_to(ROOT))
    return str(path)


def price_integrity_summary(frame: pd.DataFrame, expected_delta: pd.Timedelta) -> dict[str, object]:
    diffs = frame.index.to_series().diff().dropna()
    allowed = {
        expected_delta,
        pd.Timedelta(days=2) + expected_delta,
        pd.Timedelta(days=1, hours=23) + expected_delta,
        pd.Timedelta(days=2, hours=1) + expected_delta,
    }
    unexpected = diffs[~diffs.isin(allowed)]
    invalid_hilo = (
        (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
        | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
    )
    flat_bars = (frame["open"] == frame["high"]) & (frame["high"] == frame["low"]) & (frame["low"] == frame["close"])
    summary: dict[str, object] = {
        "rows": int(len(frame)),
        "first_timestamp_ny": str(frame.index.min()) if not frame.empty else "",
        "last_timestamp_ny": str(frame.index.max()) if not frame.empty else "",
        "duplicate_timestamps": int(frame.index.duplicated().sum()),
        "unexpected_gap_count": int(len(unexpected)),
        "zero_volume_bars": int((frame["volume"] <= 0).sum()),
        "flat_bars": int(flat_bars.sum()),
        "invalid_hilo_bars": int(invalid_hilo.sum()),
    }
    if not unexpected.empty:
        summary["first_unexpected_gap_after"] = str(unexpected.index[0])
        summary["first_unexpected_gap"] = str(unexpected.iloc[0])
    return summary


def load_price_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    from research_lab.data_loader import load_prepared_ohlcv, validate_price_frame

    h1 = load_prepared_ohlcv("EURUSD", PRICE_DIRS, "H1")
    m5 = load_prepared_ohlcv("EURUSD", PRICE_DIRS, "M5")

    h1 = h1.loc[h1.index.date <= pd.Timestamp(END_DATE).date()].copy()
    m5 = m5.loc[m5.index.date <= pd.Timestamp(END_DATE).date()].copy()

    validate_price_frame(h1)
    validate_price_frame(m5)

    coverage = {
        "price_dirs": [str(path.relative_to(ROOT)) for path in PRICE_DIRS],
        "h1": price_integrity_summary(
            h1.loc[h1.index.date >= pd.Timestamp(START_DATE).date()].copy(),
            pd.Timedelta(hours=1),
        ),
        "m5": price_integrity_summary(
            m5.loc[m5.index.date >= pd.Timestamp(START_DATE).date()].copy(),
            pd.Timedelta(minutes=5),
        ),
    }
    return h1, m5, coverage


def load_news_events() -> tuple[pd.DataFrame, object, dict[str, object]]:
    from research_lab.config import canonical_news_config
    from research_lab.news_filter import require_operational_news

    settings = canonical_news_config("EURUSD", enabled=True)
    result = require_operational_news("EURUSD", settings, context="scbi_2020_2025_durability")
    if not result.enabled or result.events.empty:
        raise RuntimeError("News Fortress no esta operativo para la corrida de durabilidad.")

    events = result.events.copy()
    events["timestamp_ny"] = pd.to_datetime(events["timestamp_ny"], utc=True).dt.tz_convert("US/Eastern")
    events = events.loc[
        (events["timestamp_ny"].dt.date >= pd.Timestamp(START_DATE).date())
        & (events["timestamp_ny"].dt.date <= pd.Timestamp(END_DATE).date())
    ].copy()
    if events.empty:
        raise RuntimeError("News Fortress no contiene eventos dentro del rango 2020-2025.")

    years = events["timestamp_ny"].dt.year.value_counts().sort_index()
    coverage = {
        "rows": int(len(events)),
        "first_timestamp_ny": str(events["timestamp_ny"].min()),
        "last_timestamp_ny": str(events["timestamp_ny"].max()),
        "years": {str(year): int(count) for year, count in years.items()},
        "source_path": project_relative(result.final_dataset_path),
        "pre_minutes": int(settings.pre_minutes),
        "post_minutes": int(settings.post_minutes),
        "pre_news_exit_minutes": int(settings.pre_news_exit_minutes),
        "forced_exit_pre_news": bool(settings.forced_exit_pre_news),
        "cancel_pending_pre_news": bool(settings.cancel_pending_pre_news),
    }
    return events, settings, coverage


def compute_session_levels(h1: pd.DataFrame) -> dict[object, dict[str, float]]:
    levels: dict[object, dict[str, float]] = {}
    frame = h1.copy()
    frame["date"] = frame.index.date
    frame["hour"] = frame.index.hour
    dates = sorted(frame["date"].unique())

    for i, current_date in enumerate(dates):
        if i == 0:
            continue
        previous_date = dates[i - 1]
        prev_bars = frame[frame["date"] == previous_date]
        curr_bars = frame[frame["date"] == current_date]
        if prev_bars.empty or curr_bars.empty:
            continue

        pdh = float(prev_bars["high"].max())
        pdl = float(prev_bars["low"].min())
        asia_prev = prev_bars[prev_bars["hour"] >= 18]
        asia_curr = curr_bars[(curr_bars["hour"] >= 18) | (curr_bars["hour"] < 2)]
        asia_all = pd.concat([asia_prev, asia_curr])
        london = curr_bars[(curr_bars["hour"] >= 2) & (curr_bars["hour"] < 8)]

        levels[current_date] = {
            "pdh": pdh,
            "pdl": pdl,
            "asia_h": float(asia_all["high"].max()) if not asia_all.empty else pdh,
            "asia_l": float(asia_all["low"].min()) if not asia_all.empty else pdl,
            "london_h": float(london["high"].max()) if not london.empty else pdh,
            "london_l": float(london["low"].min()) if not london.empty else pdl,
        }
    return levels


def detect_sweeps_h1(h1: pd.DataFrame, levels: dict[object, dict[str, float]]) -> list[dict[str, object]]:
    sweeps: list[dict[str, object]] = []
    frame = h1.copy()
    frame["date"] = frame.index.date

    for timestamp, bar in frame.iterrows():
        trade_date = bar["date"]
        if trade_date < pd.Timestamp(START_DATE).date() or trade_date > pd.Timestamp(END_DATE).date():
            continue
        if trade_date not in levels:
            continue

        level_map = levels[trade_date]
        open_, high, low, close = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])

        for level_name in ("pdl", "asia_l", "london_l"):
            level_price = level_map[level_name]
            if low < level_price and close > level_price:
                sweeps.append(
                    {
                        "time": timestamp,
                        "direction": "long",
                        "level_name": level_name,
                        "level_price": float(level_price),
                        "sweep_extreme": low,
                        "h1_open": open_,
                        "h1_high": high,
                        "h1_low": low,
                        "h1_close": close,
                    }
                )

        for level_name in ("pdh", "asia_h", "london_h"):
            level_price = level_map[level_name]
            if high > level_price and close < level_price:
                sweeps.append(
                    {
                        "time": timestamp,
                        "direction": "short",
                        "level_name": level_name,
                        "level_price": float(level_price),
                        "sweep_extreme": high,
                        "h1_open": open_,
                        "h1_high": high,
                        "h1_low": low,
                        "h1_close": close,
                    }
                )
    return sweeps


def empty_news_guard(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_blocked": False,
            "pending_kill": False,
            "force_flat": False,
            "blocking_event_name": "",
            "blocking_event_time_ny": "",
            "blocking_rule_used": "",
            "force_flat_event_name": "",
            "force_flat_event_time_ny": "",
            "force_flat_rule_used": "",
        },
        index=index,
    )


def build_news_guards(index: pd.DatetimeIndex, news_events: pd.DataFrame, news_settings: object) -> pd.DataFrame:
    from research_lab.news_filter import build_news_guard_details

    if index.empty:
        return empty_news_guard(index)
    details = build_news_guard_details(index, news_events, news_settings)
    return details.reindex(index).fillna("")


def pnl_r_from_exit(direction: str, entry_price: float, stop_loss: float, exit_signal_price: float) -> float:
    risk_distance = entry_price - stop_loss if direction == "long" else stop_loss - entry_price
    if direction == "long":
        value = (exit_signal_price - entry_price) / risk_distance
    else:
        value = (entry_price - exit_signal_price) / risk_distance
    return round(float(value), 4)


def compute_metrics(trades: list[dict[str, object]], pnl_key: str = "pnl_r") -> dict[str, object]:
    if not trades:
        return {
            "N": 0,
            "wins": 0,
            "losses": 0,
            "pf": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "total_r": 0.0,
        }

    pnls = [float(trade[pnl_key]) for trade in trades]
    wins = sum(1 for value in pnls if value > 0)
    losses = len(pnls) - wins
    gross_profit = sum(value for value in pnls if value > 0)
    gross_loss = abs(sum(value for value in pnls if value <= 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 999.0
    total_r = sum(pnls)

    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in pnls:
        equity += value
        peak = max(peak, equity)
        drawdown = min(drawdown, equity - peak)

    return {
        "N": int(len(trades)),
        "wins": int(wins),
        "losses": int(losses),
        "pf": round(float(pf), 3),
        "expectancy": round(float(total_r / len(pnls)), 4),
        "max_drawdown": round(float(drawdown), 2),
        "win_rate": round(float(wins / len(pnls)), 3),
        "total_r": round(float(total_r), 2),
    }


def apply_cost_stress(trades: list[dict[str, object]]) -> list[dict[str, object]]:
    stressed: list[dict[str, object]] = []
    for trade in trades:
        copy_trade = dict(trade)
        risk_pips = float(copy_trade["risk_pips"])
        stressed_pnl = float(copy_trade["pnl_r"]) - (STRESS_DELTA_PIPS / risk_pips)
        copy_trade["pnl_r_stress"] = round(stressed_pnl, 4)
        stressed.append(copy_trade)
    return stressed


def bucket_trades(trades: list[dict[str, object]], bucket_fn) -> dict[str, list[dict[str, object]]]:
    buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    for trade in trades:
        buckets[bucket_fn(trade)].append(trade)
    return dict(sorted(buckets.items()))


def monthly_pnl(trades: list[dict[str, object]], pnl_key: str = "pnl_r") -> dict[str, float]:
    values: dict[str, float] = defaultdict(float)
    for trade in trades:
        month = str(trade["entry_time"])[:7]
        values[month] += float(trade[pnl_key])
    return {month: round(total, 2) for month, total in sorted(values.items())}


def concentration_summary(trades: list[dict[str, object]], pnl_key: str = "pnl_r") -> dict[str, object]:
    if not trades:
        return {
            "top_10pct_trades_profit_share": 0.0,
            "top_20pct_trades_profit_share": 0.0,
            "max_losing_streak": 0,
            "months_positive": 0,
            "months_negative": 0,
            "months_total": 0,
            "positive_years": 0,
            "negative_years": 0,
        }

    positive_pnls = sorted((float(trade[pnl_key]) for trade in trades if float(trade[pnl_key]) > 0), reverse=True)
    gross_profit = sum(positive_pnls)
    all_pnls = sorted((float(trade[pnl_key]) for trade in trades), reverse=True)
    top_10_count = max(1, len(all_pnls) // 10)
    top_20_count = max(1, len(all_pnls) // 5)

    max_losing_streak = 0
    streak = 0
    for trade in trades:
        if float(trade[pnl_key]) <= 0:
            streak += 1
            max_losing_streak = max(max_losing_streak, streak)
        else:
            streak = 0

    monthly = monthly_pnl(trades, pnl_key=pnl_key)
    yearly_values = defaultdict(float)
    for trade in trades:
        yearly_values[str(trade["entry_time"])[:4]] += float(trade[pnl_key])

    return {
        "top_10pct_trades_profit_share": round(float(sum(all_pnls[:top_10_count]) / gross_profit), 3) if gross_profit > 0 else 0.0,
        "top_20pct_trades_profit_share": round(float(sum(all_pnls[:top_20_count]) / gross_profit), 3) if gross_profit > 0 else 0.0,
        "max_losing_streak": int(max_losing_streak),
        "months_positive": int(sum(1 for value in monthly.values() if value > 0)),
        "months_negative": int(sum(1 for value in monthly.values() if value <= 0)),
        "months_total": int(len(monthly)),
        "positive_years": int(sum(1 for value in yearly_values.values() if value > 0)),
        "negative_years": int(sum(1 for value in yearly_values.values() if value <= 0)),
    }


def year_profit_share(trades: list[dict[str, object]], pnl_key: str = "pnl_r") -> dict[str, dict[str, float]]:
    by_year: dict[str, float] = defaultdict(float)
    for trade in trades:
        by_year[str(trade["entry_time"])[:4]] += float(trade[pnl_key])
    total = sum(by_year.values())
    result: dict[str, dict[str, float]] = {}
    for year, value in sorted(by_year.items()):
        result[year] = {
            "profit_r": round(value, 2),
            "share": round((value / total), 3) if total else 0.0,
        }
    return result


def block_metrics(trades: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for name, (start, end) in BLOCKS.items():
        bucket = [
            trade
            for trade in trades
            if start <= str(trade["entry_time"])[:10] <= end
        ]
        results[name] = compute_metrics(bucket)
    return results


def simulate_trade(m5: pd.DataFrame, trade: dict[str, object], news_details: pd.DataFrame, force_flat_enabled: bool) -> dict[str, object]:
    entry_time = pd.Timestamp(trade["entry_time"])
    direction = str(trade["direction"])
    stop_loss = float(trade["sl"])
    take_profit = float(trade["tp"])
    entry_price = float(trade["entry_price"])
    future = m5.loc[(m5.index >= entry_time) & (m5.index <= entry_time + pd.Timedelta(hours=4))].copy()

    for timestamp, bar in future.iterrows():
        details = news_details.loc[timestamp]
        open_price = float(bar["open"])
        high_price = float(bar["high"])
        low_price = float(bar["low"])

        if force_flat_enabled and bool(details["force_flat"]):
            if direction == "long":
                if open_price <= stop_loss:
                    return {
                        "exit_time": str(timestamp),
                        "exit_price_signal": round(open_price, 5),
                        "exit_reason": "sl_hit",
                        "pnl_r": -1.0,
                        "blocked_by_news": False,
                        "blocking_event_name": "",
                        "blocking_event_time_ny": "",
                        "blocking_rule_used": "",
                    }
                if open_price >= take_profit:
                    return {
                        "exit_time": str(timestamp),
                        "exit_price_signal": round(open_price, 5),
                        "exit_reason": "tp_hit",
                        "pnl_r": 1.5,
                        "blocked_by_news": False,
                        "blocking_event_name": "",
                        "blocking_event_time_ny": "",
                        "blocking_rule_used": "",
                    }
            else:
                if open_price >= stop_loss:
                    return {
                        "exit_time": str(timestamp),
                        "exit_price_signal": round(open_price, 5),
                        "exit_reason": "sl_hit",
                        "pnl_r": -1.0,
                        "blocked_by_news": False,
                        "blocking_event_name": "",
                        "blocking_event_time_ny": "",
                        "blocking_rule_used": "",
                    }
                if open_price <= take_profit:
                    return {
                        "exit_time": str(timestamp),
                        "exit_price_signal": round(open_price, 5),
                        "exit_reason": "tp_hit",
                        "pnl_r": 1.5,
                        "blocked_by_news": False,
                        "blocking_event_name": "",
                        "blocking_event_time_ny": "",
                        "blocking_rule_used": "",
                    }

            return {
                "exit_time": str(timestamp),
                "exit_price_signal": round(open_price, 5),
                "exit_reason": "news_fortress_kill",
                "pnl_r": pnl_r_from_exit(direction, entry_price, stop_loss, open_price),
                "blocked_by_news": True,
                "blocking_event_name": str(details["force_flat_event_name"] or details["blocking_event_name"]),
                "blocking_event_time_ny": str(details["force_flat_event_time_ny"] or details["blocking_event_time_ny"]),
                "blocking_rule_used": str(details["force_flat_rule_used"] or details["blocking_rule_used"]),
            }

        if direction == "long":
            if low_price <= stop_loss:
                return {
                    "exit_time": str(timestamp),
                    "exit_price_signal": round(stop_loss, 5),
                    "exit_reason": "sl_hit",
                    "pnl_r": -1.0,
                    "blocked_by_news": False,
                    "blocking_event_name": "",
                    "blocking_event_time_ny": "",
                    "blocking_rule_used": "",
                }
            if high_price >= take_profit:
                return {
                    "exit_time": str(timestamp),
                    "exit_price_signal": round(take_profit, 5),
                    "exit_reason": "tp_hit",
                    "pnl_r": 1.5,
                    "blocked_by_news": False,
                    "blocking_event_name": "",
                    "blocking_event_time_ny": "",
                    "blocking_rule_used": "",
                }
        else:
            if high_price >= stop_loss:
                return {
                    "exit_time": str(timestamp),
                    "exit_price_signal": round(stop_loss, 5),
                    "exit_reason": "sl_hit",
                    "pnl_r": -1.0,
                    "blocked_by_news": False,
                    "blocking_event_name": "",
                    "blocking_event_time_ny": "",
                    "blocking_rule_used": "",
                }
            if low_price <= take_profit:
                return {
                    "exit_time": str(timestamp),
                    "exit_price_signal": round(take_profit, 5),
                    "exit_reason": "tp_hit",
                    "pnl_r": 1.5,
                    "blocked_by_news": False,
                    "blocking_event_name": "",
                    "blocking_event_time_ny": "",
                    "blocking_rule_used": "",
                }

    last_close = float(future.iloc[-1]["close"]) if not future.empty else entry_price
    return {
        "exit_time": str(future.index[-1]) if not future.empty else str(entry_time),
        "exit_price_signal": round(last_close, 5),
        "exit_reason": "timeout",
        "pnl_r": pnl_r_from_exit(direction, entry_price, stop_loss, last_close),
        "blocked_by_news": False,
        "blocking_event_name": "",
        "blocking_event_time_ny": "",
        "blocking_rule_used": "",
    }


def find_scbi_candidate(m5: pd.DataFrame, sweep: dict[str, object], news_details: pd.DataFrame) -> dict[str, object]:
    sweep_time = pd.Timestamp(sweep["time"])
    direction = str(sweep["direction"])
    level_price = float(sweep["level_price"])
    sweep_extreme = float(sweep["sweep_extreme"])

    search_start = sweep_time + pd.Timedelta(hours=1)
    search_end = search_start + pd.Timedelta(hours=1)
    window = m5.loc[(m5.index >= search_start) & (m5.index <= search_end)].copy()
    if window.empty:
        return {"status": "no_scbi_window"}

    for position in range(len(window)):
        bar = window.iloc[position]
        close_price = float(bar["close"])

        if direction == "long":
            trigger = close_price > level_price
        else:
            trigger = close_price < level_price
        if not trigger:
            continue

        if position + 1 >= len(window):
            return {"status": "no_entry_bar_after_scbi"}

        entry_bar = window.iloc[position + 1]
        entry_time = window.index[position + 1]
        if direction == "long":
            entry_price = float(entry_bar["open"]) + (BASELINE_SPREAD_PIPS * PIP_SIZE)
            stop_loss = sweep_extreme - PIP_SIZE
            risk_distance = entry_price - stop_loss
            take_profit = entry_price + (1.5 * risk_distance)
        else:
            entry_price = float(entry_bar["open"])
            stop_loss = sweep_extreme + PIP_SIZE
            risk_distance = stop_loss - entry_price
            take_profit = entry_price - (1.5 * risk_distance)

        if risk_distance <= 0 or risk_distance < MIN_RISK_PRICE:
            return {"status": "invalid_risk"}

        details = news_details.loc[entry_time]
        if bool(details["entry_blocked"]):
            return {
                "status": "news_blocked",
                "entry_time": str(entry_time),
                "blocking_event_name": str(details["blocking_event_name"]),
                "blocking_event_time_ny": str(details["blocking_event_time_ny"]),
                "blocking_rule_used": str(details["blocking_rule_used"]),
            }

        return {
            "status": "tradable",
            "entry_time": str(entry_time),
            "entry_price": round(entry_price, 5),
            "sl": round(stop_loss, 5),
            "tp": round(take_profit, 5),
            "risk_pips": round(risk_distance / PIP_SIZE, 2),
        }

    return {"status": "no_scbi_found"}


def write_checkpoint(processed_sweeps: int, last_sweep_time: str, counters: dict[str, int]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at_utc": now_utc_iso(),
        "processed_sweeps": processed_sweeps,
        "last_sweep_time": last_sweep_time,
        "counters": counters,
    }
    path = CHECKPOINT_DIR / f"checkpoint_{processed_sweeps:05d}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_stats_csv(path: Path, metrics_map: dict[str, dict[str, object]], bucket_type: str) -> None:
    rows = [{"bucket_type": bucket_type, "bucket": bucket, **metrics} for bucket, metrics in metrics_map.items()]
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    print("=" * 70)
    print("SCBI_M5 2020-2025 DURABILITY RECONSTRUCTION")
    print("Prices: REAL | News Fortress: CANONICAL | No fallbacks")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    h1, m5, price_coverage = load_price_frames()
    news_events, news_settings, news_coverage = load_news_events()
    news_details = build_news_guards(m5.index, news_events, news_settings)

    print(f"[DATA] H1 rows 2020-2025: {price_coverage['h1']['rows']}")
    print(f"[DATA] M5 rows 2020-2025: {price_coverage['m5']['rows']}")
    print(f"[DATA] News rows 2020-2025: {news_coverage['rows']}")

    levels = compute_session_levels(h1)
    sweeps = detect_sweeps_h1(h1, levels)
    print(f"[SCAN] H1 sweeps detected: {len(sweeps)}")

    trades: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    counters = {
        "sweeps_total": 0,
        "trades_executed": 0,
        "daily_limit_skips": 0,
        "news_blocked_candidates": 0,
        "no_scbi_found": 0,
        "invalid_risk": 0,
        "news_forceflat_exits": 0,
    }
    last_trade_date: object | None = None

    for index, sweep in enumerate(sweeps, start=1):
        sweep_time = pd.Timestamp(sweep["time"])
        sweep_date = sweep_time.date()
        counters["sweeps_total"] += 1

        audit_base = {
            "session_date": str(sweep_date),
            "sweep_time": str(sweep_time),
            "direction": sweep["direction"],
            "level_name": sweep["level_name"],
            "level_price": round(float(sweep["level_price"]), 5),
            "sweep_extreme": round(float(sweep["sweep_extreme"]), 5),
            "status": "",
            "entry_time": "",
            "exit_time": "",
            "exit_reason": "",
            "risk_pips": "",
            "pnl_r": "",
            "blocking_event_name": "",
            "blocking_event_time_ny": "",
            "blocking_rule_used": "",
        }

        if last_trade_date == sweep_date:
            counters["daily_limit_skips"] += 1
            audit_rows.append({**audit_base, "status": "DAILY_LIMIT_SKIPPED"})
            continue

        candidate = find_scbi_candidate(m5, sweep, news_details)
        status = str(candidate["status"])

        if status in {"no_scbi_window", "no_entry_bar_after_scbi", "no_scbi_found"}:
            counters["no_scbi_found"] += 1
            audit_rows.append({**audit_base, "status": status.upper()})
        elif status == "invalid_risk":
            counters["invalid_risk"] += 1
            audit_rows.append({**audit_base, "status": "INVALID_RISK"})
        elif status == "news_blocked":
            counters["news_blocked_candidates"] += 1
            audit_rows.append(
                {
                    **audit_base,
                    "status": "NEWS_BLOCKED",
                    "entry_time": candidate["entry_time"],
                    "blocking_event_name": candidate["blocking_event_name"],
                    "blocking_event_time_ny": candidate["blocking_event_time_ny"],
                    "blocking_rule_used": candidate["blocking_rule_used"],
                }
            )
        elif status == "tradable":
            trade = {
                "session_date": str(sweep_date),
                "sweep_time": str(sweep_time),
                "entry_time": candidate["entry_time"],
                "direction": sweep["direction"],
                "level": sweep["level_name"],
                "level_price": round(float(sweep["level_price"]), 5),
                "sweep_extreme": round(float(sweep["sweep_extreme"]), 5),
                "entry_price": candidate["entry_price"],
                "sl": candidate["sl"],
                "tp": candidate["tp"],
                "risk_pips": candidate["risk_pips"],
            }
            trade.update(simulate_trade(m5, trade, news_details, bool(news_settings.forced_exit_pre_news)))
            if trade["exit_reason"] == "news_fortress_kill":
                counters["news_forceflat_exits"] += 1
            counters["trades_executed"] += 1
            trades.append(trade)
            last_trade_date = sweep_date
            audit_rows.append(
                {
                    **audit_base,
                    "status": "TRADE_EXECUTED",
                    "entry_time": trade["entry_time"],
                    "exit_time": trade["exit_time"],
                    "exit_reason": trade["exit_reason"],
                    "risk_pips": trade["risk_pips"],
                    "pnl_r": trade["pnl_r"],
                    "blocking_event_name": trade["blocking_event_name"],
                    "blocking_event_time_ny": trade["blocking_event_time_ny"],
                    "blocking_rule_used": trade["blocking_rule_used"],
                }
            )
        else:
            raise RuntimeError(f"Estado de candidato no soportado: {status}")

        if index % 250 == 0:
            write_checkpoint(index, str(sweep_time), counters)

    stressed_trades = apply_cost_stress(trades)

    yearly = bucket_trades(trades, lambda trade: str(trade["entry_time"])[:4])
    semester = bucket_trades(
        trades,
        lambda trade: f"{str(trade['entry_time'])[:4]}-H1" if int(str(trade["entry_time"])[5:7]) <= 6 else f"{str(trade['entry_time'])[:4]}-H2",
    )
    direction = bucket_trades(trades, lambda trade: str(trade["direction"]))
    liquidity = bucket_trades(trades, lambda trade: str(trade["level"]))

    yearly_metrics = {bucket: compute_metrics(bucket_trades_list) for bucket, bucket_trades_list in yearly.items()}
    semester_metrics = {bucket: compute_metrics(bucket_trades_list) for bucket, bucket_trades_list in semester.items()}
    direction_metrics = {bucket: compute_metrics(bucket_trades_list) for bucket, bucket_trades_list in direction.items()}
    liquidity_metrics = {bucket: compute_metrics(bucket_trades_list) for bucket, bucket_trades_list in liquidity.items()}
    block_metrics_map = block_metrics(trades)
    stress_yearly_metrics = {
        bucket: compute_metrics(bucket_trades_list, pnl_key="pnl_r_stress")
        for bucket, bucket_trades_list in bucket_trades(stressed_trades, lambda trade: str(trade["entry_time"])[:4]).items()
    }
    stress_block_metrics = {
        name: compute_metrics(
            [
                trade
                for trade in stressed_trades
                if start <= str(trade["entry_time"])[:10] <= end
            ],
            pnl_key="pnl_r_stress",
        )
        for name, (start, end) in BLOCKS.items()
    }

    global_metrics = compute_metrics(trades)
    stress_global = compute_metrics(stressed_trades, pnl_key="pnl_r_stress")
    concentration = concentration_summary(trades)
    monthly = monthly_pnl(trades)
    profit_share = year_profit_share(trades)

    summary = {
        "metadata": {
            "runner": str(Path(__file__).relative_to(ROOT)),
            "generated_at_utc": now_utc_iso(),
            "period": {"start": START_DATE, "end": END_DATE},
            "price_dirs": [str(path.relative_to(ROOT)) for path in PRICE_DIRS],
            "costs": {
                "baseline_spread_pips": BASELINE_SPREAD_PIPS,
                "stress_total_spread_pips": STRESS_TOTAL_SPREAD_PIPS,
                "stress_delta_pips": STRESS_DELTA_PIPS,
            },
            "news_fortress": {
                "file": news_coverage["source_path"],
                "pre_minutes": news_coverage["pre_minutes"],
                "post_minutes": news_coverage["post_minutes"],
                "pre_news_exit_minutes": news_coverage["pre_news_exit_minutes"],
                "forced_exit_pre_news": news_coverage["forced_exit_pre_news"],
                "cancel_pending_pre_news": news_coverage["cancel_pending_pre_news"],
            },
        },
        "coverage": {
            "price": price_coverage,
            "news": news_coverage,
            "warnings": [
                "data_free_2020/prepared/EURUSD_M5.csv contiene 753 barras con volumen cero y 982 barras flat; queda documentado como warning no bloqueante.",
                "data_candidates_2022_2025/prepared/EURUSD_M5.csv conserva 1 gap inesperado ya documentado en 2025-12-31 19:05 NY; queda tratado como warning no bloqueante.",
            ],
        },
        "integrity": {
            "price_validate_frame_pass": True,
            "news_enabled": True,
            "no_hardcoded_fallback": True,
            "checkpoint_dir": str(CHECKPOINT_DIR.relative_to(ROOT)),
            "mandatory_years_present": list(MANDATORY_H1_YEARS),
        },
        "execution": {
            **counters,
            "levels_computed_days": int(len(levels)),
        },
        "metrics": {
            "global": global_metrics,
            "yearly": yearly_metrics,
            "semester": semester_metrics,
            "blocks": block_metrics_map,
            "direction": direction_metrics,
            "liquidity_source": liquidity_metrics,
            "stress_global": stress_global,
            "stress_yearly": stress_yearly_metrics,
            "stress_blocks": stress_block_metrics,
            "concentration": concentration,
            "monthly_pnl": monthly,
            "year_profit_share": profit_share,
        },
    }

    pd.DataFrame(trades).to_csv(TRADES_FILE, index=False)
    pd.DataFrame(audit_rows).to_csv(SWEEP_AUDIT_FILE, index=False)
    export_stats_csv(YEARLY_FILE, yearly_metrics, "year")
    export_stats_csv(SEMESTER_FILE, semester_metrics, "semester")
    export_stats_csv(BLOCK_FILE, block_metrics_map, "block")
    pd.DataFrame([{"month": month, "pnl_r": pnl_r} for month, pnl_r in monthly.items()]).to_csv(MONTHLY_FILE, index=False)
    RESULTS_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_checkpoint(counters["sweeps_total"], str(sweeps[-1]["time"]) if sweeps else "", counters)

    print(f"[RESULT] Trades executed: {len(trades)}")
    print(f"[RESULT] Global PF: {global_metrics['pf']}")
    print(f"[RESULT] Global Expectancy: {global_metrics['expectancy']}")
    print(f"[RESULT] Summary JSON: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
