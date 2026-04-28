from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PIP = 0.0001
NY_TZ = "America/New_York"
VALID_M3_STATUSES = {
    "M3_BID_ASK_CERTIFIED",
    "M3_BID_ASK_CERTIFIED_FULL",
    "M3_BID_ASK_CERTIFIED_WITH_DATA_QUALITY_MASK",
    "M3_CERTIFIED_WITH_WARNINGS",
}


class NativeM3UnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class Phase19RepairedConfig:
    start_time: str = "08:00"
    end_time: str = "11:00"
    hard_start_time: str = "07:00"
    hard_end_time: str = "20:00"
    forced_close_time: str = "20:00"
    max_trades_per_day: int = 1
    max_minutes_post_sweep: int = 30
    min_sweep_pips: float = 0.5
    tp_r: float = 2.5
    fractal_n_h1: int = 2
    fractal_n_m3: int = 2
    news_guard_minutes: int = 30


def load_manifest(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def require_native_m3(manifest: dict, period: str) -> tuple[Path, Path]:
    data = manifest.get(period, {})
    bid = data.get("m3_bid")
    ask = data.get("m3_ask")
    if isinstance(bid, dict):
        if bid.get("certification_status") not in VALID_M3_STATUSES:
            raise NativeM3UnavailableError(
                f"M3_NATIVO_AUSENTE: m3_bid certification_status={bid.get('certification_status')}"
            )
        bid = bid.get("path")
    if isinstance(ask, dict):
        if ask.get("certification_status") not in VALID_M3_STATUSES:
            raise NativeM3UnavailableError(
                f"M3_NATIVO_AUSENTE: m3_ask certification_status={ask.get('certification_status')}"
            )
        ask = ask.get("path")
    if not bid or not ask:
        raise NativeM3UnavailableError(
            f"M3_NATIVO_AUSENTE: manifest period {period} must contain m3_bid and m3_ask"
        )
    bid_path = Path(bid)
    ask_path = Path(ask)
    if not bid_path.exists() or not ask_path.exists():
        raise NativeM3UnavailableError(
            f"M3_NATIVO_AUSENTE: missing files m3_bid={bid_path.exists()} m3_ask={ask_path.exists()}"
        )
    return bid_path, ask_path


def require_data_quality_mask(manifest: dict, period: str) -> Path | None:
    data = manifest.get(period, {})
    m3_entries = [data.get("m3_bid"), data.get("m3_ask"), data.get("m3_spread")]
    mask_required = any(isinstance(entry, dict) and entry.get("requires_data_quality_mask") for entry in m3_entries)
    if not mask_required:
        return None
    mask = data.get("m3_data_quality_mask")
    if not isinstance(mask, dict):
        raise NativeM3UnavailableError("DATA_QUALITY_MASK_REQUIRED_BUT_MISSING")
    if mask.get("certification_status") != "DATA_QUALITY_MASK_FAIL_CLOSED":
        raise NativeM3UnavailableError("DATA_QUALITY_MASK_NOT_FAIL_CLOSED")
    if mask.get("enforced_for_phase19_repaired") is not True:
        raise NativeM3UnavailableError("DATA_QUALITY_MASK_NOT_ENFORCED")
    path = Path(mask.get("path", ""))
    if not path.exists():
        raise NativeM3UnavailableError("DATA_QUALITY_MASK_PATH_MISSING")
    return path


def enforce_phase19_data_quality_mask(df_m3: pd.DataFrame, mask_path: Path | None) -> pd.DataFrame:
    if mask_path is None:
        return df_m3
    mask = pd.read_csv(mask_path)
    if "date_ny" not in mask.columns or "allow_phase19_repaired" not in mask.columns:
        raise NativeM3UnavailableError("DATA_QUALITY_MASK_SCHEMA_INVALID")
    allowed_dates = set(mask[mask["allow_phase19_repaired"].astype(bool)]["date_ny"].astype(str))
    df = df_m3.copy()
    if "timestamp_ny" not in df.columns:
        df["timestamp_ny"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY_TZ)
    date_ny = df["timestamp_ny"].dt.date.astype(str)
    return df[date_ny.isin(allowed_dates)].reset_index(drop=True)


def _load_ohlc_pair(bid_path: Path, ask_path: Path) -> pd.DataFrame:
    bid = pd.read_csv(bid_path)
    ask = pd.read_csv(ask_path)
    bid["timestamp"] = pd.to_datetime(bid["timestamp"], utc=True)
    ask["timestamp"] = pd.to_datetime(ask["timestamp"], utc=True)
    df = pd.merge(bid, ask, on="timestamp", suffixes=("_bid", "_ask"))
    df["timestamp_ny"] = df["timestamp"].dt.tz_convert(NY_TZ)
    return df.sort_values("timestamp").reset_index(drop=True)


def load_native_m3(manifest_path: str | Path, period: str) -> pd.DataFrame:
    manifest = load_manifest(manifest_path)
    bid_path, ask_path = require_native_m3(manifest, period)
    mask_path = require_data_quality_mask(manifest, period)
    df = _load_ohlc_pair(bid_path, ask_path)
    return enforce_phase19_data_quality_mask(df, mask_path)


def load_h1_bid(manifest_path: str | Path, period: str) -> pd.DataFrame:
    manifest = load_manifest(manifest_path)
    path = Path(manifest[period]["h1_bid"])
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.rename(columns={c: f"{c}_bid" for c in ["open", "high", "low", "close"] if c in df.columns})
    df["timestamp_ny"] = df["timestamp"].dt.tz_convert(NY_TZ)
    return df.sort_values("timestamp").reset_index(drop=True)


def load_news(manifest_path: str | Path, period: str) -> pd.DataFrame:
    manifest = load_manifest(manifest_path)
    path = Path(manifest[period]["news"])
    df = pd.read_csv(path)
    source = "timestamp_utc" if "timestamp_utc" in df.columns else "timestamp"
    df["timestamp"] = pd.to_datetime(df[source], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def get_confirmed_fractals(df: pd.DataFrame, n: int) -> tuple[np.ndarray, np.ndarray]:
    highs = df["high_bid"].to_numpy()
    lows = df["low_bid"].to_numpy()
    f_high = np.full(len(df), np.nan)
    f_low = np.full(len(df), np.nan)
    for i in range(2 * n, len(df)):
        center = i - n
        left = center - n
        right = center + n + 1
        if all(highs[center] > highs[j] for j in range(left, right) if j != center):
            f_high[i] = highs[center]
        if all(lows[center] < lows[j] for j in range(left, right) if j != center):
            f_low[i] = lows[center]
    return f_high, f_low


def detect_h1_fractal_sweeps(df_h1: pd.DataFrame, config: Phase19RepairedConfig) -> pd.DataFrame:
    df = df_h1.copy().reset_index(drop=True)
    fh, fl = get_confirmed_fractals(df, config.fractal_n_h1)
    df["last_fh"] = pd.Series(fh).ffill()
    df["last_fl"] = pd.Series(fl).ffill()
    rows = []
    for i, row in df.iterrows():
        if i == 0:
            continue
        high = row["high_bid"]
        low = row["low_bid"]
        close = row["close_bid"]
        if pd.notna(row["last_fh"]) and high > row["last_fh"] and close < row["last_fh"]:
            depth = (high - row["last_fh"]) / PIP
            if depth >= config.min_sweep_pips:
                rows.append(
                    {
                        "sweep_id": len(rows),
                        "timestamp_ny": row["timestamp_ny"],
                        "type": "BEARISH_SWEEP",
                        "level_type": "h1_fractal_high",
                        "level_price": row["last_fh"],
                        "peak_price": high,
                        "depth_pips": depth,
                    }
                )
        if pd.notna(row["last_fl"]) and low < row["last_fl"] and close > row["last_fl"]:
            depth = (row["last_fl"] - low) / PIP
            if depth >= config.min_sweep_pips:
                rows.append(
                    {
                        "sweep_id": len(rows),
                        "timestamp_ny": row["timestamp_ny"],
                        "type": "BULLISH_SWEEP",
                        "level_type": "h1_fractal_low",
                        "level_price": row["last_fl"],
                        "peak_price": low,
                        "depth_pips": depth,
                    }
                )
    return pd.DataFrame(rows)


def detect_first_m3_choch(
    df_m3: pd.DataFrame, sweeps: pd.DataFrame, config: Phase19RepairedConfig
) -> pd.DataFrame:
    if sweeps.empty:
        return pd.DataFrame()
    df = df_m3.copy().reset_index(drop=True)
    fh, fl = get_confirmed_fractals(df, config.fractal_n_m3)
    df["last_fh"] = pd.Series(fh).ffill()
    df["last_fl"] = pd.Series(fl).ffill()
    rows = []
    for _, sweep in sweeps.iterrows():
        sweep_time = pd.Timestamp(sweep["timestamp_ny"])
        window_end = sweep_time + pd.Timedelta(minutes=config.max_minutes_post_sweep)
        window = df[(df["timestamp_ny"] >= sweep_time) & (df["timestamp_ny"] <= window_end)]
        if window.empty:
            continue
        for idx, bar in window.iterrows():
            entry_idx = int(idx) + 1
            if entry_idx >= len(df):
                break
            if sweep["type"] == "BULLISH_SWEEP" and pd.notna(bar["last_fh"]):
                if bar["close_bid"] > bar["last_fh"]:
                    next_bar = df.iloc[entry_idx]
                    rows.append(
                        {
                            "signal_id": len(rows),
                            "sweep_id": int(sweep["sweep_id"]),
                            "sweep_time": sweep_time,
                            "choch_time": bar["timestamp_ny"],
                            "entry_time": next_bar["timestamp_ny"],
                            "signal_bar_index": int(idx),
                            "entry_bar_index": entry_idx,
                            "direction": "LONG",
                            "entry_price": float(next_bar["open_ask"]),
                            "sl_price": float(sweep["peak_price"]),
                            "sweep_level_type": sweep["level_type"],
                            "sweep_peak_price": float(sweep["peak_price"]),
                        }
                    )
                    break
            if sweep["type"] == "BEARISH_SWEEP" and pd.notna(bar["last_fl"]):
                if bar["close_bid"] < bar["last_fl"]:
                    next_bar = df.iloc[entry_idx]
                    rows.append(
                        {
                            "signal_id": len(rows),
                            "sweep_id": int(sweep["sweep_id"]),
                            "sweep_time": sweep_time,
                            "choch_time": bar["timestamp_ny"],
                            "entry_time": next_bar["timestamp_ny"],
                            "signal_bar_index": int(idx),
                            "entry_bar_index": entry_idx,
                            "direction": "SHORT",
                            "entry_price": float(next_bar["open_bid"]),
                            "sl_price": float(sweep["peak_price"]),
                            "sweep_level_type": sweep["level_type"],
                            "sweep_peak_price": float(sweep["peak_price"]),
                        }
                    )
                    break
    return pd.DataFrame(rows)


def _time_value(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%H:%M")


def _in_time_range(ts: pd.Timestamp, start: str, end: str) -> bool:
    value = _time_value(ts)
    return start <= value < end


def is_news_blocked(entry_time: pd.Timestamp, news: pd.DataFrame, config: Phase19RepairedConfig) -> bool:
    if news is None or news.empty:
        return False
    ts = pd.Timestamp(entry_time).tz_convert("UTC")
    relevant = news
    if "currency" in relevant.columns:
        relevant = relevant[relevant["currency"].isin(["USD", "EUR"])]
    if "impact_level" in relevant.columns:
        relevant = relevant[relevant["impact_level"].astype(str).str.upper().isin(["HIGH", "MEDIUM"])]
    if relevant.empty:
        return False
    delta = pd.Timedelta(minutes=config.news_guard_minutes)
    times = pd.to_datetime(relevant["timestamp"], utc=True)
    return bool(((times >= ts - delta) & (times <= ts + delta)).any())


def simulate_repaired_backtest(
    df_m3: pd.DataFrame,
    signals: pd.DataFrame,
    news: pd.DataFrame,
    config: Phase19RepairedConfig,
    extra_cost_pips: float = 0.0,
) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    df = df_m3.copy().reset_index(drop=True)
    signals = signals.sort_values(["entry_time", "signal_id"]).reset_index(drop=True)
    trades = []
    trades_by_day: dict[object, int] = {}
    last_exit_time = None
    used_levels_by_day: dict[object, set[str]] = {}

    for _, sig in signals.iterrows():
        entry_time = pd.Timestamp(sig["entry_time"])
        day = entry_time.date()
        if not _in_time_range(entry_time, config.hard_start_time, config.hard_end_time):
            continue
        if not _in_time_range(entry_time, config.start_time, config.end_time):
            continue
        if trades_by_day.get(day, 0) >= config.max_trades_per_day:
            continue
        if last_exit_time is not None and entry_time <= last_exit_time:
            continue
        level_key = f"{sig['sweep_level_type']}|{sig['sweep_peak_price']:.5f}"
        if level_key in used_levels_by_day.setdefault(day, set()):
            continue
        if is_news_blocked(entry_time, news, config):
            continue

        direction = sig["direction"]
        entry_price = float(sig["entry_price"])
        if direction == "LONG":
            entry_price += extra_cost_pips * PIP
            risk = entry_price - float(sig["sl_price"])
            tp_price = entry_price + risk * config.tp_r
        else:
            entry_price -= extra_cost_pips * PIP
            risk = float(sig["sl_price"]) - entry_price
            tp_price = entry_price - risk * config.tp_r
        if risk <= 0:
            continue

        status = "NO_EXIT"
        exit_price = np.nan
        exit_time = pd.NaT
        same_bar = False
        entry_bar_index = int(sig["entry_bar_index"])
        for j in range(entry_bar_index + 1, len(df)):
            bar = df.iloc[j]
            bar_time = pd.Timestamp(bar["timestamp_ny"])
            if bar_time.date() != day or _time_value(bar_time) >= config.forced_close_time:
                status = "FORCED_CLOSE_2000"
                exit_time = bar_time
                exit_price = float(bar["close_bid"] if direction == "LONG" else bar["close_ask"])
                break
            if direction == "LONG":
                sl_hit = float(bar["low_bid"]) <= float(sig["sl_price"])
                tp_hit = float(bar["high_bid"]) >= tp_price
                if sl_hit:
                    status = "SL"
                    exit_time = bar_time
                    exit_price = float(sig["sl_price"])
                    break
                if tp_hit:
                    status = "TP"
                    exit_time = bar_time
                    exit_price = tp_price
                    break
            else:
                sl_hit = float(bar["high_ask"]) >= float(sig["sl_price"])
                tp_hit = float(bar["low_ask"]) <= tp_price
                if sl_hit:
                    status = "SL"
                    exit_time = bar_time
                    exit_price = float(sig["sl_price"])
                    break
                if tp_hit:
                    status = "TP"
                    exit_time = bar_time
                    exit_price = tp_price
                    break
        if status == "NO_EXIT":
            continue
        pnl = (exit_price - entry_price) if direction == "LONG" else (entry_price - exit_price)
        r_pnl = pnl / risk
        trades.append(
            {
                "trade_id": len(trades),
                "entry_time": entry_time,
                "exit_time": exit_time,
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "sl_price": float(sig["sl_price"]),
                "tp_price": tp_price,
                "risk_pips": risk / PIP,
                "status": status,
                "r_pnl": r_pnl,
                "same_bar": same_bar,
                "signal_id": int(sig["signal_id"]),
                "sweep_id": int(sig["sweep_id"]),
                "sweep_level_type": sig["sweep_level_type"],
                "sweep_peak_price": float(sig["sweep_peak_price"]),
            }
        )
        trades_by_day[day] = trades_by_day.get(day, 0) + 1
        used_levels_by_day[day].add(level_key)
        last_exit_time = pd.Timestamp(exit_time)
    return pd.DataFrame(trades)


def compute_metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "sample": 0,
            "pf": 0.0,
            "expectancy_R": 0.0,
            "max_drawdown_R": 0.0,
            "max_loss_streak": 0,
            "trades_month": 0.0,
            "win_rate": 0.0,
            "out_of_hours": 0,
            "forced_close": 0,
            "same_bar": 0,
        }
    df = trades.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["r_pnl"] = pd.to_numeric(df["r_pnl"])
    wins = df[df["r_pnl"] > 0]["r_pnl"].sum()
    losses = abs(df[df["r_pnl"] < 0]["r_pnl"].sum())
    eq = df["r_pnl"].cumsum()
    dd = eq - eq.cummax()
    streak = 0
    max_streak = 0
    for value in df["r_pnl"]:
        if value < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    first = df["entry_time"].min()
    last = df["entry_time"].max()
    months = max(1, (last.year - first.year) * 12 + (last.month - first.month) + 1)
    hour = df["entry_time"].dt.hour + df["entry_time"].dt.minute / 60.0
    return {
        "sample": int(len(df)),
        "pf": round(float(wins / losses), 6) if losses else 0.0,
        "expectancy_R": round(float(df["r_pnl"].mean()), 6),
        "max_drawdown_R": round(float(dd.min()), 6),
        "max_loss_streak": int(max_streak),
        "trades_month": round(float(len(df) / months), 6),
        "win_rate": round(float((df["r_pnl"] > 0).mean()), 6),
        "out_of_hours": int(((hour < 7.0) | (hour >= 20.0)).sum()),
        "forced_close": int((df["status"] == "FORCED_CLOSE_2000").sum()),
        "same_bar": int(df["same_bar"].sum()),
    }


def run_repaired_screening(manifest_path: str | Path, period: str, config: Phase19RepairedConfig) -> pd.DataFrame:
    manifest = load_manifest(manifest_path)
    require_native_m3(manifest, period)
    df_h1 = load_h1_bid(manifest_path, period)
    df_m3 = load_native_m3(manifest_path, period)
    news = load_news(manifest_path, period)
    sweeps = detect_h1_fractal_sweeps(df_h1, config)
    signals = detect_first_m3_choch(df_m3, sweeps, config)
    trades = simulate_repaired_backtest(df_m3, signals, news, config)
    return trades
