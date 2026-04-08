#!/usr/bin/env python
"""
Backtester multi-timeframe para FX con sincronizacion M5 / M15 / H1.

Uso rapido:
    python fx_multi_timeframe_backtester.py prepare-data --download-missing
    python fx_multi_timeframe_backtester.py run
    python fx_multi_timeframe_backtester.py optimize --max-combinations 16
"""

from __future__ import annotations

import argparse
import json
import itertools
import lzma
import math
import os
import struct
import textwrap
import time as pytime
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

DEFAULT_MPLCONFIGDIR = Path.cwd() / ".mplconfig"
if "MPLCONFIGDIR" not in os.environ:
    DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(DEFAULT_MPLCONFIGDIR)

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


NY_TZ = "America/New_York"
UTC = timezone.utc
ONE_MINUTE = timedelta(minutes=1)


PAIR_META = {
    "EURUSD": {"base": "EUR", "quote": "USD", "pip_size": 0.0001, "price_divisor": 100000},
    "GBPUSD": {"base": "GBP", "quote": "USD", "pip_size": 0.0001, "price_divisor": 100000},
    "USDJPY": {"base": "USD", "quote": "JPY", "pip_size": 0.01, "price_divisor": 1000},
    "AUDUSD": {"base": "AUD", "quote": "USD", "pip_size": 0.0001, "price_divisor": 100000},
    "USDCAD": {"base": "USD", "quote": "CAD", "pip_size": 0.0001, "price_divisor": 100000},
    "USDCHF": {"base": "USD", "quote": "CHF", "pip_size": 0.0001, "price_divisor": 100000},
    "GBPJPY": {"base": "GBP", "quote": "JPY", "pip_size": 0.01, "price_divisor": 1000},
}

DEFAULT_PAIRS = tuple(PAIR_META.keys())
DEFAULT_CORRELATION_GROUPS = (
    ("EURUSD", "GBPUSD", "AUDUSD"),
    ("USDJPY", "USDCAD", "USDCHF"),
    ("GBPUSD", "GBPJPY"),
)
DEFAULT_SPREAD_PIPS = {
    "EURUSD": 0.2,
    "GBPUSD": 0.4,
    "USDJPY": 0.3,
    "AUDUSD": 0.4,
    "USDCAD": 0.5,
    "USDCHF": 0.5,
    "GBPJPY": 0.9,
}

COUNTRY_TO_CURRENCY = {
    "United States": "USD",
    "Euro Area": "EUR",
    "Japan": "JPY",
    "Australia": "AUD",
    "Canada": "CAD",
    "Switzerland": "CHF",
    "United Kingdom": "GBP",
}
TE_COUNTRY_BY_CURRENCY = {
    "USD": "united states",
    "EUR": "euro area",
    "JPY": "japan",
    "AUD": "australia",
    "CAD": "canada",
    "CHF": "switzerland",
    "GBP": "united kingdom",
}
DEFAULT_HARD_NEWS_KEYWORDS = (
    "non farm payroll",
    "non-farm payroll",
    "nfp",
    "cpi",
    "inflation rate",
    "core inflation",
    "consumer price index",
    "interest rate decision",
    "rate decision",
    "fomc",
    "federal funds rate",
    "fed chair",
    "powell",
    "ecb rate",
    "boe rate",
    "boj rate",
    "monetary policy",
    "central bank",
    "gdp",
    "employment change",
    "unemployment rate",
    "retail sales",
    "pmi",
)


@dataclass(frozen=True)
class BrokerConfig:
    initial_capital: float = 100_000.0
    risk_fraction: float = 0.01
    risk_budget_mode: str = "initial_capital"
    commission_rate: float = 0.00002
    slippage_pips: float = 0.2
    use_spread_model: bool = True
    max_leverage: float = 20.0
    lot_step: int = 1000
    session_start: str = "11:00"
    session_end: str = "18:45"

    @property
    def session_start_time(self) -> time:
        return time.fromisoformat(self.session_start)

    @property
    def session_end_time(self) -> time:
        return time.fromisoformat(self.session_end)


@dataclass(frozen=True)
class StrategyParameters:
    strategy_family: str = "trend_pullback"
    h1_ema_fast: int = 50
    h1_ema_slow: int = 200
    h1_adx_period: int = 14
    h1_adx_threshold: float = 18.0
    m15_ema_period: int = 20
    m15_rsi_period: int = 7
    m15_rsi_pullback_long: float = 45.0
    m15_rsi_pullback_short: float = 55.0
    m15_atr_period: int = 14
    m15_max_prev_range_atr: float = 1.5
    m5_rsi_period: int = 7
    m5_atr_period: int = 14
    m5_trigger_rsi_midpoint: float = 50.0
    stop_atr_multiple: float = 1.2
    take_profit_rr: float = 1.8
    mr_h1_adx_max: float = 28.0
    mr_m15_extension_atr: float = 0.8
    mr_m15_rsi_long: float = 35.0
    mr_m15_rsi_short: float = 65.0
    mr_m5_reclaim_rsi_long: float = 40.0
    mr_m5_reclaim_rsi_short: float = 60.0
    mr_target_atr_buffer: float = 0.15
    asr_h1_adx_max: float = 24.0
    asr_h1_distance_atr_max: float = 1.2
    asr_m15_extension_atr: float = 0.45
    asr_m15_rsi_long: float = 38.0
    asr_m15_rsi_short: float = 62.0
    asr_m5_reclaim_rsi_long: float = 42.0
    asr_m5_reclaim_rsi_short: float = 58.0
    asr_target_atr_buffer: float = 0.05
    asr_take_profit_rr_cap: float = 0.9
    asr_max_hold_bars: int = 9
    asr_entry_end: str = "15:30"
    asr_context_min_samples_soft: int = 3
    asr_context_min_samples_hard: int = 6
    asr_context_min_win_rate_soft: float = 0.42
    asr_context_min_win_rate_hard: float = 0.50
    asr_context_min_expectancy_soft: float = -0.05
    asr_context_min_expectancy_hard: float = 0.05
    csr_entry_start: str = "12:30"
    csr_entry_end: str = "15:30"
    csr_h1_adx_max: float = 20.0
    csr_h1_distance_atr_max: float = 0.8
    csr_h1_ema_spread_atr_max: float = 0.6
    csr_m15_extension_atr: float = 0.55
    csr_m15_rsi_long: float = 34.0
    csr_m15_rsi_short: float = 66.0
    csr_m5_reclaim_rsi_long: float = 42.0
    csr_m5_reclaim_rsi_short: float = 58.0
    csr_reversal_body_atr_min: float = 0.15
    csr_target_atr_buffer: float = 0.0
    csr_take_profit_rr_cap: float = 0.6
    csr_max_hold_bars: int = 5
    str_entry_start: str = "11:30"
    str_entry_end: str = "15:30"
    str_h1_adx_min: float = 18.0
    str_h1_distance_atr_max: float = 1.2
    str_m15_pullback_atr: float = 0.15
    str_m15_rsi_long_min: float = 48.0
    str_m15_rsi_short_max: float = 52.0
    str_m5_reclaim_rsi_long: float = 52.0
    str_m5_reclaim_rsi_short: float = 48.0
    str_take_profit_rr_cap: float = 0.8
    str_max_hold_bars: int = 8
    pb_core_entry_start: str = "12:00"
    pb_core_entry_end: str = "14:15"
    pb_breakout_entry_start: str = "13:00"
    pb_breakout_entry_end: str = "15:30"
    pb_late_entry_start: str = "14:15"
    pb_late_entry_end: str = "15:30"
    pb_balanced_adx_max: float = 20.0
    pb_active_adx_min: float = 24.0
    pb_h1_distance_atr_max: float = 0.9
    pb_core_extension_atr: float = 0.4
    pb_core_rsi_long: float = 35.0
    pb_core_rsi_short: float = 65.0
    pb_core_reclaim_rsi_long: float = 40.0
    pb_core_reclaim_rsi_short: float = 60.0
    pb_breakout_compression_atr_max: float = 0.55
    pb_breakout_trigger_range_atr_min: float = 0.6
    pb_breakout_rsi_long: float = 55.0
    pb_breakout_rsi_short: float = 45.0
    pb_late_reclaim_rsi_long: float = 54.0
    pb_late_reclaim_rsi_short: float = 46.0
    pb_take_profit_rr_cap: float = 0.8
    pb_max_hold_bars: int = 6
    pb_enable_core_reversion: bool = True
    pb_enable_late_continuation: bool = True
    pb_enable_compression_breakout: bool = True
    context_whitelist_weekdays: str = ""
    context_whitelist_times: str = ""
    context_whitelist_regimes: str = ""
    context_whitelist_extensions: str = ""
    context_whitelist_directions: str = ""


@dataclass(frozen=True)
class ResearchGoals:
    target_trades_per_month: float = 25.0
    min_trades_per_month: float = 12.0
    target_win_rate_pct: float = 55.0
    min_win_rate_pct: float = 48.0
    target_return_pct: float = 20.0
    max_drawdown_pct: float = 15.0
    min_profit_factor: float = 1.15
    max_consecutive_losses: int = 8


@dataclass(frozen=True)
class RunConfig:
    start: str = "2020-01-01"
    end: str = "2025-12-31"
    pairs: tuple[str, ...] = DEFAULT_PAIRS
    data_dir: Path = Path("data")
    report_dir: Path = Path("reports")
    correlation_groups: tuple[tuple[str, ...], ...] = DEFAULT_CORRELATION_GROUPS
    source: str = "auto"
    download_missing: bool = False
    force_download: bool = False
    synthetic: bool = False
    strict_data_quality: bool = False
    news_source: str = "none"
    news_file: Path | None = None
    news_api_key: str = "guest:guest"
    news_timezone: str = "UTC"
    news_min_importance: int = 3
    news_no_entry_pre_minutes: int = 45
    news_no_entry_post_minutes: int = 30
    news_flatten_minutes_before: int = 15
    hard_news_veto_enabled: bool = True
    news_hard_no_entry_pre_minutes: int = 90
    news_hard_no_entry_post_minutes: int = 60
    news_hard_flatten_minutes_before: int = 30
    news_hard_keywords: tuple[str, ...] = DEFAULT_HARD_NEWS_KEYWORDS
    shock_guard_enabled: bool = True
    shock_no_entry_atr_multiple: float = 2.5
    shock_flatten_atr_multiple: float = 3.0
    shock_cooldown_minutes: int = 30


@dataclass
class PendingEntry:
    pair: str
    direction: str
    signal_pos: int
    execute_pos: int
    planned_entry_time: pd.Timestamp
    signal_atr: float
    stop_distance: float | None = None
    take_profit_price: float | None = None
    take_profit_rr_cap: float | None = None
    max_hold_bars: int | None = None
    context_key: str | None = None
    setup_tag: str | None = None
    context_trades_before: int = 0
    context_win_rate_before: float = 0.0
    context_expectancy_before: float = 0.0


@dataclass
class Position:
    pair: str
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    take_profit_price: float
    units: float
    entry_commission_usd: float
    initial_risk_usd: float
    signal_pos: int
    opened_on_pos: int
    max_hold_bars: int | None = None
    context_key: str | None = None
    setup_tag: str | None = None
    context_trades_before: int = 0
    context_win_rate_before: float = 0.0
    context_expectancy_before: float = 0.0


@dataclass
class PairDataView:
    pair: str
    index: pd.DatetimeIndex
    arrays: dict[str, np.ndarray]
    valid_mask: np.ndarray
    prev_bar_pos: np.ndarray
    next_bar_pos: np.ndarray
    asof_pos: np.ndarray


@dataclass
class BacktestResult:
    parameters: dict[str, Any]
    broker: dict[str, Any]
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    data_quality: pd.DataFrame
    news_events: pd.DataFrame
    context_summary: pd.DataFrame
    setup_summary: pd.DataFrame
    pair_summary: pd.DataFrame
    yearly_summary: pd.DataFrame
    monthly_summary: pd.DataFrame
    portfolio_summary: dict[str, Any]
    robustness_score: float


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result.fillna(50.0)


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_series = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0.0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_series
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_series
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)) * 100
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df.resample(rule, label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )


def month_starts_between(start_date: date, end_date: date) -> list[date]:
    months: list[date] = []
    current = date(start_date.year, start_date.month, 1)
    limit = date(end_date.year, end_date.month, 1)
    while current <= limit:
        months.append(current)
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return months


def daterange(start_date: date, end_date: date) -> Iterable[date]:
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def ensure_ohlcv_schema(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    first_column = df.columns[0] if len(df.columns) else None
    for column in df.columns:
        lower = str(column).strip().lower()
        if lower in {"datetime", "timestamp", "date", "time"}:
            rename_map[column] = "datetime"
        elif first_column == column and lower.startswith("unnamed"):
            rename_map[column] = "datetime"
        elif lower in {"open", "o"}:
            rename_map[column] = "open"
        elif lower in {"high", "h"}:
            rename_map[column] = "high"
        elif lower in {"low", "l"}:
            rename_map[column] = "low"
        elif lower in {"close", "c"}:
            rename_map[column] = "close"
        elif lower in {"volume", "v"}:
            rename_map[column] = "volume"

    normalized = df.rename(columns=rename_map).copy()
    if "datetime" not in normalized.columns:
        raise ValueError("No se encontro una columna de fecha/hora en el dataset.")

    normalized["datetime"] = pd.to_datetime(normalized["datetime"], utc=True, errors="coerce")
    normalized = normalized.dropna(subset=["datetime"])

    for column in ("open", "high", "low", "close"):
        if column not in normalized.columns:
            raise ValueError(f"Falta la columna obligatoria '{column}'.")
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    if "volume" not in normalized.columns:
        normalized["volume"] = 0.0
    normalized["volume"] = pd.to_numeric(normalized["volume"], errors="coerce").fillna(0.0)

    normalized = normalized.dropna(subset=["open", "high", "low", "close"])
    normalized = normalized.set_index("datetime").sort_index()
    return normalized[["open", "high", "low", "close", "volume"]]


def resolve_local_pair_path(pair: str, data_dir: Path) -> Path | None:
    candidates = [
        data_dir / f"{pair}_M5.csv",
        data_dir / f"{pair}_M5.csv.gz",
        data_dir / f"{pair}.csv",
        data_dir / f"{pair}.csv.gz",
        data_dir / f"{pair}.parquet",
        data_dir / pair / "M5.csv",
        data_dir / pair / "M5.csv.gz",
        data_dir / pair / f"{pair}_M5.csv",
        data_dir / pair / f"{pair}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_local_m5(pair: str, data_dir: Path) -> pd.DataFrame:
    path = resolve_local_pair_path(pair, data_dir)
    if path is None:
        raise FileNotFoundError(f"No se encontro un archivo local M5 para {pair} en {data_dir}.")

    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    return ensure_ohlcv_schema(frame)


class DukascopyDownloader:
    RECORD_STRUCT = struct.Struct(">IIIIIf")

    def __init__(self, cache_dir: Path, user_agent: str = "Mozilla/5.0"):
        self.cache_dir = cache_dir
        self.user_agent = user_agent
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = 5
        self.retry_base_seconds = 1.5

    def load_or_download_m5(
        self,
        pair: str,
        start_date: date,
        end_date: date,
        force_download: bool = False,
        verbose: bool = True,
    ) -> pd.DataFrame:
        pair_dir = self.cache_dir / pair
        pair_dir.mkdir(parents=True, exist_ok=True)
        monthly_frames: list[pd.DataFrame] = []

        for month_start in month_starts_between(start_date, end_date):
            cache_path = pair_dir / f"{pair}_{month_start:%Y_%m}_M5.csv.gz"
            if cache_path.exists() and not force_download:
                monthly_frames.append(ensure_ohlcv_schema(pd.read_csv(cache_path)))
                continue

            month_end = self._month_end(month_start)
            period_end = min(month_end, end_date)
            rows: list[pd.DataFrame] = []

            if verbose:
                print(f"[data] Descargando {pair} {month_start:%Y-%m} desde Dukascopy...")

            for current_day in daterange(month_start, period_end):
                day_frame = self._download_day(pair, current_day)
                if day_frame is not None and not day_frame.empty:
                    rows.append(day_frame)

            if not rows:
                raise FileNotFoundError(
                    f"No se pudieron descargar candles M1 para {pair} en {month_start:%Y-%m}."
                )

            month_frame = pd.concat(rows).sort_index()
            month_m5 = resample_ohlcv(month_frame, "5min")
            month_m5.to_csv(cache_path)
            monthly_frames.append(month_m5)

        combined = pd.concat(monthly_frames).sort_index()
        start_ts = pd.Timestamp(start_date, tz=UTC)
        end_ts = pd.Timestamp(end_date + timedelta(days=1), tz=UTC)
        return combined[(combined.index >= start_ts) & (combined.index < end_ts)]

    def _build_url(self, pair: str, current_day: date) -> str:
        month_zero_based = current_day.month - 1
        return (
            f"https://datafeed.dukascopy.com/datafeed/{pair}/"
            f"{current_day.year}/{month_zero_based:02d}/{current_day.day:02d}/"
            "BID_candles_min_1.bi5"
        )

    def _download_day(self, pair: str, current_day: date) -> pd.DataFrame | None:
        url = self._build_url(pair, current_day)
        payload = None

        for attempt in range(1, self.max_retries + 1):
            request = Request(url, headers={"User-Agent": self.user_agent})
            try:
                with urlopen(request, timeout=45) as response:
                    payload = response.read()
                break
            except HTTPError as exc:
                if exc.code == 404:
                    return None
                if exc.code not in {429, 500, 502, 503, 504} or attempt == self.max_retries:
                    raise
            except URLError:
                if attempt == self.max_retries:
                    raise

            pytime.sleep(self.retry_base_seconds * attempt)

        if not payload:
            return None

        try:
            decompressed = lzma.decompress(payload)
        except lzma.LZMAError:
            return None

        divisor = PAIR_META[pair]["price_divisor"]
        day_start = datetime.combine(current_day, time(0, 0), tzinfo=UTC)
        records: list[tuple[datetime, float, float, float, float, float]] = []

        if len(decompressed) % self.RECORD_STRUCT.size != 0:
            return None

        for seconds, open_i, high_i, low_i, close_i, volume in self.RECORD_STRUCT.iter_unpack(decompressed):
            timestamp = day_start + timedelta(seconds=int(seconds)) + ONE_MINUTE
            records.append(
                (
                    timestamp,
                    open_i / divisor,
                    high_i / divisor,
                    low_i / divisor,
                    close_i / divisor,
                    float(volume),
                )
            )

        if not records:
            return None

        frame = pd.DataFrame(records, columns=["datetime", "open", "high", "low", "close", "volume"])
        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
        return frame.set_index("datetime")

    @staticmethod
    def _month_end(month_start: date) -> date:
        if month_start.month == 12:
            next_month = date(month_start.year + 1, 1, 1)
        else:
            next_month = date(month_start.year, month_start.month + 1, 1)
        return next_month - timedelta(days=1)


def generate_synthetic_m5(pair: str, start_date: date, end_date: date, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed + abs(hash(pair)) % 10_000)
    index = pd.date_range(
        start=pd.Timestamp(start_date, tz=UTC),
        end=pd.Timestamp(end_date + timedelta(days=1), tz=UTC),
        freq="5min",
        inclusive="left",
    )
    index_ny = index.tz_convert(NY_TZ)
    minutes = index_ny.hour * 60 + index_ny.minute
    dow = index_ny.dayofweek
    mask = (
        ((dow >= 0) & (dow <= 3))
        | ((dow == 4) & (minutes <= 17 * 60))
        | ((dow == 6) & (minutes > 17 * 60))
    )
    index = index[mask]

    meta = PAIR_META[pair]
    anchor_price = {
        "EURUSD": 1.11,
        "GBPUSD": 1.28,
        "USDJPY": 138.0,
        "AUDUSD": 0.72,
        "USDCAD": 1.35,
        "USDCHF": 0.91,
        "GBPJPY": 178.0,
    }[pair]

    drift = np.sin(np.linspace(0, 18 * math.pi, len(index))) * meta["pip_size"] * 3
    noise = rng.normal(0.0, meta["pip_size"] * (4 if meta["quote"] != "JPY" else 35), len(index))
    close = anchor_price + np.cumsum(drift + noise)
    open_ = np.concatenate(([close[0]], close[:-1]))
    spread = np.abs(rng.normal(meta["pip_size"] * 4, meta["pip_size"] * 3, len(index)))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(20, 250, len(index)).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def load_pair_m5(pair: str, run_config: RunConfig) -> pd.DataFrame:
    start_date = date.fromisoformat(run_config.start)
    end_date = date.fromisoformat(run_config.end)

    if run_config.synthetic:
        return generate_synthetic_m5(pair, start_date, end_date)

    if run_config.source in {"local", "auto"}:
        try:
            return load_local_m5(pair, run_config.data_dir)
        except FileNotFoundError:
            if run_config.source == "local" or not run_config.download_missing:
                raise

    if run_config.download_missing or run_config.source == "dukascopy":
        downloader = DukascopyDownloader(run_config.data_dir / "cache" / "dukascopy")
        return downloader.load_or_download_m5(
            pair=pair,
            start_date=start_date,
            end_date=end_date,
            force_download=run_config.force_download,
        )

    raise FileNotFoundError(
        f"No hay dataset local para {pair} y no se habilito la descarga automatica."
    )


def clip_to_run_window(df: pd.DataFrame, run_config: RunConfig) -> pd.DataFrame:
    if df.index.tz is None:
        df = df.tz_localize(UTC)
    else:
        df = df.tz_convert(UTC)

    start_ts = pd.Timestamp(run_config.start, tz=UTC)
    end_ts = pd.Timestamp(date.fromisoformat(run_config.end) + timedelta(days=1), tz=UTC)
    clipped = df[(df.index >= start_ts) & (df.index < end_ts)].copy()
    if clipped.empty:
        raise ValueError("El rango solicitado no devuelve datos.")
    clipped = clipped.tz_convert(NY_TZ)
    return sanitize_fx_ohlcv(clipped)


def expected_fx_5m_index(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DatetimeIndex:
    grid = pd.date_range(
        start=start_ts.floor("5min"),
        end=end_ts.ceil("5min"),
        freq="5min",
        tz=start_ts.tz,
    )
    minutes = grid.hour * 60 + grid.minute
    dow = grid.dayofweek
    mask = (
        ((dow >= 0) & (dow <= 3))
        | ((dow == 4) & (minutes <= 17 * 60))
        | ((dow == 6) & (minutes > 17 * 60))
    )
    return grid[mask]


def sanitize_fx_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    sanitized = df.sort_index().copy()
    sanitized = sanitized[~sanitized.index.duplicated(keep="last")]

    ohlc = sanitized[["open", "high", "low", "close"]].to_numpy(dtype=float)
    sanitized["high"] = np.nanmax(ohlc, axis=1)
    sanitized["low"] = np.nanmin(ohlc, axis=1)

    if isinstance(sanitized.index, pd.DatetimeIndex) and sanitized.index.tz is not None and len(sanitized):
        expected = expected_fx_5m_index(sanitized.index[0], sanitized.index[-1])
        sanitized = sanitized.loc[sanitized.index.intersection(expected)]

    return sanitized


def validate_m5_dataset(pair: str, df: pd.DataFrame) -> dict[str, Any]:
    ordered = df.sort_index()
    duplicates = int(ordered.index.duplicated().sum())
    weekend_mask = ordered.index.dayofweek == 5
    weekend_mask |= (ordered.index.dayofweek == 6) & ((ordered.index.hour * 60 + ordered.index.minute) <= 17 * 60)
    weekend_mask |= (ordered.index.dayofweek == 4) & ((ordered.index.hour * 60 + ordered.index.minute) > 17 * 60)
    weekend_bars = int(weekend_mask.sum())

    nonpositive_prices = int(
        ((ordered[["open", "high", "low", "close"]] <= 0).any(axis=1)).sum()
    )
    invalid_hilo = int(((ordered["high"] < ordered["low"]) | (ordered["high"] < ordered["close"]) | (ordered["low"] > ordered["close"]) | (ordered["high"] < ordered["open"]) | (ordered["low"] > ordered["open"])).sum())
    zero_range_bars = int(((ordered["high"] - ordered["low"]) <= 0).sum())

    expected = expected_fx_5m_index(ordered.index[0], ordered.index[-1])
    missing_bars = expected.difference(ordered.index)
    unexpected_bars = ordered.index.difference(expected)
    coverage_ratio = len(ordered.index.intersection(expected)) / len(expected) if len(expected) else 0.0

    actual_positions = expected.get_indexer(ordered.index.intersection(expected))
    actual_positions = actual_positions[actual_positions >= 0]
    missing_streaks = np.diff(actual_positions) - 1 if len(actual_positions) > 1 else np.array([], dtype=int)
    max_missing_streak = int(missing_streaks.max()) if len(missing_streaks) else 0

    log_returns = np.log(ordered["close"]).diff().abs().replace([np.inf, -np.inf], np.nan)
    median_abs_return = float(log_returns.median()) if log_returns.notna().any() else 0.0
    outlier_threshold = median_abs_return * 25 if median_abs_return > 0 else float("inf")
    outlier_return_bars = int((log_returns > outlier_threshold).sum()) if np.isfinite(outlier_threshold) else 0

    quality_score = 100.0
    quality_score -= min(40.0, (1.0 - coverage_ratio) * 4000)
    quality_score -= min(15.0, duplicates * 0.05)
    quality_score -= min(10.0, weekend_bars * 0.01)
    quality_score -= min(10.0, zero_range_bars * 0.005)
    quality_score -= min(15.0, outlier_return_bars * 0.05)
    if nonpositive_prices:
        quality_score -= 30.0
    if invalid_hilo:
        quality_score -= 25.0
    quality_score = max(0.0, quality_score)

    if nonpositive_prices or invalid_hilo or coverage_ratio < 0.97:
        status = "fail"
    elif coverage_ratio < 0.995 or duplicates or weekend_bars or outlier_return_bars > 20:
        status = "warn"
    else:
        status = "pass"

    return {
        "pair": pair,
        "bars": int(len(ordered)),
        "coverage_ratio": coverage_ratio,
        "missing_bars": int(len(missing_bars)),
        "max_missing_streak_bars": max_missing_streak,
        "duplicate_timestamps": duplicates,
        "unexpected_schedule_bars": int(len(unexpected_bars)),
        "weekend_bars": weekend_bars,
        "zero_range_bars": zero_range_bars,
        "nonpositive_prices": nonpositive_prices,
        "invalid_hilo_bars": invalid_hilo,
        "outlier_return_bars": outlier_return_bars,
        "quality_score": quality_score,
        "quality_status": status,
    }


def validate_data_bundle(raw_bundle: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = [validate_m5_dataset(pair, df) for pair, df in raw_bundle.items()]
    frame = pd.DataFrame(rows)
    status_order = {"fail": 0, "warn": 1, "pass": 2}
    frame["_status_order"] = frame["quality_status"].map(status_order).fillna(9)
    frame = frame.sort_values(["_status_order", "quality_score", "pair"], ascending=[True, False, True])
    return frame.drop(columns="_status_order").reset_index(drop=True)


def prepare_pair_features(m5: pd.DataFrame, params: StrategyParameters) -> pd.DataFrame:
    m5 = m5.copy().sort_index()
    m15 = resample_ohlcv(m5, "15min")
    h1 = resample_ohlcv(m5, "1h")

    m5["m5_rsi"] = rsi(m5["close"], params.m5_rsi_period)
    m5["m5_atr"] = atr(m5, params.m5_atr_period)
    m5["m5_range"] = m5["high"] - m5["low"]
    m5["m5_prev_range"] = m5["m5_range"].shift(1)
    m5["m5_body"] = m5["close"] - m5["open"]
    m5["m5_ema_fast"] = ema(m5["close"], 8)
    m5["m5_ema_slow"] = ema(m5["close"], 21)
    m5["m5_roll_high_3"] = m5["high"].rolling(3).max().shift(1)
    m5["m5_roll_low_3"] = m5["low"].rolling(3).min().shift(1)

    m15_features = pd.DataFrame(index=m15.index)
    m15_features["m15_close"] = m15["close"]
    m15_features["m15_high"] = m15["high"]
    m15_features["m15_low"] = m15["low"]
    m15_features["m15_ema"] = ema(m15["close"], params.m15_ema_period)
    m15_features["m15_ema_slope"] = m15_features["m15_ema"].diff()
    m15_features["m15_rsi"] = rsi(m15["close"], params.m15_rsi_period)
    m15_features["m15_atr"] = atr(m15, params.m15_atr_period)
    m15_features["m15_prev_range"] = (m15["high"] - m15["low"]).shift(1)
    m15_features["m15_prev_atr"] = m15_features["m15_atr"].shift(1)

    h1_features = pd.DataFrame(index=h1.index)
    h1_features["h1_ema_fast"] = ema(h1["close"], params.h1_ema_fast)
    h1_features["h1_ema_slow"] = ema(h1["close"], params.h1_ema_slow)
    h1_features["h1_ema_slow_slope"] = h1_features["h1_ema_slow"].diff()
    h1_features["h1_adx"] = adx(h1, params.h1_adx_period)
    h1_features["h1_atr"] = atr(h1, params.h1_adx_period)

    merged = m5.join(m15_features.reindex(m5.index, method="ffill"))
    merged = merged.join(h1_features.reindex(m5.index, method="ffill"))
    required = [
        "m5_rsi",
        "m5_atr",
        "m5_range",
        "m5_ema_fast",
        "m5_ema_slow",
        "m5_roll_high_3",
        "m5_roll_low_3",
        "m15_ema",
        "m15_ema_slope",
        "m15_rsi",
        "m15_prev_range",
        "m15_prev_atr",
        "h1_ema_fast",
        "h1_ema_slow",
        "h1_ema_slow_slope",
        "h1_adx",
        "h1_atr",
    ]
    return merged.dropna(subset=required).copy()


def load_raw_bundle(run_config: RunConfig) -> dict[str, pd.DataFrame]:
    bundle: dict[str, pd.DataFrame] = {}
    for pair in run_config.pairs:
        raw = load_pair_m5(pair, run_config)
        bundle[pair] = clip_to_run_window(raw, run_config)
    return bundle


def relevant_news_currencies(pairs: Iterable[str]) -> set[str]:
    currencies = set()
    for pair in pairs:
        meta = PAIR_META[pair]
        currencies.add(meta["base"])
        currencies.add(meta["quote"])
    return currencies


def normalize_news_keywords(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = list(value)
    keywords = []
    for part in parts:
        normalized = " ".join(str(part).strip().lower().split())
        if normalized:
            keywords.append(normalized)
    return tuple(dict.fromkeys(keywords))


def normalize_event_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def is_hard_news_event(event_text: Any, keywords: Iterable[str]) -> bool:
    normalized = normalize_event_text(event_text)
    if not normalized:
        return False
    return any(keyword in normalized for keyword in keywords)


def normalize_news_importance(value: Any) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(value)

    text = str(value).strip().lower()
    if not text:
        return 0
    if "high" in text or "red" in text:
        return 3
    if "medium" in text or "orange" in text:
        return 2
    if "low" in text or "yellow" in text:
        return 1
    for digit in ("3", "2", "1"):
        if digit in text:
            return int(digit)
    return 0


def normalize_news_frame(frame: pd.DataFrame, run_config: RunConfig, source: str) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    lower_columns = {str(column).strip().lower(): column for column in frame.columns}
    for column in frame.columns:
        lower = str(column).strip().lower()
        if lower in {"datetime", "timestamp", "date"}:
            rename_map[column] = "datetime"
        elif lower in {"time"} and "datetime" not in rename_map.values():
            rename_map[column] = "time"
        elif lower in {"currency", "ccy", "curr"}:
            rename_map[column] = "currency"
        elif lower in {"impact", "importance", "star", "stars"}:
            rename_map[column] = "importance"
        elif lower in {"event", "title", "name", "category", "indicator"}:
            rename_map[column] = "event"
        elif lower in {"country"}:
            rename_map[column] = "country"

    normalized = frame.rename(columns=rename_map).copy()
    if "datetime" not in normalized.columns and {"date", "time"}.issubset(lower_columns):
        date_column = lower_columns["date"]
        time_column = lower_columns["time"]
        normalized["datetime"] = frame[date_column].astype(str).str.strip() + " " + frame[time_column].astype(str).str.strip()

    if "datetime" not in normalized.columns:
        return pd.DataFrame(columns=["timestamp", "currency", "importance", "event", "country", "source"])

    datetime_text = normalized["datetime"].astype(str).str.strip()
    has_explicit_tz = datetime_text.str.contains(r"(?:z|[+-]\d{2}:?\d{2})$", case=False, regex=True, na=False)
    if has_explicit_tz.any():
        datetimes = pd.to_datetime(datetime_text, errors="coerce", utc=True)
        if (~has_explicit_tz).any():
            naive_datetimes = pd.to_datetime(datetime_text[~has_explicit_tz], errors="coerce")
            naive_datetimes = naive_datetimes.dt.tz_localize(
                run_config.news_timezone,
                ambiguous="NaT",
                nonexistent="shift_forward",
            ).dt.tz_convert(UTC)
            datetimes.loc[~has_explicit_tz] = naive_datetimes
        datetimes = datetimes.dt.tz_convert(NY_TZ)
    else:
        datetimes = pd.to_datetime(datetime_text, errors="coerce")
        datetimes = datetimes.dt.tz_localize(run_config.news_timezone, ambiguous="NaT", nonexistent="shift_forward")
        datetimes = datetimes.dt.tz_convert(NY_TZ)
    normalized["timestamp"] = datetimes

    if "currency" not in normalized.columns and "country" in normalized.columns:
        normalized["currency"] = normalized["country"].map(COUNTRY_TO_CURRENCY)
    if "currency" not in normalized.columns:
        normalized["currency"] = pd.Series(index=normalized.index, dtype=object)
    if "importance" not in normalized.columns:
        normalized["importance"] = pd.Series(0, index=normalized.index)
    if "event" not in normalized.columns:
        normalized["event"] = pd.Series("", index=normalized.index)
    if "country" not in normalized.columns:
        normalized["country"] = pd.Series("", index=normalized.index)

    normalized["currency"] = normalized["currency"].astype(str).str.upper().str.strip()
    normalized["importance"] = normalized["importance"].map(normalize_news_importance)
    normalized["event"] = normalized["event"].astype(str).str.strip()
    normalized["country"] = normalized["country"].astype(str).str.strip()
    hard_keywords = normalize_news_keywords(run_config.news_hard_keywords)
    normalized["event_key"] = normalized["event"].map(normalize_event_text)
    normalized["hard_event"] = normalized["event_key"].map(lambda text: is_hard_news_event(text, hard_keywords))
    normalized["event_tier"] = np.where(normalized["hard_event"], "hard", "standard")
    normalized["source"] = source

    relevant = relevant_news_currencies(run_config.pairs)
    max_pre_minutes = max(run_config.news_no_entry_pre_minutes, run_config.news_hard_no_entry_pre_minutes)
    max_post_minutes = max(run_config.news_no_entry_post_minutes, run_config.news_hard_no_entry_post_minutes)
    start_ts = pd.Timestamp(run_config.start, tz=NY_TZ) - pd.Timedelta(minutes=max_pre_minutes)
    end_ts = pd.Timestamp(date.fromisoformat(run_config.end) + timedelta(days=1), tz=NY_TZ) + pd.Timedelta(minutes=max_post_minutes)

    normalized = normalized.dropna(subset=["timestamp"])
    normalized = normalized[
        normalized["currency"].isin(relevant)
        & ((normalized["importance"] >= run_config.news_min_importance) | normalized["hard_event"])
        & (normalized["timestamp"] >= start_ts)
        & (normalized["timestamp"] < end_ts)
    ]
    return normalized[
        ["timestamp", "currency", "importance", "event", "country", "source", "hard_event", "event_tier"]
    ].sort_values("timestamp").reset_index(drop=True)


def load_local_news_events(run_config: RunConfig) -> pd.DataFrame:
    if run_config.news_file is None:
        return pd.DataFrame(columns=["timestamp", "currency", "importance", "event", "country", "source", "hard_event", "event_tier"])
    frame = pd.read_csv(run_config.news_file)
    return normalize_news_frame(frame, run_config, source="csv")


def fetch_tradingeconomics_news(run_config: RunConfig) -> pd.DataFrame:
    encoded_key = quote(run_config.news_api_key, safe=":")
    currencies = sorted(relevant_news_currencies(run_config.pairs))
    country_tokens = [quote(TE_COUNTRY_BY_CURRENCY[currency], safe="") for currency in currencies if currency in TE_COUNTRY_BY_CURRENCY]
    countries = ",".join(country_tokens)
    if not countries:
        return pd.DataFrame(columns=["timestamp", "currency", "importance", "event", "country", "source", "hard_event", "event_tier"])
    url = (
        f"https://api.tradingeconomics.com/calendar/country/{countries}/"
        f"{run_config.start}/{run_config.end}?c={encoded_key}"
    )
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=60) as response:
        payload = response.read().decode("utf-8")
    records = json.loads(payload)
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "currency", "importance", "event", "country", "source", "hard_event", "event_tier"])
    return normalize_news_frame(frame, run_config, source="tradingeconomics")


def load_news_events(run_config: RunConfig) -> pd.DataFrame:
    if run_config.news_source == "none":
        return pd.DataFrame(columns=["timestamp", "currency", "importance", "event", "country", "source", "hard_event", "event_tier"])
    if run_config.news_source == "csv":
        return load_local_news_events(run_config)
    if run_config.news_source == "tradingeconomics":
        return fetch_tradingeconomics_news(run_config)
    raise ValueError(f"Fuente de noticias no soportada: {run_config.news_source}")


def build_pair_news_masks(
    master_index: pd.DatetimeIndex,
    pairs: Iterable[str],
    news_events: pd.DataFrame,
    run_config: RunConfig,
) -> dict[str, dict[str, np.ndarray]]:
    masks = {
        pair: {
            "block": np.zeros(len(master_index), dtype=bool),
            "flatten": np.zeros(len(master_index), dtype=bool),
        }
        for pair in pairs
    }
    if news_events.empty or len(master_index) == 0:
        return masks

    for event in news_events.itertuples(index=False):
        event_ts = pd.Timestamp(event.timestamp).tz_convert(master_index.tz)
        hard_event = bool(getattr(event, "hard_event", False)) and run_config.hard_news_veto_enabled
        pre_minutes = run_config.news_hard_no_entry_pre_minutes if hard_event else run_config.news_no_entry_pre_minutes
        post_minutes = run_config.news_hard_no_entry_post_minutes if hard_event else run_config.news_no_entry_post_minutes
        flatten_minutes = run_config.news_hard_flatten_minutes_before if hard_event else run_config.news_flatten_minutes_before
        block_start = event_ts - pd.Timedelta(minutes=pre_minutes)
        block_end = event_ts + pd.Timedelta(minutes=post_minutes)
        flatten_start = event_ts - pd.Timedelta(minutes=flatten_minutes)

        block_left = master_index.searchsorted(block_start, side="left")
        block_right = master_index.searchsorted(block_end, side="right")
        flatten_left = master_index.searchsorted(flatten_start, side="left")
        flatten_right = block_right

        for pair in pairs:
            if event.currency not in relevant_news_currencies((pair,)):
                continue
            masks[pair]["block"][block_left:block_right] = True
            masks[pair]["flatten"][flatten_left:flatten_right] = True
    return masks


def build_pair_shock_masks(
    master_index: pd.DatetimeIndex,
    views: dict[str, PairDataView],
    run_config: RunConfig,
) -> dict[str, dict[str, np.ndarray]]:
    masks = {
        pair: {
            "block": np.zeros(len(master_index), dtype=bool),
            "flatten": np.zeros(len(master_index), dtype=bool),
        }
        for pair in views
    }
    if not run_config.shock_guard_enabled or len(master_index) == 0:
        return masks

    bar_minutes = 5.0
    if len(master_index) > 1:
        inferred = (master_index[1] - master_index[0]).total_seconds() / 60.0
        if inferred > 0:
            bar_minutes = inferred
    cooldown_bars = max(1, int(math.ceil(run_config.shock_cooldown_minutes / bar_minutes)))

    for pair, view in views.items():
        arrays = view.arrays
        valid_positions = np.flatnonzero(view.valid_mask)
        for pos in valid_positions:
            atr_value = arrays["m5_atr"][pos]
            if not np.isfinite(atr_value) or atr_value <= 0:
                continue

            candle_range_ratio = (arrays["high"][pos] - arrays["low"][pos]) / atr_value
            candle_body_ratio = abs(arrays["close"][pos] - arrays["open"][pos]) / atr_value
            block_triggered = (
                candle_range_ratio >= run_config.shock_no_entry_atr_multiple
                or candle_body_ratio >= run_config.shock_no_entry_atr_multiple
            )
            flatten_triggered = (
                candle_range_ratio >= run_config.shock_flatten_atr_multiple
                or candle_body_ratio >= run_config.shock_flatten_atr_multiple
            )
            if not block_triggered and not flatten_triggered:
                continue

            right = min(len(master_index), pos + cooldown_bars + 1)
            if block_triggered:
                masks[pair]["block"][pos:right] = True
            if flatten_triggered:
                masks[pair]["flatten"][pos:right] = True
    return masks


def merge_pair_event_masks(
    left_masks: dict[str, dict[str, np.ndarray]],
    right_masks: dict[str, dict[str, np.ndarray]],
    pairs: Iterable[str],
) -> dict[str, dict[str, np.ndarray]]:
    merged: dict[str, dict[str, np.ndarray]] = {}
    for pair in pairs:
        left = left_masks[pair]
        right = right_masks[pair]
        merged[pair] = {
            "block": np.logical_or(left["block"], right["block"]),
            "flatten": np.logical_or(left["flatten"], right["flatten"]),
        }
    return merged


def build_pair_data_views(prepared_data: dict[str, pd.DataFrame], pairs: Iterable[str]) -> tuple[pd.DatetimeIndex, dict[str, PairDataView]]:
    master_index = pd.DatetimeIndex(sorted(set().union(*(frame.index for frame in prepared_data.values()))))
    views: dict[str, PairDataView] = {}

    for pair in pairs:
        aligned = prepared_data[pair].reindex(master_index)
        valid_mask = aligned["close"].notna().to_numpy()
        positions = np.flatnonzero(valid_mask)
        prev_bar_pos = np.full(len(master_index), -1, dtype=int)
        next_bar_pos = np.full(len(master_index), -1, dtype=int)
        asof_pos = np.full(len(master_index), -1, dtype=int)

        last_seen = -1
        for i in range(len(master_index)):
            if valid_mask[i]:
                last_seen = i
            asof_pos[i] = last_seen

        if len(positions) > 1:
            prev_bar_pos[positions[1:]] = positions[:-1]
            next_bar_pos[positions[:-1]] = positions[1:]

        arrays = {column: aligned[column].to_numpy(dtype=float) for column in aligned.columns}
        views[pair] = PairDataView(
            pair=pair,
            index=master_index,
            arrays=arrays,
            valid_mask=valid_mask,
            prev_bar_pos=prev_bar_pos,
            next_bar_pos=next_bar_pos,
            asof_pos=asof_pos,
        )

    return master_index, views


class MultiTimeframeFXBacktester:
    def __init__(
        self,
        views: dict[str, PairDataView],
        master_index: pd.DatetimeIndex,
        params: StrategyParameters,
        broker: BrokerConfig,
        pairs: Iterable[str],
        correlation_groups: Iterable[Iterable[str]],
        news_masks: dict[str, dict[str, np.ndarray]] | None = None,
        shock_masks: dict[str, dict[str, np.ndarray]] | None = None,
    ) -> None:
        self.views = views
        self.master_index = master_index
        self.params = params
        self.broker = broker
        self.pairs = tuple(pairs)
        self.cash = broker.initial_capital
        self.positions: dict[str, Position] = {}
        self.pending: dict[str, PendingEntry] = {}
        self.trades: list[dict[str, Any]] = []
        self.equity_points: list[dict[str, Any]] = []
        self.daily_direction_book: dict[str, dict[date, set[str]]] = defaultdict(lambda: defaultdict(set))
        self.context_stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {"trades": 0.0, "wins": 0.0, "losses": 0.0, "r_sum": 0.0, "pnl_sum": 0.0}
        )
        self.news_masks = news_masks or {
            pair: {
                "block": np.zeros(len(master_index), dtype=bool),
                "flatten": np.zeros(len(master_index), dtype=bool),
            }
            for pair in self.pairs
        }
        self.shock_masks = shock_masks or {
            pair: {
                "block": np.zeros(len(master_index), dtype=bool),
                "flatten": np.zeros(len(master_index), dtype=bool),
            }
            for pair in self.pairs
        }

        pair_to_groups: dict[str, list[tuple[str, ...]]] = defaultdict(list)
        for group in correlation_groups:
            normalized = tuple(group)
            for pair in normalized:
                pair_to_groups[pair].append(normalized)
        self.pair_to_groups = dict(pair_to_groups)

    def run(self) -> BacktestResult:
        for pos, timestamp in enumerate(self.master_index):
            for pair in self.pairs:
                if not self.views[pair].valid_mask[pos]:
                    continue
                pending = self.pending.get(pair)
                if pending and pending.execute_pos == pos:
                    if self.news_masks[pair]["block"][pos] or self.shock_masks[pair]["block"][pos]:
                        self.pending.pop(pair, None)
                        continue
                    self._execute_pending_entry(pair, pending, pos)
                    self.pending.pop(pair, None)

            for pair in list(self.positions.keys()):
                view = self.views[pair]
                if not view.valid_mask[pos]:
                    continue
                self._process_position_bar(pair, pos)

            equity = self._portfolio_equity(pos)
            self.equity_points.append({"timestamp": timestamp, "equity": equity, "cash": self.cash})

            for pair in self.pairs:
                view = self.views[pair]
                if not view.valid_mask[pos]:
                    continue
                if pair in self.positions or pair in self.pending:
                    continue
                self._maybe_schedule_signal(pair, pos)

        final_pos = len(self.master_index) - 1
        for pair in list(self.positions.keys()):
            self._close_position(pair, final_pos, self.views[pair].arrays["close"][final_pos], "end_of_data")

        trades_df = pd.DataFrame(self.trades)
        if trades_df.empty:
            trades_df = pd.DataFrame(
                columns=[
                    "pair",
                    "direction",
                    "entry_time",
                    "exit_time",
                    "entry_price",
                    "exit_price",
                    "units",
                    "gross_pnl_usd",
                    "net_pnl_usd",
                    "r_multiple",
                    "commission_usd",
                    "exit_reason",
                    "setup_tag",
                ]
            )
        else:
            trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True).dt.tz_convert(NY_TZ)
            trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], utc=True).dt.tz_convert(NY_TZ)

        equity_curve = pd.DataFrame(self.equity_points)
        equity_curve["timestamp"] = pd.to_datetime(equity_curve["timestamp"], utc=True).dt.tz_convert(NY_TZ)

        pair_summary = summarize_pair_stats(trades_df)
        context_summary = summarize_context_stats(trades_df)
        setup_summary = summarize_setup_stats(trades_df)
        yearly_summary = summarize_period_stats(trades_df, "Y")
        monthly_summary = summarize_period_stats(trades_df, "M")
        portfolio_summary = summarize_portfolio(trades_df, equity_curve, self.broker.initial_capital)
        robustness_score = compute_robustness_score(portfolio_summary, yearly_summary, monthly_summary)

        return BacktestResult(
            parameters=asdict(self.params),
            broker=asdict(self.broker),
            trades=trades_df,
            equity_curve=equity_curve,
            data_quality=pd.DataFrame(),
            news_events=pd.DataFrame(),
            context_summary=context_summary,
            setup_summary=setup_summary,
            pair_summary=pair_summary,
            yearly_summary=yearly_summary,
            monthly_summary=monthly_summary,
            portfolio_summary=portfolio_summary,
            robustness_score=robustness_score,
        )

    def _maybe_schedule_signal(self, pair: str, pos: int) -> None:
        view = self.views[pair]
        prev_pos = view.prev_bar_pos[pos]
        next_pos = view.next_bar_pos[pos]
        if prev_pos < 0 or next_pos < 0:
            return

        entry_time = self.master_index[pos]
        local_date = entry_time.date()
        session_time = entry_time.time()
        if not (self.broker.session_start_time <= session_time < self.broker.session_end_time):
            return
        if (
            self.news_masks[pair]["block"][pos]
            or self.news_masks[pair]["block"][next_pos]
            or self.shock_masks[pair]["block"][pos]
            or self.shock_masks[pair]["block"][next_pos]
        ):
            return

        long_pending = self._build_pending_entry(pair, "long", pos, prev_pos, next_pos, entry_time)
        short_pending = self._build_pending_entry(pair, "short", pos, prev_pos, next_pos, entry_time)

        if long_pending is not None and "long" not in self.daily_direction_book[pair][local_date]:
            self.pending[pair] = long_pending
        elif short_pending is not None and "short" not in self.daily_direction_book[pair][local_date]:
            self.pending[pair] = short_pending

    def _build_pending_entry(
        self,
        pair: str,
        direction: str,
        pos: int,
        prev_pos: int,
        next_pos: int,
        entry_time: pd.Timestamp,
    ) -> PendingEntry | None:
        if self.params.strategy_family == "usdjp_session_playbook":
            return self._build_usdjpy_session_playbook_entry(pair, direction, pos, prev_pos, next_pos, entry_time)
        if self.params.strategy_family == "session_trend_reclaim":
            return self._build_session_trend_reclaim_entry(pair, direction, pos, prev_pos, next_pos, entry_time)
        if self.params.strategy_family == "core_session_reversion":
            return self._build_core_session_reversion_entry(pair, direction, pos, prev_pos, next_pos, entry_time)
        if self.params.strategy_family == "adaptive_session_reversion":
            return self._build_adaptive_session_reversion_entry(pair, direction, pos, prev_pos, next_pos, entry_time)
        if self.params.strategy_family == "session_mean_reversion":
            return self._build_mean_reversion_entry(pair, direction, pos, prev_pos, next_pos, entry_time)
        return self._build_trend_pullback_entry(pair, direction, pos, prev_pos, next_pos, entry_time)

    def _build_trend_pullback_entry(
        self,
        pair: str,
        direction: str,
        pos: int,
        prev_pos: int,
        next_pos: int,
        entry_time: pd.Timestamp,
    ) -> PendingEntry | None:
        is_long = direction == "long"
        if (is_long and self._long_signal(pair, pos, prev_pos)) or ((not is_long) and self._short_signal(pair, pos, prev_pos)):
            return PendingEntry(pair, direction, pos, next_pos, entry_time, self.views[pair].arrays["m5_atr"][pos])
        return None

    def _long_signal(self, pair: str, pos: int, prev_pos: int) -> bool:
        arrays = self.views[pair].arrays
        regime_ok = (
            arrays["h1_ema_fast"][pos] > arrays["h1_ema_slow"][pos]
            and arrays["h1_ema_slow_slope"][pos] > 0
            and arrays["h1_adx"][pos] > self.params.h1_adx_threshold
        )
        setup_ok = (
            arrays["m15_low"][pos] <= arrays["m15_ema"][pos]
            and arrays["m15_rsi"][pos] < self.params.m15_rsi_pullback_long
            and arrays["m15_prev_range"][pos] <= self.params.m15_max_prev_range_atr * arrays["m15_prev_atr"][pos]
        )
        trigger_ok = (
            arrays["m5_rsi"][prev_pos] <= self.params.m5_trigger_rsi_midpoint
            and arrays["m5_rsi"][pos] > self.params.m5_trigger_rsi_midpoint
            and arrays["close"][pos] > arrays["high"][prev_pos]
        )
        return bool(regime_ok and setup_ok and trigger_ok)

    def _short_signal(self, pair: str, pos: int, prev_pos: int) -> bool:
        arrays = self.views[pair].arrays
        regime_ok = (
            arrays["h1_ema_fast"][pos] < arrays["h1_ema_slow"][pos]
            and arrays["h1_ema_slow_slope"][pos] < 0
            and arrays["h1_adx"][pos] > self.params.h1_adx_threshold
        )
        setup_ok = (
            arrays["m15_high"][pos] >= arrays["m15_ema"][pos]
            and arrays["m15_rsi"][pos] > self.params.m15_rsi_pullback_short
            and arrays["m15_prev_range"][pos] <= self.params.m15_max_prev_range_atr * arrays["m15_prev_atr"][pos]
        )
        trigger_ok = (
            arrays["m5_rsi"][prev_pos] >= self.params.m5_trigger_rsi_midpoint
            and arrays["m5_rsi"][pos] < self.params.m5_trigger_rsi_midpoint
            and arrays["close"][pos] < arrays["low"][prev_pos]
        )
        return bool(regime_ok and setup_ok and trigger_ok)

    def _build_mean_reversion_entry(
        self,
        pair: str,
        direction: str,
        pos: int,
        prev_pos: int,
        next_pos: int,
        entry_time: pd.Timestamp,
    ) -> PendingEntry | None:
        arrays = self.views[pair].arrays
        m5_atr_value = arrays["m5_atr"][pos]
        m15_atr_value = arrays["m15_atr"][pos]
        if not np.isfinite(m5_atr_value) or not np.isfinite(m15_atr_value) or m5_atr_value <= 0 or m15_atr_value <= 0:
            return None

        if direction == "long":
            regime_ok = (
                arrays["close"][pos] >= arrays["h1_ema_slow"][pos]
                and arrays["h1_ema_fast"][pos] >= arrays["h1_ema_slow"][pos]
                and arrays["h1_adx"][pos] <= self.params.mr_h1_adx_max
            )
            extension_ok = arrays["m15_close"][pos] <= (arrays["m15_ema"][pos] - self.params.mr_m15_extension_atr * m15_atr_value)
            setup_ok = (
                extension_ok
                and arrays["m15_rsi"][pos] <= self.params.mr_m15_rsi_long
                and arrays["m15_prev_range"][pos] <= self.params.m15_max_prev_range_atr * arrays["m15_prev_atr"][pos]
            )
            trigger_ok = (
                arrays["m5_rsi"][prev_pos] <= self.params.mr_m5_reclaim_rsi_long
                and arrays["m5_rsi"][pos] > self.params.mr_m5_reclaim_rsi_long
                and arrays["close"][pos] > arrays["close"][prev_pos]
            )
            target_price = arrays["m15_ema"][pos] - (self.params.mr_target_atr_buffer * m5_atr_value)
        else:
            regime_ok = (
                arrays["close"][pos] <= arrays["h1_ema_slow"][pos]
                and arrays["h1_ema_fast"][pos] <= arrays["h1_ema_slow"][pos]
                and arrays["h1_adx"][pos] <= self.params.mr_h1_adx_max
            )
            extension_ok = arrays["m15_close"][pos] >= (arrays["m15_ema"][pos] + self.params.mr_m15_extension_atr * m15_atr_value)
            setup_ok = (
                extension_ok
                and arrays["m15_rsi"][pos] >= self.params.mr_m15_rsi_short
                and arrays["m15_prev_range"][pos] <= self.params.m15_max_prev_range_atr * arrays["m15_prev_atr"][pos]
            )
            trigger_ok = (
                arrays["m5_rsi"][prev_pos] >= self.params.mr_m5_reclaim_rsi_short
                and arrays["m5_rsi"][pos] < self.params.mr_m5_reclaim_rsi_short
                and arrays["close"][pos] < arrays["close"][prev_pos]
            )
            target_price = arrays["m15_ema"][pos] + (self.params.mr_target_atr_buffer * m5_atr_value)

        if not (regime_ok and setup_ok and trigger_ok):
            return None

        return PendingEntry(
            pair=pair,
            direction=direction,
            signal_pos=pos,
            execute_pos=next_pos,
            planned_entry_time=entry_time,
            signal_atr=m5_atr_value,
            stop_distance=m5_atr_value * self.params.stop_atr_multiple,
            take_profit_price=target_price,
        )

    def _build_adaptive_session_reversion_entry(
        self,
        pair: str,
        direction: str,
        pos: int,
        prev_pos: int,
        next_pos: int,
        entry_time: pd.Timestamp,
    ) -> PendingEntry | None:
        if entry_time.time() >= time.fromisoformat(self.params.asr_entry_end):
            return None

        arrays = self.views[pair].arrays
        m5_atr_value = arrays["m5_atr"][pos]
        m15_atr_value = arrays["m15_atr"][pos]
        h1_atr_value = arrays["h1_atr"][pos]
        if (
            not np.isfinite(m5_atr_value)
            or not np.isfinite(m15_atr_value)
            or not np.isfinite(h1_atr_value)
            or m5_atr_value <= 0
            or m15_atr_value <= 0
            or h1_atr_value <= 0
        ):
            return None

        h1_distance_atr = abs(arrays["close"][pos] - arrays["h1_ema_slow"][pos]) / h1_atr_value
        common_regime = (
            arrays["h1_adx"][pos] <= self.params.asr_h1_adx_max
            and h1_distance_atr <= self.params.asr_h1_distance_atr_max
            and arrays["m15_prev_range"][pos] <= self.params.m15_max_prev_range_atr * arrays["m15_prev_atr"][pos]
        )
        if not common_regime:
            return None

        if not self._context_static_allows(pair, direction, pos, entry_time):
            return None

        if direction == "long":
            setup_ok = (
                arrays["m15_close"][pos] <= arrays["m15_ema"][pos] - self.params.asr_m15_extension_atr * m15_atr_value
                and arrays["m15_rsi"][pos] <= self.params.asr_m15_rsi_long
                and arrays["m5_rsi"][prev_pos] <= self.params.asr_m5_reclaim_rsi_long
                and arrays["m5_rsi"][pos] > self.params.asr_m5_reclaim_rsi_long
                and arrays["close"][pos] > arrays["high"][prev_pos]
            )
            target_price = arrays["m15_ema"][pos] - self.params.asr_target_atr_buffer * m5_atr_value
        else:
            setup_ok = (
                arrays["m15_close"][pos] >= arrays["m15_ema"][pos] + self.params.asr_m15_extension_atr * m15_atr_value
                and arrays["m15_rsi"][pos] >= self.params.asr_m15_rsi_short
                and arrays["m5_rsi"][prev_pos] >= self.params.asr_m5_reclaim_rsi_short
                and arrays["m5_rsi"][pos] < self.params.asr_m5_reclaim_rsi_short
                and arrays["close"][pos] < arrays["low"][prev_pos]
            )
            target_price = arrays["m15_ema"][pos] + self.params.asr_target_atr_buffer * m5_atr_value

        if not setup_ok:
            return None

        context_key = self._build_context_key(pair, direction, pos, entry_time)
        context_allowed, context_snapshot = self._context_allows_trade(context_key)
        if not context_allowed:
            return None

        return PendingEntry(
            pair=pair,
            direction=direction,
            signal_pos=pos,
            execute_pos=next_pos,
            planned_entry_time=entry_time,
            signal_atr=m5_atr_value,
            stop_distance=m5_atr_value * self.params.stop_atr_multiple,
            take_profit_price=target_price,
            take_profit_rr_cap=self.params.asr_take_profit_rr_cap,
            max_hold_bars=self.params.asr_max_hold_bars,
            context_key=context_key,
            context_trades_before=int(context_snapshot["trades"]),
            context_win_rate_before=float(context_snapshot["win_rate"]),
            context_expectancy_before=float(context_snapshot["expectancy_r"]),
        )

    def _build_core_session_reversion_entry(
        self,
        pair: str,
        direction: str,
        pos: int,
        prev_pos: int,
        next_pos: int,
        entry_time: pd.Timestamp,
    ) -> PendingEntry | None:
        entry_clock = entry_time.time()
        if not (time.fromisoformat(self.params.csr_entry_start) <= entry_clock < time.fromisoformat(self.params.csr_entry_end)):
            return None

        arrays = self.views[pair].arrays
        m5_atr_value = arrays["m5_atr"][pos]
        m15_atr_value = arrays["m15_atr"][pos]
        h1_atr_value = arrays["h1_atr"][pos]
        if (
            not np.isfinite(m5_atr_value)
            or not np.isfinite(m15_atr_value)
            or not np.isfinite(h1_atr_value)
            or m5_atr_value <= 0
            or m15_atr_value <= 0
            or h1_atr_value <= 0
        ):
            return None

        h1_distance_atr = abs(arrays["close"][pos] - arrays["h1_ema_slow"][pos]) / h1_atr_value
        h1_ema_spread_atr = abs(arrays["h1_ema_fast"][pos] - arrays["h1_ema_slow"][pos]) / h1_atr_value
        body_atr = abs(arrays["close"][pos] - arrays["open"][pos]) / m5_atr_value
        common_regime = (
            arrays["h1_adx"][pos] <= self.params.csr_h1_adx_max
            and h1_distance_atr <= self.params.csr_h1_distance_atr_max
            and h1_ema_spread_atr <= self.params.csr_h1_ema_spread_atr_max
            and arrays["m15_prev_range"][pos] <= self.params.m15_max_prev_range_atr * arrays["m15_prev_atr"][pos]
            and body_atr >= self.params.csr_reversal_body_atr_min
        )
        if not common_regime:
            return None

        if not self._context_static_allows(pair, direction, pos, entry_time):
            return None

        if direction == "long":
            setup_ok = (
                arrays["m15_close"][pos] <= arrays["m15_ema"][pos] - self.params.csr_m15_extension_atr * m15_atr_value
                and arrays["m15_rsi"][pos] <= self.params.csr_m15_rsi_long
                and arrays["m5_rsi"][prev_pos] <= self.params.csr_m5_reclaim_rsi_long
                and arrays["m5_rsi"][pos] > self.params.csr_m5_reclaim_rsi_long
                and arrays["close"][pos] > arrays["high"][prev_pos]
                and arrays["close"][pos] > arrays["open"][pos]
            )
            target_price = arrays["m15_ema"][pos] - self.params.csr_target_atr_buffer * m5_atr_value
        else:
            setup_ok = (
                arrays["m15_close"][pos] >= arrays["m15_ema"][pos] + self.params.csr_m15_extension_atr * m15_atr_value
                and arrays["m15_rsi"][pos] >= self.params.csr_m15_rsi_short
                and arrays["m5_rsi"][prev_pos] >= self.params.csr_m5_reclaim_rsi_short
                and arrays["m5_rsi"][pos] < self.params.csr_m5_reclaim_rsi_short
                and arrays["close"][pos] < arrays["low"][prev_pos]
                and arrays["close"][pos] < arrays["open"][pos]
            )
            target_price = arrays["m15_ema"][pos] + self.params.csr_target_atr_buffer * m5_atr_value

        if not setup_ok:
            return None

        context_key = self._build_context_key(pair, direction, pos, entry_time)
        context_allowed, context_snapshot = self._context_allows_trade(context_key)
        if not context_allowed:
            return None

        return PendingEntry(
            pair=pair,
            direction=direction,
            signal_pos=pos,
            execute_pos=next_pos,
            planned_entry_time=entry_time,
            signal_atr=m5_atr_value,
            stop_distance=m5_atr_value * self.params.stop_atr_multiple,
            take_profit_price=target_price,
            take_profit_rr_cap=self.params.csr_take_profit_rr_cap,
            max_hold_bars=self.params.csr_max_hold_bars,
            context_key=context_key,
            context_trades_before=int(context_snapshot["trades"]),
            context_win_rate_before=float(context_snapshot["win_rate"]),
            context_expectancy_before=float(context_snapshot["expectancy_r"]),
        )

    def _build_session_trend_reclaim_entry(
        self,
        pair: str,
        direction: str,
        pos: int,
        prev_pos: int,
        next_pos: int,
        entry_time: pd.Timestamp,
    ) -> PendingEntry | None:
        entry_clock = entry_time.time()
        if not (time.fromisoformat(self.params.str_entry_start) <= entry_clock < time.fromisoformat(self.params.str_entry_end)):
            return None

        arrays = self.views[pair].arrays
        m5_atr_value = arrays["m5_atr"][pos]
        m15_atr_value = arrays["m15_atr"][pos]
        h1_atr_value = arrays["h1_atr"][pos]
        if (
            not np.isfinite(m5_atr_value)
            or not np.isfinite(m15_atr_value)
            or not np.isfinite(h1_atr_value)
            or m5_atr_value <= 0
            or m15_atr_value <= 0
            or h1_atr_value <= 0
        ):
            return None

        if not self._context_static_allows(pair, direction, pos, entry_time):
            return None

        h1_distance_atr = abs(arrays["close"][pos] - arrays["h1_ema_fast"][pos]) / h1_atr_value
        common_regime = (
            h1_distance_atr <= self.params.str_h1_distance_atr_max
            and arrays["m15_prev_range"][pos] <= self.params.m15_max_prev_range_atr * arrays["m15_prev_atr"][pos]
        )
        if not common_regime:
            return None

        if direction == "long":
            regime_ok = (
                arrays["h1_ema_fast"][pos] > arrays["h1_ema_slow"][pos]
                and arrays["h1_ema_slow_slope"][pos] > 0
                and arrays["h1_adx"][pos] >= self.params.str_h1_adx_min
                and arrays["m15_ema_slope"][pos] >= 0
            )
            setup_ok = (
                arrays["m15_low"][pos] <= arrays["m15_ema"][pos] + 0.10 * m15_atr_value
                and arrays["m15_close"][pos] >= arrays["m15_ema"][pos] - self.params.str_m15_pullback_atr * m15_atr_value
                and arrays["m15_rsi"][pos] >= self.params.str_m15_rsi_long_min
                and arrays["m5_rsi"][prev_pos] <= self.params.str_m5_reclaim_rsi_long
                and arrays["m5_rsi"][pos] > self.params.str_m5_reclaim_rsi_long
                and arrays["m5_ema_fast"][pos] >= arrays["m5_ema_slow"][pos]
                and arrays["close"][pos] > arrays["high"][prev_pos]
                and arrays["close"][pos] > arrays["open"][pos]
                and arrays["close"][pos] >= arrays["m5_ema_fast"][pos]
            )
        else:
            regime_ok = (
                arrays["h1_ema_fast"][pos] < arrays["h1_ema_slow"][pos]
                and arrays["h1_ema_slow_slope"][pos] < 0
                and arrays["h1_adx"][pos] >= self.params.str_h1_adx_min
                and arrays["m15_ema_slope"][pos] <= 0
            )
            setup_ok = (
                arrays["m15_high"][pos] >= arrays["m15_ema"][pos] - 0.10 * m15_atr_value
                and arrays["m15_close"][pos] <= arrays["m15_ema"][pos] + self.params.str_m15_pullback_atr * m15_atr_value
                and arrays["m15_rsi"][pos] <= self.params.str_m15_rsi_short_max
                and arrays["m5_rsi"][prev_pos] >= self.params.str_m5_reclaim_rsi_short
                and arrays["m5_rsi"][pos] < self.params.str_m5_reclaim_rsi_short
                and arrays["m5_ema_fast"][pos] <= arrays["m5_ema_slow"][pos]
                and arrays["close"][pos] < arrays["low"][prev_pos]
                and arrays["close"][pos] < arrays["open"][pos]
                and arrays["close"][pos] <= arrays["m5_ema_fast"][pos]
            )

        if not (regime_ok and setup_ok):
            return None

        context_key = self._build_context_key(pair, direction, pos, entry_time)
        context_allowed, context_snapshot = self._context_allows_trade(context_key)
        if not context_allowed:
            return None

        return PendingEntry(
            pair=pair,
            direction=direction,
            signal_pos=pos,
            execute_pos=next_pos,
            planned_entry_time=entry_time,
            signal_atr=m5_atr_value,
            stop_distance=m5_atr_value * self.params.stop_atr_multiple,
            take_profit_price=None,
            take_profit_rr_cap=self.params.str_take_profit_rr_cap,
            max_hold_bars=self.params.str_max_hold_bars,
            context_key=context_key,
            context_trades_before=int(context_snapshot["trades"]),
            context_win_rate_before=float(context_snapshot["win_rate"]),
            context_expectancy_before=float(context_snapshot["expectancy_r"]),
        )

    def _build_usdjpy_session_playbook_entry(
        self,
        pair: str,
        direction: str,
        pos: int,
        prev_pos: int,
        next_pos: int,
        entry_time: pd.Timestamp,
    ) -> PendingEntry | None:
        arrays = self.views[pair].arrays
        h1_atr_value = arrays["h1_atr"][pos]
        if not np.isfinite(h1_atr_value) or h1_atr_value <= 0:
            return None

        h1_distance_atr = abs(arrays["close"][pos] - arrays["h1_ema_slow"][pos]) / h1_atr_value
        adx_value = arrays["h1_adx"][pos]
        clock = entry_time.time()

        if (
            self.params.pb_enable_core_reversion
            and
            adx_value <= self.params.pb_balanced_adx_max
            and h1_distance_atr <= self.params.pb_h1_distance_atr_max
            and time.fromisoformat(self.params.pb_core_entry_start) <= clock < time.fromisoformat(self.params.pb_core_entry_end)
        ):
            entry = self._build_playbook_core_reversion(pair, direction, pos, prev_pos, next_pos, entry_time)
            if entry is not None:
                return entry

        if (
            self.params.pb_enable_compression_breakout
            and time.fromisoformat(self.params.pb_breakout_entry_start) <= clock < time.fromisoformat(self.params.pb_breakout_entry_end)
        ):
            entry = self._build_playbook_compression_breakout(pair, direction, pos, prev_pos, next_pos, entry_time)
            if entry is not None:
                return entry

        if (
            self.params.pb_enable_late_continuation
            and
            adx_value >= self.params.pb_active_adx_min
            and time.fromisoformat(self.params.pb_late_entry_start) <= clock < time.fromisoformat(self.params.pb_late_entry_end)
        ):
            entry = self._build_playbook_late_continuation(pair, direction, pos, prev_pos, next_pos, entry_time)
            if entry is not None:
                return entry

        return None

    def _build_playbook_entry_payload(
        self,
        pair: str,
        direction: str,
        pos: int,
        next_pos: int,
        entry_time: pd.Timestamp,
        signal_atr: float,
        setup_tag: str,
        take_profit_price: float | None,
    ) -> PendingEntry | None:
        if not self._context_static_allows(pair, direction, pos, entry_time):
            return None

        context_key = f"{setup_tag}|{self._build_context_key(pair, direction, pos, entry_time)}"
        context_allowed, context_snapshot = self._context_allows_trade(context_key)
        if not context_allowed:
            return None

        return PendingEntry(
            pair=pair,
            direction=direction,
            signal_pos=pos,
            execute_pos=next_pos,
            planned_entry_time=entry_time,
            signal_atr=signal_atr,
            stop_distance=signal_atr * self.params.stop_atr_multiple,
            take_profit_price=take_profit_price,
            take_profit_rr_cap=self.params.pb_take_profit_rr_cap,
            max_hold_bars=self.params.pb_max_hold_bars,
            context_key=context_key,
            setup_tag=setup_tag,
            context_trades_before=int(context_snapshot["trades"]),
            context_win_rate_before=float(context_snapshot["win_rate"]),
            context_expectancy_before=float(context_snapshot["expectancy_r"]),
        )

    def _build_playbook_core_reversion(
        self,
        pair: str,
        direction: str,
        pos: int,
        prev_pos: int,
        next_pos: int,
        entry_time: pd.Timestamp,
    ) -> PendingEntry | None:
        arrays = self.views[pair].arrays
        m5_atr_value = arrays["m5_atr"][pos]
        m15_atr_value = arrays["m15_atr"][pos]
        if not np.isfinite(m5_atr_value) or not np.isfinite(m15_atr_value) or m5_atr_value <= 0 or m15_atr_value <= 0:
            return None

        if direction == "long":
            setup_ok = (
                arrays["m15_close"][pos] <= arrays["m15_ema"][pos] - self.params.pb_core_extension_atr * m15_atr_value
                and arrays["m15_rsi"][pos] <= self.params.pb_core_rsi_long
                and arrays["m5_rsi"][prev_pos] <= self.params.pb_core_reclaim_rsi_long
                and arrays["m5_rsi"][pos] > self.params.pb_core_reclaim_rsi_long
                and arrays["close"][pos] > arrays["high"][prev_pos]
                and arrays["close"][pos] > arrays["open"][pos]
            )
            target_price = arrays["m15_ema"][pos]
        else:
            setup_ok = (
                arrays["m15_close"][pos] >= arrays["m15_ema"][pos] + self.params.pb_core_extension_atr * m15_atr_value
                and arrays["m15_rsi"][pos] >= self.params.pb_core_rsi_short
                and arrays["m5_rsi"][prev_pos] >= self.params.pb_core_reclaim_rsi_short
                and arrays["m5_rsi"][pos] < self.params.pb_core_reclaim_rsi_short
                and arrays["close"][pos] < arrays["low"][prev_pos]
                and arrays["close"][pos] < arrays["open"][pos]
            )
            target_price = arrays["m15_ema"][pos]

        if not setup_ok:
            return None
        return self._build_playbook_entry_payload(pair, direction, pos, next_pos, entry_time, m5_atr_value, "core_reversion", target_price)

    def _build_playbook_compression_breakout(
        self,
        pair: str,
        direction: str,
        pos: int,
        prev_pos: int,
        next_pos: int,
        entry_time: pd.Timestamp,
    ) -> PendingEntry | None:
        arrays = self.views[pair].arrays
        m5_atr_value = arrays["m5_atr"][pos]
        m15_atr_value = arrays["m15_atr"][pos]
        if not np.isfinite(m5_atr_value) or not np.isfinite(m15_atr_value) or m5_atr_value <= 0 or m15_atr_value <= 0:
            return None

        compression_ok = arrays["m15_prev_range"][pos] <= self.params.pb_breakout_compression_atr_max * arrays["m15_prev_atr"][pos]
        trigger_range_ok = arrays["m5_range"][pos] >= self.params.pb_breakout_trigger_range_atr_min * m5_atr_value
        if not (compression_ok and trigger_range_ok):
            return None

        if direction == "long":
            setup_ok = (
                arrays["close"][pos] > arrays["m5_roll_high_3"][pos]
                and arrays["m5_ema_fast"][pos] >= arrays["m5_ema_slow"][pos]
                and arrays["m5_rsi"][pos] >= self.params.pb_breakout_rsi_long
                and arrays["close"][pos] > arrays["open"][pos]
            )
        else:
            setup_ok = (
                arrays["close"][pos] < arrays["m5_roll_low_3"][pos]
                and arrays["m5_ema_fast"][pos] <= arrays["m5_ema_slow"][pos]
                and arrays["m5_rsi"][pos] <= self.params.pb_breakout_rsi_short
                and arrays["close"][pos] < arrays["open"][pos]
            )

        if not setup_ok:
            return None
        return self._build_playbook_entry_payload(pair, direction, pos, next_pos, entry_time, m5_atr_value, "compression_breakout", None)

    def _build_playbook_late_continuation(
        self,
        pair: str,
        direction: str,
        pos: int,
        prev_pos: int,
        next_pos: int,
        entry_time: pd.Timestamp,
    ) -> PendingEntry | None:
        arrays = self.views[pair].arrays
        m5_atr_value = arrays["m5_atr"][pos]
        if not np.isfinite(m5_atr_value) or m5_atr_value <= 0:
            return None

        if direction == "long":
            setup_ok = (
                arrays["h1_ema_fast"][pos] > arrays["h1_ema_slow"][pos]
                and arrays["h1_ema_slow_slope"][pos] > 0
                and arrays["m15_close"][pos] >= arrays["m15_ema"][pos]
                and arrays["m15_ema_slope"][pos] >= 0
                and arrays["m5_rsi"][prev_pos] <= self.params.pb_late_reclaim_rsi_long
                and arrays["m5_rsi"][pos] > self.params.pb_late_reclaim_rsi_long
                and arrays["close"][pos] > arrays["high"][prev_pos]
                and arrays["m5_ema_fast"][pos] >= arrays["m5_ema_slow"][pos]
            )
        else:
            setup_ok = (
                arrays["h1_ema_fast"][pos] < arrays["h1_ema_slow"][pos]
                and arrays["h1_ema_slow_slope"][pos] < 0
                and arrays["m15_close"][pos] <= arrays["m15_ema"][pos]
                and arrays["m15_ema_slope"][pos] <= 0
                and arrays["m5_rsi"][prev_pos] >= self.params.pb_late_reclaim_rsi_short
                and arrays["m5_rsi"][pos] < self.params.pb_late_reclaim_rsi_short
                and arrays["close"][pos] < arrays["low"][prev_pos]
                and arrays["m5_ema_fast"][pos] <= arrays["m5_ema_slow"][pos]
            )

        if not setup_ok:
            return None
        return self._build_playbook_entry_payload(pair, direction, pos, next_pos, entry_time, m5_atr_value, "late_continuation", None)

    def _execute_pending_entry(self, pair: str, pending: PendingEntry, pos: int) -> None:
        if pair in self.positions or self._blocked_by_correlation(pair, pending.direction):
            return

        view = self.views[pair]
        open_price = view.arrays["open"][pos]
        adjustment = self._execution_price_adjustment(pair, self.master_index[pos])
        entry_price = open_price + adjustment if pending.direction == "long" else open_price - adjustment
        stop_distance = pending.stop_distance if pending.stop_distance is not None else pending.signal_atr * self.params.stop_atr_multiple
        if not np.isfinite(stop_distance) or stop_distance <= 0:
            return

        stop_price = entry_price - stop_distance if pending.direction == "long" else entry_price + stop_distance
        risk_distance = abs(entry_price - stop_price)
        if pending.take_profit_price is not None:
            take_profit_price = pending.take_profit_price
            valid_tp = (
                (pending.direction == "long" and take_profit_price > entry_price)
                or (pending.direction == "short" and take_profit_price < entry_price)
            )
            if not valid_tp:
                return
        else:
            take_profit_price = (
                entry_price + risk_distance * self.params.take_profit_rr
                if pending.direction == "long"
                else entry_price - risk_distance * self.params.take_profit_rr
            )

        if pending.take_profit_rr_cap is not None:
            rr_cap_price = (
                entry_price + risk_distance * pending.take_profit_rr_cap
                if pending.direction == "long"
                else entry_price - risk_distance * pending.take_profit_rr_cap
            )
            if pending.direction == "long":
                take_profit_price = min(take_profit_price, rr_cap_price)
            else:
                take_profit_price = max(take_profit_price, rr_cap_price)

        equity = self._portfolio_equity(max(pending.signal_pos, 0))
        if self.broker.risk_budget_mode == "initial_capital":
            risk_budget_base = self.broker.initial_capital
        else:
            risk_budget_base = equity
        risk_budget = risk_budget_base * self.broker.risk_fraction
        quote_to_usd = self._quote_to_usd_rate(pair, pending.signal_pos)
        base_to_usd = self._base_to_usd_rate(pair, pending.signal_pos, entry_price)
        if not np.isfinite(quote_to_usd) or not np.isfinite(base_to_usd):
            return

        risk_per_unit_usd = risk_distance * quote_to_usd
        raw_units = risk_budget / risk_per_unit_usd if risk_per_unit_usd > 0 else 0.0
        max_units = (equity * self.broker.max_leverage) / max(base_to_usd, 1e-9)
        units = min(raw_units, max_units)
        units = math.floor(units / self.broker.lot_step) * self.broker.lot_step
        if units < self.broker.lot_step:
            return

        entry_notional_usd = units * base_to_usd
        entry_commission_usd = entry_notional_usd * self.broker.commission_rate
        self.cash -= entry_commission_usd

        self.positions[pair] = Position(
            pair=pair,
            direction=pending.direction,
            entry_time=pending.planned_entry_time,
            entry_price=entry_price,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            units=units,
            entry_commission_usd=entry_commission_usd,
            initial_risk_usd=units * risk_distance * quote_to_usd,
            signal_pos=pending.signal_pos,
            opened_on_pos=pos,
            max_hold_bars=pending.max_hold_bars,
            context_key=pending.context_key,
            setup_tag=pending.setup_tag,
            context_trades_before=pending.context_trades_before,
            context_win_rate_before=pending.context_win_rate_before,
            context_expectancy_before=pending.context_expectancy_before,
        )
        self.daily_direction_book[pair][pending.planned_entry_time.date()].add(pending.direction)

    def _process_position_bar(self, pair: str, pos: int) -> None:
        position = self.positions[pair]
        view = self.views[pair]
        low = view.arrays["low"][pos]
        high = view.arrays["high"][pos]
        close = view.arrays["close"][pos]
        session_time = self.master_index[pos].time()

        if self.news_masks[pair]["flatten"][pos]:
            self._close_position(pair, pos, close, "news_flatten")
            return
        if self.shock_masks[pair]["flatten"][pos]:
            self._close_position(pair, pos, close, "volatility_shock")
            return

        if position.max_hold_bars is not None and (pos - position.opened_on_pos) >= position.max_hold_bars:
            self._close_position(pair, pos, close, "time_stop")
            return

        if self.params.strategy_family in {"adaptive_session_reversion", "core_session_reversion"}:
            mean_price = view.arrays["m15_ema"][pos]
            if position.direction == "long" and close >= mean_price:
                self._close_position(pair, pos, close, "mean_reversion_exit")
                return
            if position.direction == "short" and close <= mean_price:
                self._close_position(pair, pos, close, "mean_reversion_exit")
                return

        if position.direction == "long":
            stop_hit = low <= position.stop_price
            tp_hit = high >= position.take_profit_price
        else:
            stop_hit = high >= position.stop_price
            tp_hit = low <= position.take_profit_price

        if stop_hit and tp_hit:
            self._close_position(pair, pos, position.stop_price, "stop_loss")
            return
        if stop_hit:
            self._close_position(pair, pos, position.stop_price, "stop_loss")
            return
        if tp_hit:
            self._close_position(pair, pos, position.take_profit_price, "take_profit")
            return
        if session_time >= self.broker.session_end_time:
            self._close_position(pair, pos, close, "session_close")

    def _close_position(self, pair: str, pos: int, raw_exit_price: float, exit_reason: str) -> None:
        position = self.positions.pop(pair, None)
        if position is None:
            return

        adjustment = self._execution_price_adjustment(pair, self.master_index[pos])
        exit_price = raw_exit_price - adjustment if position.direction == "long" else raw_exit_price + adjustment
        sign = 1 if position.direction == "long" else -1

        quote_to_usd = self._quote_to_usd_rate(pair, pos)
        base_to_usd = self._base_to_usd_rate(pair, pos, exit_price)
        gross_pnl_quote = sign * (exit_price - position.entry_price) * position.units
        gross_pnl_usd = gross_pnl_quote * quote_to_usd
        exit_notional_usd = position.units * base_to_usd
        exit_commission_usd = exit_notional_usd * self.broker.commission_rate

        self.cash += gross_pnl_usd - exit_commission_usd
        net_pnl_usd = gross_pnl_usd - position.entry_commission_usd - exit_commission_usd
        r_multiple = net_pnl_usd / position.initial_risk_usd if position.initial_risk_usd > 0 else np.nan
        self._update_context_stats(position.context_key, net_pnl_usd, r_multiple)

        self.trades.append(
            {
                "pair": pair,
                "direction": position.direction,
                "entry_time": position.entry_time.tz_convert(UTC),
                "exit_time": self.master_index[pos].tz_convert(UTC),
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "units": position.units,
                "gross_pnl_usd": gross_pnl_usd,
                "net_pnl_usd": net_pnl_usd,
                "r_multiple": r_multiple,
                "commission_usd": position.entry_commission_usd + exit_commission_usd,
                "exit_reason": exit_reason,
                "strategy_family": self.params.strategy_family,
                "context_key": position.context_key,
                "setup_tag": position.setup_tag,
                "context_trades_before": position.context_trades_before,
                "context_win_rate_before": position.context_win_rate_before,
                "context_expectancy_before": position.context_expectancy_before,
            }
        )

    def _portfolio_equity(self, pos: int) -> float:
        equity = self.cash
        for pair, position in self.positions.items():
            view = self.views[pair]
            asof_pos = view.asof_pos[pos]
            if asof_pos < 0:
                continue
            current_price = view.arrays["close"][asof_pos]
            sign = 1 if position.direction == "long" else -1
            gross_quote = sign * (current_price - position.entry_price) * position.units
            equity += gross_quote * self._quote_to_usd_rate(pair, asof_pos)
        return equity

    def _blocked_by_correlation(self, pair: str, direction: str) -> bool:
        for group in self.pair_to_groups.get(pair, []):
            for other_pair in group:
                if other_pair == pair:
                    continue
                other_position = self.positions.get(other_pair)
                if other_position and other_position.direction == direction:
                    return True
        return False

    def _build_context_components(self, pair: str, direction: str, pos: int, entry_time: pd.Timestamp) -> tuple[str, str, str, str, str]:
        arrays = self.views[pair].arrays
        weekday = entry_time.weekday()
        if weekday <= 1:
            weekday_bucket = "mon_tue"
        elif weekday <= 3:
            weekday_bucket = "wed_thu"
        else:
            weekday_bucket = "fri"

        minutes = entry_time.hour * 60 + entry_time.minute
        if minutes < 12 * 60 + 30:
            time_bucket = "early"
        elif minutes < 14 * 60 + 30:
            time_bucket = "core"
        else:
            time_bucket = "late"

        adx_value = arrays["h1_adx"][pos]
        if adx_value < 12:
            regime_bucket = "calm"
        elif adx_value < 20:
            regime_bucket = "balanced"
        else:
            regime_bucket = "active"

        extension_ratio = abs(arrays["m15_close"][pos] - arrays["m15_ema"][pos]) / max(arrays["m15_atr"][pos], 1e-9)
        extension_bucket = "mild" if extension_ratio < 0.7 else "deep"
        return weekday_bucket, time_bucket, regime_bucket, extension_bucket, direction

    @staticmethod
    def _parse_context_bucket_list(raw_value: str) -> set[str]:
        return {item.strip() for item in raw_value.split(",") if item.strip()}

    def _context_static_allows(self, pair: str, direction: str, pos: int, entry_time: pd.Timestamp) -> bool:
        weekday_bucket, time_bucket, regime_bucket, extension_bucket, direction_bucket = self._build_context_components(
            pair,
            direction,
            pos,
            entry_time,
        )
        filters = (
            (self.params.context_whitelist_weekdays, weekday_bucket),
            (self.params.context_whitelist_times, time_bucket),
            (self.params.context_whitelist_regimes, regime_bucket),
            (self.params.context_whitelist_extensions, extension_bucket),
            (self.params.context_whitelist_directions, direction_bucket),
        )
        for raw_filter, current_bucket in filters:
            if not raw_filter:
                continue
            allowed_buckets = self._parse_context_bucket_list(raw_filter)
            if current_bucket not in allowed_buckets:
                return False
        return True

    def _build_context_key(self, pair: str, direction: str, pos: int, entry_time: pd.Timestamp) -> str:
        context_components = self._build_context_components(pair, direction, pos, entry_time)
        return "|".join(context_components)

    def _context_allows_trade(self, context_key: str) -> tuple[bool, dict[str, float]]:
        stats = self.context_stats[context_key]
        trades = int(stats["trades"])
        wins = int(stats["wins"])
        win_rate = wins / trades if trades else 0.0
        expectancy_r = stats["r_sum"] / trades if trades else 0.0
        snapshot = {"trades": trades, "win_rate": win_rate, "expectancy_r": expectancy_r}

        if trades >= self.params.asr_context_min_samples_hard:
            allowed = (
                win_rate >= self.params.asr_context_min_win_rate_hard
                and expectancy_r >= self.params.asr_context_min_expectancy_hard
            )
            return allowed, snapshot

        if trades >= self.params.asr_context_min_samples_soft:
            allowed = (
                win_rate >= self.params.asr_context_min_win_rate_soft
                and expectancy_r >= self.params.asr_context_min_expectancy_soft
            )
            return allowed, snapshot

        return True, snapshot

    def _update_context_stats(self, context_key: str | None, net_pnl_usd: float, r_multiple: float) -> None:
        if not context_key:
            return
        stats = self.context_stats[context_key]
        stats["trades"] += 1
        stats["pnl_sum"] += float(net_pnl_usd)
        if np.isfinite(r_multiple):
            stats["r_sum"] += float(r_multiple)
        if net_pnl_usd > 0:
            stats["wins"] += 1
        elif net_pnl_usd < 0:
            stats["losses"] += 1

    def _execution_price_adjustment(self, pair: str, timestamp: pd.Timestamp) -> float:
        pip = PAIR_META[pair]["pip_size"]
        slippage_component = self.broker.slippage_pips * pip
        if not self.broker.use_spread_model:
            return slippage_component

        spread_pips = DEFAULT_SPREAD_PIPS.get(pair, 0.8)
        minutes_of_day = timestamp.hour * 60 + timestamp.minute
        spread_multiplier = 1.0
        if 16 * 60 + 55 <= minutes_of_day <= 17 * 60 + 10:
            spread_multiplier = 2.2
        elif 17 * 60 + 10 < minutes_of_day <= 18 * 60:
            spread_multiplier = 1.5

        half_spread_component = (spread_pips * spread_multiplier * 0.5) * pip
        return slippage_component + half_spread_component

    def _close_asof(self, pair: str, pos: int) -> float:
        view = self.views[pair]
        asof_pos = view.asof_pos[pos]
        if asof_pos < 0:
            return float("nan")
        return view.arrays["close"][asof_pos]

    def _quote_to_usd_rate(self, pair: str, pos: int) -> float:
        quote = PAIR_META[pair]["quote"]
        if quote == "USD":
            return 1.0

        direct = f"USD{quote}"
        inverse = f"{quote}USD"
        if direct in self.views:
            direct_price = self._close_asof(direct, pos)
            return 1.0 / direct_price if direct_price > 0 else float("nan")
        if inverse in self.views:
            return self._close_asof(inverse, pos)
        return float("nan")

    def _base_to_usd_rate(self, pair: str, pos: int, fallback_pair_price: float) -> float:
        base = PAIR_META[pair]["base"]
        quote = PAIR_META[pair]["quote"]
        if base == "USD":
            return 1.0
        if quote == "USD":
            return fallback_pair_price

        direct = f"{base}USD"
        inverse = f"USD{base}"
        if direct in self.views:
            return self._close_asof(direct, pos)
        if inverse in self.views:
            inverse_price = self._close_asof(inverse, pos)
            return 1.0 / inverse_price if inverse_price > 0 else float("nan")
        return float("nan")


def summarize_pair_stats(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=["pair", "trades", "wins", "losses", "win_rate_pct", "profit_factor", "net_pnl_usd", "avg_r_multiple"]
        )

    rows = []
    for pair, chunk in trades.groupby("pair"):
        wins = int((chunk["net_pnl_usd"] > 0).sum())
        losses = int((chunk["net_pnl_usd"] < 0).sum())
        gross_profit = chunk.loc[chunk["net_pnl_usd"] > 0, "net_pnl_usd"].sum()
        gross_loss = chunk.loc[chunk["net_pnl_usd"] < 0, "net_pnl_usd"].sum()
        profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else np.inf
        rows.append(
            {
                "pair": pair,
                "trades": len(chunk),
                "wins": wins,
                "losses": losses,
                "win_rate_pct": (wins / len(chunk)) * 100 if len(chunk) else 0.0,
                "profit_factor": profit_factor,
                "net_pnl_usd": chunk["net_pnl_usd"].sum(),
                "avg_r_multiple": chunk["r_multiple"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("pair").reset_index(drop=True)


def summarize_context_stats(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "context_key" not in trades.columns:
        return pd.DataFrame(columns=["context_key", "trades", "wins", "losses", "win_rate_pct", "profit_factor", "net_pnl_usd", "avg_r_multiple"])

    frame = trades.dropna(subset=["context_key"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=["context_key", "trades", "wins", "losses", "win_rate_pct", "profit_factor", "net_pnl_usd", "avg_r_multiple"])

    rows = []
    for context_key, chunk in frame.groupby("context_key"):
        wins = int((chunk["net_pnl_usd"] > 0).sum())
        losses = int((chunk["net_pnl_usd"] < 0).sum())
        gross_profit = chunk.loc[chunk["net_pnl_usd"] > 0, "net_pnl_usd"].sum()
        gross_loss = chunk.loc[chunk["net_pnl_usd"] < 0, "net_pnl_usd"].sum()
        profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else np.inf
        rows.append(
            {
                "context_key": context_key,
                "trades": len(chunk),
                "wins": wins,
                "losses": losses,
                "win_rate_pct": (wins / len(chunk)) * 100 if len(chunk) else 0.0,
                "profit_factor": profit_factor,
                "net_pnl_usd": chunk["net_pnl_usd"].sum(),
                "avg_r_multiple": chunk["r_multiple"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["net_pnl_usd", "trades"], ascending=[False, False]).reset_index(drop=True)


def summarize_setup_stats(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "setup_tag" not in trades.columns:
        return pd.DataFrame(columns=["setup_tag", "trades", "wins", "losses", "win_rate_pct", "profit_factor", "net_pnl_usd", "avg_r_multiple"])

    frame = trades.dropna(subset=["setup_tag"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=["setup_tag", "trades", "wins", "losses", "win_rate_pct", "profit_factor", "net_pnl_usd", "avg_r_multiple"])

    rows = []
    for setup_tag, chunk in frame.groupby("setup_tag"):
        wins = int((chunk["net_pnl_usd"] > 0).sum())
        losses = int((chunk["net_pnl_usd"] < 0).sum())
        gross_profit = chunk.loc[chunk["net_pnl_usd"] > 0, "net_pnl_usd"].sum()
        gross_loss = chunk.loc[chunk["net_pnl_usd"] < 0, "net_pnl_usd"].sum()
        profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else np.inf
        rows.append(
            {
                "setup_tag": setup_tag,
                "trades": len(chunk),
                "wins": wins,
                "losses": losses,
                "win_rate_pct": (wins / len(chunk)) * 100 if len(chunk) else 0.0,
                "profit_factor": profit_factor,
                "net_pnl_usd": chunk["net_pnl_usd"].sum(),
                "avg_r_multiple": chunk["r_multiple"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["net_pnl_usd", "trades"], ascending=[False, False]).reset_index(drop=True)


def summarize_period_stats(trades: pd.DataFrame, freq: str) -> pd.DataFrame:
    label = "year" if freq == "Y" else "month"
    if trades.empty:
        return pd.DataFrame(columns=["pair", label, "trades", "wins", "losses", "net_pnl_usd", "profit_factor"])

    frame = trades.copy()
    frame[label] = frame["entry_time"].dt.tz_localize(None).dt.to_period(freq).astype(str)
    rows = []
    for (pair, bucket), chunk in frame.groupby(["pair", label]):
        wins = int((chunk["net_pnl_usd"] > 0).sum())
        losses = int((chunk["net_pnl_usd"] < 0).sum())
        gross_profit = chunk.loc[chunk["net_pnl_usd"] > 0, "net_pnl_usd"].sum()
        gross_loss = chunk.loc[chunk["net_pnl_usd"] < 0, "net_pnl_usd"].sum()
        profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else np.inf
        rows.append(
            {
                "pair": pair,
                label: bucket,
                "trades": len(chunk),
                "wins": wins,
                "losses": losses,
                "net_pnl_usd": chunk["net_pnl_usd"].sum(),
                "profit_factor": profit_factor,
            }
    )
    return pd.DataFrame(rows).sort_values(["pair", label]).reset_index(drop=True)


def compute_trade_streaks(pnl_values: Iterable[float]) -> tuple[int, int]:
    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0

    for pnl in pnl_values:
        if pnl > 0:
            current_wins += 1
            current_losses = 0
        elif pnl < 0:
            current_losses += 1
            current_wins = 0
        else:
            current_wins = 0
            current_losses = 0
        max_wins = max(max_wins, current_wins)
        max_losses = max(max_losses, current_losses)

    return max_wins, max_losses


def summarize_portfolio(trades: pd.DataFrame, equity_curve: pd.DataFrame, initial_capital: float) -> dict[str, Any]:
    final_equity = float(equity_curve["equity"].iloc[-1]) if not equity_curve.empty else initial_capital
    total_return_pct = ((final_equity / initial_capital) - 1) * 100 if initial_capital else 0.0

    if equity_curve.empty:
        return {
            "initial_capital": initial_capital,
            "final_equity": final_equity,
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "break_even_trades": 0,
            "win_rate_pct": 0.0,
            "trades_per_month": 0.0,
            "profit_month_ratio": 0.0,
            "avg_monthly_net_pnl_usd": 0.0,
            "expectancy_usd": 0.0,
            "avg_trade_minutes": 0.0,
            "median_trade_minutes": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
        }

    equity = equity_curve.set_index("timestamp")["equity"].astype(float)
    peak = equity.cummax()
    drawdown = (equity - peak) / peak.replace(0.0, np.nan)
    max_drawdown_pct = abs(drawdown.min()) * 100 if len(drawdown) else 0.0

    daily_equity = equity.resample("1D").last().dropna()
    daily_returns = daily_equity.pct_change().dropna()
    sharpe_ratio = 0.0
    if len(daily_returns) > 1 and daily_returns.std(ddof=0) > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std(ddof=0)) * math.sqrt(252)

    wins = int((trades["net_pnl_usd"] > 0).sum()) if not trades.empty else 0
    losses = int((trades["net_pnl_usd"] < 0).sum()) if not trades.empty else 0
    gross_profit = trades.loc[trades["net_pnl_usd"] > 0, "net_pnl_usd"].sum() if not trades.empty else 0.0
    gross_loss = trades.loc[trades["net_pnl_usd"] < 0, "net_pnl_usd"].sum() if not trades.empty else 0.0
    profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else np.inf
    total_trades = len(trades)
    win_rate_pct = (wins / total_trades) * 100 if total_trades else 0.0
    break_even_trades = int((trades["net_pnl_usd"] == 0).sum()) if not trades.empty else 0

    monthly_totals = pd.Series(dtype=float)
    profit_month_ratio = 0.0
    avg_monthly_net_pnl_usd = 0.0
    trades_per_month = 0.0
    if not trades.empty:
        month_keys = trades["entry_time"].dt.tz_localize(None).dt.to_period("M").astype(str)
        monthly_totals = trades.groupby(month_keys)["net_pnl_usd"].sum()
        avg_monthly_net_pnl_usd = float(monthly_totals.mean()) if len(monthly_totals) else 0.0
        profit_month_ratio = float((monthly_totals > 0).mean()) if len(monthly_totals) else 0.0
        trades_per_month = total_trades / max(len(monthly_totals), 1)

    expectancy_usd = float(trades["net_pnl_usd"].mean()) if not trades.empty else 0.0
    trade_minutes = pd.Series(dtype=float)
    if not trades.empty:
        trade_minutes = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 60.0
    avg_trade_minutes = float(trade_minutes.mean()) if len(trade_minutes) else 0.0
    median_trade_minutes = float(trade_minutes.median()) if len(trade_minutes) else 0.0
    max_consecutive_wins, max_consecutive_losses = compute_trade_streaks(
        trades["net_pnl_usd"].tolist() if not trades.empty else []
    )

    return {
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe_ratio": sharpe_ratio,
        "profit_factor": profit_factor,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "break_even_trades": break_even_trades,
        "win_rate_pct": win_rate_pct,
        "trades_per_month": trades_per_month,
        "profit_month_ratio": profit_month_ratio,
        "avg_monthly_net_pnl_usd": avg_monthly_net_pnl_usd,
        "expectancy_usd": expectancy_usd,
        "avg_trade_minutes": avg_trade_minutes,
        "median_trade_minutes": median_trade_minutes,
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
    }


def compute_robustness_score(portfolio_summary: dict[str, Any], yearly_summary: pd.DataFrame, monthly_summary: pd.DataFrame) -> float:
    profitable_year_ratio = 0.0
    if not yearly_summary.empty:
        yearly_total = yearly_summary.groupby("year")["net_pnl_usd"].sum()
        profitable_year_ratio = float((yearly_total > 0).mean())

    negative_month_ratio = 0.0
    if not monthly_summary.empty:
        monthly_total = monthly_summary.groupby("month")["net_pnl_usd"].sum()
        negative_month_ratio = float((monthly_total < 0).mean())

    sharpe = max(float(portfolio_summary.get("sharpe_ratio", 0.0)), 0.0)
    profit_factor = min(float(portfolio_summary.get("profit_factor", 0.0)), 5.0)
    return_pct = max(float(portfolio_summary.get("total_return_pct", 0.0)), -100.0)
    max_dd = float(portfolio_summary.get("max_drawdown_pct", 0.0))

    score = ((1.0 + sharpe) * max(profit_factor, 0.1) * (1.0 + profitable_year_ratio))
    score *= (1.0 - negative_month_ratio * 0.5)
    score *= max(0.1, 1.0 + (return_pct / 100.0))
    score /= 1.0 + (max_dd / 100.0)
    return float(score)


def default_research_goals(profile: str = "consistency") -> ResearchGoals:
    if profile == "winrate":
        return ResearchGoals(
            target_trades_per_month=18.0,
            min_trades_per_month=10.0,
            target_win_rate_pct=60.0,
            min_win_rate_pct=52.0,
            target_return_pct=18.0,
            max_drawdown_pct=12.0,
            min_profit_factor=1.10,
            max_consecutive_losses=7,
        )
    if profile == "frequency":
        return ResearchGoals(
            target_trades_per_month=35.0,
            min_trades_per_month=20.0,
            target_win_rate_pct=52.0,
            min_win_rate_pct=46.0,
            target_return_pct=20.0,
            max_drawdown_pct=18.0,
            min_profit_factor=1.08,
            max_consecutive_losses=10,
        )
    return ResearchGoals()


def compute_goal_fit_score(
    portfolio_summary: dict[str, Any],
    yearly_summary: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    goals: ResearchGoals,
) -> float:
    profitable_year_ratio = 0.0
    if not yearly_summary.empty:
        yearly_total = yearly_summary.groupby("year")["net_pnl_usd"].sum()
        profitable_year_ratio = float((yearly_total > 0).mean())

    trades_per_month = float(portfolio_summary.get("trades_per_month", 0.0))
    win_rate = float(portfolio_summary.get("win_rate_pct", 0.0))
    total_return = float(portfolio_summary.get("total_return_pct", 0.0))
    max_dd = float(portfolio_summary.get("max_drawdown_pct", 0.0))
    profit_factor = float(portfolio_summary.get("profit_factor", 0.0))
    sharpe = float(portfolio_summary.get("sharpe_ratio", 0.0))
    profit_month_ratio = float(portfolio_summary.get("profit_month_ratio", 0.0))
    max_consecutive_losses = int(portfolio_summary.get("max_consecutive_losses", 0))

    score = 0.0
    score += min(trades_per_month / max(goals.target_trades_per_month, 1e-9), 1.35) * 18.0
    score += min(max(win_rate, 0.0) / max(goals.target_win_rate_pct, 1e-9), 1.25) * 28.0
    score += min(max(total_return, 0.0) / max(goals.target_return_pct, 1e-9), 1.50) * 22.0
    score += min(max(profit_factor, 0.0) / max(goals.min_profit_factor, 1e-9), 1.80) * 12.0
    score += min(max(profit_month_ratio, 0.0), 1.0) * 10.0
    score += min(max(profitable_year_ratio, 0.0), 1.0) * 8.0
    score += max(min(sharpe, 2.0), -1.0) * 2.5

    if trades_per_month < goals.min_trades_per_month:
        score -= (goals.min_trades_per_month - trades_per_month) * 2.5
    if win_rate < goals.min_win_rate_pct:
        score -= (goals.min_win_rate_pct - win_rate) * 1.6
    if max_dd > goals.max_drawdown_pct:
        score -= (max_dd - goals.max_drawdown_pct) * 2.2
    if profit_factor < goals.min_profit_factor:
        score -= (goals.min_profit_factor - profit_factor) * 18.0
    if max_consecutive_losses > goals.max_consecutive_losses:
        score -= (max_consecutive_losses - goals.max_consecutive_losses) * 2.5
    if total_return < 0:
        score -= abs(total_return) * 0.9

    return float(score)


def export_result(result: BacktestResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.trades.to_csv(output_dir / "trades.csv", index=False)
    result.equity_curve.to_csv(output_dir / "equity_curve.csv", index=False)
    result.data_quality.to_csv(output_dir / "data_quality.csv", index=False)
    result.news_events.to_csv(output_dir / "news_events.csv", index=False)
    result.context_summary.to_csv(output_dir / "context_summary.csv", index=False)
    result.setup_summary.to_csv(output_dir / "setup_summary.csv", index=False)
    result.pair_summary.to_csv(output_dir / "pair_summary.csv", index=False)
    result.yearly_summary.to_csv(output_dir / "yearly_summary.csv", index=False)
    result.monthly_summary.to_csv(output_dir / "monthly_summary.csv", index=False)

    payload = {
        "parameters": sanitize_for_json(result.parameters),
        "broker": sanitize_for_json(result.broker),
        "data_quality": sanitize_for_json(result.data_quality.to_dict(orient="records")),
        "news_events_count": int(len(result.news_events)),
        "portfolio_summary": sanitize_for_json(result.portfolio_summary),
        "robustness_score": sanitize_for_json(result.robustness_score),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if plt is not None and not result.equity_curve.empty:
        plot_equity_and_drawdown(result.equity_curve, output_dir / "equity_drawdown.png")
        plot_monthly_net(result.monthly_summary, output_dir / "monthly_net_pnl.png")
        plot_yearly_pair_net(result.yearly_summary, output_dir / "yearly_pair_pnl.png")


def plot_equity_and_drawdown(equity_curve: pd.DataFrame, target: Path) -> None:
    series = equity_curve.set_index("timestamp")["equity"].astype(float)
    peak = series.cummax()
    drawdown = (series - peak) / peak.replace(0.0, np.nan) * 100

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    axes[0].plot(series.index, series.values, color="#0f5a7a", linewidth=1.4)
    axes[0].set_title("Curva de capital")
    axes[0].set_ylabel("Equity (USD)")
    axes[0].grid(alpha=0.25)

    axes[1].fill_between(drawdown.index, drawdown.values, 0, color="#b03a2e", alpha=0.35)
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("%")
    axes[1].grid(alpha=0.25)
    fig.savefig(target, dpi=150)
    plt.close(fig)


def plot_monthly_net(monthly_summary: pd.DataFrame, target: Path) -> None:
    if monthly_summary.empty:
        return
    totals = monthly_summary.groupby("month")["net_pnl_usd"].sum()
    colors = np.where(totals >= 0, "#1b7f5b", "#b03a2e")
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ax.bar(totals.index.astype(str), totals.values, color=colors)
    ax.set_title("PNL neto mensual del portfolio")
    ax.set_ylabel("USD")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(target, dpi=150)
    plt.close(fig)


def plot_yearly_pair_net(yearly_summary: pd.DataFrame, target: Path) -> None:
    if yearly_summary.empty:
        return
    pivot = yearly_summary.pivot_table(index="year", columns="pair", values="net_pnl_usd", aggfunc="sum", fill_value=0.0)
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("PNL anual por par")
    ax.set_ylabel("USD")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(target, dpi=150)
    plt.close(fig)


def print_run_summary(result: BacktestResult) -> None:
    if not result.data_quality.empty:
        print("\n=== CALIDAD DE DATOS ===")
        print(result.data_quality.to_string(index=False))

    summary = result.portfolio_summary
    print("\n=== RESUMEN PORTFOLIO ===")
    for key in (
        "initial_capital",
        "final_equity",
        "total_return_pct",
        "max_drawdown_pct",
        "sharpe_ratio",
        "profit_factor",
        "total_trades",
        "trades_per_month",
        "wins",
        "losses",
        "break_even_trades",
        "win_rate_pct",
        "profit_month_ratio",
        "expectancy_usd",
        "avg_trade_minutes",
        "median_trade_minutes",
        "max_consecutive_wins",
        "max_consecutive_losses",
    ):
        value = summary[key]
        print(f"{key:>20}: {value:,.4f}" if isinstance(value, float) else f"{key:>20}: {value}")

    print(f"{'robustness_score':>20}: {result.robustness_score:,.4f}")
    if not result.pair_summary.empty:
        print("\n=== RESUMEN POR PAR ===")
        print(result.pair_summary.to_string(index=False))
    if not result.context_summary.empty:
        print("\n=== TOP CONTEXTOS ===")
        print(result.context_summary.head(15).to_string(index=False))
    if not result.setup_summary.empty:
        print("\n=== RESUMEN POR SETUP ===")
        print(result.setup_summary.to_string(index=False))
    if not result.yearly_summary.empty:
        print("\n=== ESTADISTICAS ANUALES ===")
        print(result.yearly_summary.to_string(index=False))
    if not result.monthly_summary.empty:
        print("\n=== ESTADISTICAS MENSUALES ===")
        print(result.monthly_summary.tail(36).to_string(index=False))


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, (np.floating, float)):
        if np.isfinite(value):
            return float(value)
        return "inf" if value > 0 else "-inf"
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def default_parameter_grid(
    strategy_family: str = "trend_pullback",
    profile: str = "consistency",
    params: StrategyParameters | None = None,
) -> dict[str, list[Any]]:
    if strategy_family == "session_trend_reclaim":
        if profile == "frequency":
            return {
                "str_h1_adx_min": [16.0, 20.0],
                "str_m15_pullback_atr": [0.15, 0.25, 0.35],
                "stop_atr_multiple": [0.6, 0.8],
                "str_take_profit_rr_cap": [0.6, 0.8],
            }
        if profile == "winrate":
            return {
                "str_h1_adx_min": [18.0, 22.0],
                "str_m15_pullback_atr": [0.05, 0.15, 0.25],
                "stop_atr_multiple": [0.6, 0.8],
                "str_take_profit_rr_cap": [0.6, 0.8],
            }
        return {
            "str_h1_adx_min": [16.0, 20.0],
            "str_m15_pullback_atr": [0.05, 0.15, 0.25],
            "stop_atr_multiple": [0.6, 0.8],
            "str_take_profit_rr_cap": [0.6, 0.8],
        }

    if strategy_family == "usdjp_session_playbook":
        if params is not None:
            enabled = [
                params.pb_enable_core_reversion,
                params.pb_enable_late_continuation,
                params.pb_enable_compression_breakout,
            ]
            if sum(bool(flag) for flag in enabled) == 1:
                if params.pb_enable_core_reversion:
                    if profile == "frequency":
                        return {
                            "pb_balanced_adx_max": [18.0, 20.0, 22.0],
                            "pb_core_extension_atr": [0.25, 0.35, 0.45],
                            "stop_atr_multiple": [0.6, 0.8],
                            "pb_take_profit_rr_cap": [0.6, 0.8],
                            "pb_max_hold_bars": [4, 6],
                        }
                    if profile == "winrate":
                        return {
                            "pb_balanced_adx_max": [14.0, 16.0, 18.0],
                            "pb_core_extension_atr": [0.35, 0.45, 0.55],
                            "stop_atr_multiple": [0.6, 0.8],
                            "pb_take_profit_rr_cap": [0.6, 0.8],
                            "pb_max_hold_bars": [4, 6],
                        }
                    return {
                        "pb_balanced_adx_max": [16.0, 18.0, 20.0],
                        "pb_core_extension_atr": [0.30, 0.40, 0.50],
                        "stop_atr_multiple": [0.6, 0.8],
                        "pb_take_profit_rr_cap": [0.6, 0.8],
                        "pb_max_hold_bars": [4, 6],
                    }
                if params.pb_enable_late_continuation:
                    if profile == "frequency":
                        return {
                            "pb_active_adx_min": [20.0, 22.0, 24.0],
                            "pb_late_reclaim_rsi_long": [50.0, 52.0, 54.0],
                            "pb_late_reclaim_rsi_short": [50.0, 48.0, 46.0],
                            "stop_atr_multiple": [0.6, 0.8],
                            "pb_take_profit_rr_cap": [0.6, 0.8],
                            "pb_max_hold_bars": [4, 6],
                        }
                    if profile == "winrate":
                        return {
                            "pb_active_adx_min": [24.0, 28.0, 32.0],
                            "pb_late_reclaim_rsi_long": [54.0, 56.0, 58.0],
                            "pb_late_reclaim_rsi_short": [46.0, 44.0, 42.0],
                            "stop_atr_multiple": [0.6, 0.8],
                            "pb_take_profit_rr_cap": [0.6, 0.8],
                            "pb_max_hold_bars": [4, 6],
                        }
                    return {
                        "pb_active_adx_min": [22.0, 26.0, 30.0],
                        "pb_late_reclaim_rsi_long": [52.0, 54.0, 56.0],
                        "pb_late_reclaim_rsi_short": [48.0, 46.0, 44.0],
                        "stop_atr_multiple": [0.6, 0.8],
                        "pb_take_profit_rr_cap": [0.6, 0.8],
                        "pb_max_hold_bars": [4, 6],
                    }
                if params.pb_enable_compression_breakout:
                    if profile == "frequency":
                        return {
                            "pb_breakout_compression_atr_max": [0.45, 0.55, 0.65],
                            "pb_breakout_trigger_range_atr_min": [0.50, 0.70, 0.90],
                            "stop_atr_multiple": [0.6, 0.8],
                            "pb_take_profit_rr_cap": [0.4, 0.6],
                            "pb_max_hold_bars": [3, 5],
                        }
                    if profile == "winrate":
                        return {
                            "pb_breakout_compression_atr_max": [0.30, 0.35, 0.45],
                            "pb_breakout_trigger_range_atr_min": [0.80, 1.00, 1.20],
                            "stop_atr_multiple": [0.6, 0.8],
                            "pb_take_profit_rr_cap": [0.4, 0.6],
                            "pb_max_hold_bars": [3, 5],
                        }
                    return {
                        "pb_breakout_compression_atr_max": [0.35, 0.45, 0.55],
                        "pb_breakout_trigger_range_atr_min": [0.60, 0.80, 1.00],
                        "stop_atr_multiple": [0.6, 0.8],
                        "pb_take_profit_rr_cap": [0.4, 0.6],
                        "pb_max_hold_bars": [3, 5],
                    }
        if profile == "winrate":
            return {
                "pb_balanced_adx_max": [18.0, 20.0],
                "pb_active_adx_min": [22.0, 26.0],
                "pb_core_extension_atr": [0.35, 0.45],
                "pb_breakout_compression_atr_max": [0.45, 0.60],
                "pb_breakout_trigger_range_atr_min": [0.50, 0.70],
                "stop_atr_multiple": [0.6, 0.8],
                "pb_take_profit_rr_cap": [0.6, 0.8],
                "pb_max_hold_bars": [4, 6],
            }
        if profile == "frequency":
            return {
                "pb_balanced_adx_max": [20.0, 22.0],
                "pb_active_adx_min": [20.0, 24.0],
                "pb_core_extension_atr": [0.30, 0.40],
                "pb_breakout_compression_atr_max": [0.55, 0.70],
                "pb_breakout_trigger_range_atr_min": [0.40, 0.60],
                "stop_atr_multiple": [0.6, 0.8],
                "pb_take_profit_rr_cap": [0.6, 0.8],
                "pb_max_hold_bars": [4, 6],
            }
        return {
            "pb_balanced_adx_max": [18.0, 20.0],
            "pb_active_adx_min": [22.0, 26.0],
            "pb_core_extension_atr": [0.35, 0.45],
            "pb_breakout_compression_atr_max": [0.45, 0.60],
            "pb_breakout_trigger_range_atr_min": [0.50, 0.70],
            "stop_atr_multiple": [0.6, 0.8],
            "pb_take_profit_rr_cap": [0.6, 0.8],
            "pb_max_hold_bars": [4, 6],
        }

    if strategy_family == "core_session_reversion":
        if profile == "winrate":
            return {
                "csr_h1_adx_max": [16.0, 20.0],
                "csr_m15_extension_atr": [0.55, 0.70],
                "csr_reversal_body_atr_min": [0.10, 0.20],
                "stop_atr_multiple": [0.6, 0.8],
                "csr_take_profit_rr_cap": [0.4, 0.6, 0.8],
                "csr_max_hold_bars": [3, 5],
            }
        if profile == "frequency":
            return {
                "csr_h1_adx_max": [18.0, 22.0],
                "csr_m15_extension_atr": [0.40, 0.55],
                "csr_reversal_body_atr_min": [0.05, 0.10],
                "stop_atr_multiple": [0.6, 0.8],
                "csr_take_profit_rr_cap": [0.4, 0.6],
                "csr_max_hold_bars": [3, 5],
            }
        return {
            "csr_h1_adx_max": [16.0, 20.0],
            "csr_m15_extension_atr": [0.45, 0.60],
            "csr_reversal_body_atr_min": [0.10, 0.20],
            "stop_atr_multiple": [0.6, 0.8],
            "csr_take_profit_rr_cap": [0.4, 0.6, 0.8],
            "csr_max_hold_bars": [3, 5],
        }

    if strategy_family == "adaptive_session_reversion":
        if profile == "winrate":
            return {
                "h1_ema_fast": [21, 34, 50],
                "h1_ema_slow": [200],
                "asr_h1_adx_max": [18.0, 22.0, 26.0],
                "asr_h1_distance_atr_max": [0.8, 1.0, 1.2],
                "m15_ema_period": [10, 14, 20],
                "asr_m15_extension_atr": [0.35, 0.45, 0.60],
                "asr_m15_rsi_long": [34.0, 38.0, 42.0],
                "asr_m15_rsi_short": [66.0, 62.0, 58.0],
                "asr_m5_reclaim_rsi_long": [38.0, 42.0, 45.0],
                "asr_m5_reclaim_rsi_short": [62.0, 58.0, 55.0],
                "stop_atr_multiple": [0.8, 1.0],
                "asr_take_profit_rr_cap": [0.5, 0.7, 0.9],
                "asr_max_hold_bars": [4, 6, 9],
            }
        if profile == "frequency":
            return {
                "h1_ema_fast": [21, 34],
                "h1_ema_slow": [200],
                "asr_h1_adx_max": [22.0, 26.0, 30.0],
                "asr_h1_distance_atr_max": [1.0, 1.2, 1.5],
                "m15_ema_period": [10, 14],
                "asr_m15_extension_atr": [0.25, 0.35, 0.45],
                "asr_m15_rsi_long": [38.0, 42.0, 45.0],
                "asr_m15_rsi_short": [62.0, 58.0, 55.0],
                "asr_m5_reclaim_rsi_long": [40.0, 42.0, 45.0],
                "asr_m5_reclaim_rsi_short": [60.0, 58.0, 55.0],
                "stop_atr_multiple": [0.8, 1.0],
                "asr_take_profit_rr_cap": [0.5, 0.7],
                "asr_max_hold_bars": [4, 6],
            }
        return {
            "h1_ema_fast": [21, 34, 50],
            "h1_ema_slow": [200],
            "asr_h1_adx_max": [20.0, 24.0, 28.0],
            "asr_h1_distance_atr_max": [0.8, 1.0, 1.2],
            "m15_ema_period": [10, 14, 20],
            "asr_m15_extension_atr": [0.30, 0.45, 0.60],
            "asr_m15_rsi_long": [34.0, 38.0, 42.0],
            "asr_m15_rsi_short": [66.0, 62.0, 58.0],
            "asr_m5_reclaim_rsi_long": [38.0, 42.0, 45.0],
            "asr_m5_reclaim_rsi_short": [62.0, 58.0, 55.0],
            "stop_atr_multiple": [0.8, 1.0],
            "asr_take_profit_rr_cap": [0.5, 0.7, 0.9],
            "asr_max_hold_bars": [4, 6, 9],
        }

    if strategy_family == "session_mean_reversion":
        if profile == "winrate":
            return {
                "h1_ema_fast": [34, 50],
                "h1_ema_slow": [200],
                "mr_h1_adx_max": [18.0, 22.0, 28.0],
                "m15_ema_period": [10, 14, 20],
                "mr_m15_extension_atr": [0.5, 0.7, 0.9],
                "mr_m15_rsi_long": [32.0, 35.0, 38.0],
                "mr_m15_rsi_short": [68.0, 65.0, 62.0],
                "mr_m5_reclaim_rsi_long": [35.0, 40.0, 45.0],
                "mr_m5_reclaim_rsi_short": [65.0, 60.0, 55.0],
                "stop_atr_multiple": [0.8, 1.0],
                "mr_target_atr_buffer": [0.05, 0.10, 0.15],
            }
        if profile == "frequency":
            return {
                "h1_ema_fast": [21, 34, 50],
                "h1_ema_slow": [200],
                "mr_h1_adx_max": [22.0, 28.0, 35.0],
                "m15_ema_period": [10, 14],
                "mr_m15_extension_atr": [0.4, 0.5, 0.7],
                "mr_m15_rsi_long": [35.0, 38.0, 42.0],
                "mr_m15_rsi_short": [65.0, 62.0, 58.0],
                "mr_m5_reclaim_rsi_long": [38.0, 42.0, 45.0],
                "mr_m5_reclaim_rsi_short": [62.0, 58.0, 55.0],
                "stop_atr_multiple": [0.8, 1.0],
                "mr_target_atr_buffer": [0.10, 0.15],
            }
        return {
            "h1_ema_fast": [34, 50],
            "h1_ema_slow": [200],
            "mr_h1_adx_max": [20.0, 24.0, 28.0],
            "m15_ema_period": [10, 14, 20],
            "mr_m15_extension_atr": [0.5, 0.7, 0.9],
            "mr_m15_rsi_long": [32.0, 35.0, 38.0],
            "mr_m15_rsi_short": [68.0, 65.0, 62.0],
            "mr_m5_reclaim_rsi_long": [35.0, 40.0, 45.0],
            "mr_m5_reclaim_rsi_short": [65.0, 60.0, 55.0],
            "stop_atr_multiple": [0.8, 1.0, 1.2],
            "mr_target_atr_buffer": [0.05, 0.10, 0.15],
        }

    if profile == "winrate":
        return {
            "h1_adx_threshold": [16.0, 18.0, 20.0],
            "h1_ema_fast": [34, 50],
            "h1_ema_slow": [200],
            "m15_ema_period": [14, 20],
            "m15_rsi_pullback_long": [45.0, 48.0],
            "m15_rsi_pullback_short": [55.0, 52.0],
            "m5_trigger_rsi_midpoint": [45.0, 50.0],
            "stop_atr_multiple": [0.8, 1.0, 1.2],
            "take_profit_rr": [0.6, 0.8, 1.0, 1.2],
        }
    if profile == "frequency":
        return {
            "h1_adx_threshold": [12.0, 16.0, 18.0],
            "h1_ema_fast": [21, 34, 50],
            "h1_ema_slow": [200],
            "m15_ema_period": [10, 14, 20],
            "m15_rsi_pullback_long": [45.0, 48.0, 50.0],
            "m15_rsi_pullback_short": [55.0, 52.0, 50.0],
            "m5_trigger_rsi_midpoint": [45.0, 50.0],
            "stop_atr_multiple": [0.8, 1.0, 1.2],
            "take_profit_rr": [0.8, 1.0, 1.2],
        }
    return {
        "h1_adx_threshold": [16.0, 18.0, 20.0],
        "h1_ema_fast": [34, 50],
        "h1_ema_slow": [200],
        "m15_ema_period": [14, 20, 34],
        "m15_rsi_pullback_long": [42.0, 45.0, 48.0],
        "m15_rsi_pullback_short": [58.0, 55.0, 52.0],
        "m5_trigger_rsi_midpoint": [45.0, 50.0, 55.0],
        "stop_atr_multiple": [0.8, 1.0, 1.2],
        "take_profit_rr": [0.8, 1.0, 1.2, 1.5],
    }


def iter_grid(grid: dict[str, list[Any]]) -> Iterable[dict[str, Any]]:
    keys = list(grid)
    for values in itertools.product(*(grid[key] for key in keys)):
        yield dict(zip(keys, values))


def build_validation_windows(start: str, end: str) -> list[tuple[str, str, str]]:
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    windows: list[tuple[str, str, str]] = []

    total_days = (end_date - start_date).days + 1
    if total_days <= 370:
        quarter_starts = [date(start_date.year, 1, 1), date(start_date.year, 4, 1), date(start_date.year, 7, 1), date(start_date.year, 10, 1)]
        for q_index, q_start in enumerate(quarter_starts, start=1):
            if q_start.month == 10:
                q_end = date(q_start.year, 12, 31)
            else:
                next_q = quarter_starts[q_index] if q_index < len(quarter_starts) else date(q_start.year + 1, 1, 1)
                q_end = next_q - timedelta(days=1)
            window_start = max(start_date, q_start)
            window_end = min(end_date, q_end)
            if window_start <= window_end:
                windows.append((f"{window_start.year}-Q{q_index}", window_start.isoformat(), window_end.isoformat()))
        return windows

    for year in range(start_date.year, end_date.year + 1):
        window_start = max(start_date, date(year, 1, 1))
        window_end = min(end_date, date(year, 12, 31))
        if window_start <= window_end:
            windows.append((str(year), window_start.isoformat(), window_end.isoformat()))
    return windows


def run_backtest(
    run_config: RunConfig,
    params: StrategyParameters,
    broker: BrokerConfig,
    raw_bundle: dict[str, pd.DataFrame] | None = None,
    news_events: pd.DataFrame | None = None,
) -> BacktestResult:
    prepared: dict[str, pd.DataFrame] = {}
    source_bundle = raw_bundle or load_raw_bundle(run_config)
    active_news_events = news_events.copy() if news_events is not None else load_news_events(run_config)
    active_raw_bundle = {
        pair: clip_to_run_window(source_bundle[pair], run_config)
        for pair in run_config.pairs
    }
    data_quality = validate_data_bundle(active_raw_bundle)
    if run_config.strict_data_quality and (data_quality["quality_status"] == "fail").any():
        failed_pairs = data_quality.loc[data_quality["quality_status"] == "fail", "pair"].tolist()
        raise ValueError(f"Los datos fallaron la validacion de calidad para: {', '.join(failed_pairs)}")

    for pair in run_config.pairs:
        prepared[pair] = prepare_pair_features(active_raw_bundle[pair], params)

    master_index, views = build_pair_data_views(prepared, run_config.pairs)
    news_masks = build_pair_news_masks(master_index, run_config.pairs, active_news_events, run_config)
    shock_masks = build_pair_shock_masks(master_index, views, run_config)
    engine = MultiTimeframeFXBacktester(
        views=views,
        master_index=master_index,
        params=params,
        broker=broker,
        pairs=run_config.pairs,
        correlation_groups=run_config.correlation_groups,
        news_masks=news_masks,
        shock_masks=shock_masks,
    )
    result = engine.run()
    result.data_quality = data_quality
    result.news_events = active_news_events
    return result


def run_grid_search(
    run_config: RunConfig,
    base_params: StrategyParameters,
    broker: BrokerConfig,
    optimization_profile: str = "consistency",
    max_combinations: int | None = None,
) -> tuple[BacktestResult, pd.DataFrame]:
    goals = default_research_goals(optimization_profile)
    grid = default_parameter_grid(base_params.strategy_family, optimization_profile, base_params)
    raw_bundle = load_raw_bundle(run_config)
    news_events = load_news_events(run_config)
    validation_windows = build_validation_windows(run_config.start, run_config.end)
    rows = []
    best_result: BacktestResult | None = None
    best_validation_score = -np.inf

    for index, values in enumerate(iter_grid(grid), start=1):
        if max_combinations is not None and index > max_combinations:
            break
        candidate = replace(base_params, **values)
        print(f"\n[opt] Ejecutando combinacion {index}: {values}")
        result = run_backtest(run_config, candidate, broker, raw_bundle=raw_bundle, news_events=news_events)
        full_sample_goal_score = compute_goal_fit_score(
            result.portfolio_summary,
            result.yearly_summary,
            result.monthly_summary,
            goals,
        )

        validation_scores = []
        validation_goal_scores = []
        validation_returns = []
        validation_sharpes = []
        validation_drawdowns = []
        profitable_windows = 0

        for label, window_start, window_end in validation_windows:
            window_config = replace(run_config, start=window_start, end=window_end)
            window_result = run_backtest(window_config, candidate, broker, raw_bundle=raw_bundle, news_events=news_events)
            validation_scores.append(window_result.robustness_score)
            validation_returns.append(window_result.portfolio_summary["total_return_pct"])
            validation_sharpes.append(window_result.portfolio_summary["sharpe_ratio"])
            validation_drawdowns.append(window_result.portfolio_summary["max_drawdown_pct"])
            validation_goal_scores.append(
                compute_goal_fit_score(
                    window_result.portfolio_summary,
                    window_result.yearly_summary,
                    window_result.monthly_summary,
                    goals,
                )
            )
            if window_result.portfolio_summary["total_return_pct"] > 0:
                profitable_windows += 1

        profitable_window_ratio = profitable_windows / len(validation_windows) if validation_windows else 0.0
        avg_validation_score = float(np.mean(validation_scores)) if validation_scores else 0.0
        avg_validation_goal_score = float(np.mean(validation_goal_scores)) if validation_goal_scores else 0.0
        avg_validation_return = float(np.mean(validation_returns)) if validation_returns else 0.0
        avg_validation_sharpe = float(np.mean(validation_sharpes)) if validation_sharpes else 0.0
        worst_validation_dd = float(max(validation_drawdowns)) if validation_drawdowns else 0.0
        validation_score = (
            avg_validation_goal_score
            * (1.0 + profitable_window_ratio)
            * max(0.1, 1.0 + (avg_validation_return / 100.0))
            / (1.0 + worst_validation_dd / 100.0)
        )

        rows.append(
            {
                **values,
                "validation_score": validation_score,
                "full_sample_goal_score": full_sample_goal_score,
                "avg_validation_goal_score": avg_validation_goal_score,
                "avg_validation_robustness": avg_validation_score,
                "avg_validation_return_pct": avg_validation_return,
                "avg_validation_sharpe": avg_validation_sharpe,
                "worst_validation_drawdown_pct": worst_validation_dd,
                "profitable_window_ratio": profitable_window_ratio,
                "full_sample_robustness": result.robustness_score,
                **result.portfolio_summary,
            }
        )

        if best_result is None or validation_score > best_validation_score:
            best_result = result
            best_validation_score = validation_score
            best_result.robustness_score = validation_score

    if best_result is None:
        raise RuntimeError("No se pudieron evaluar combinaciones del grid search.")

    ranking = pd.DataFrame(rows).sort_values(
        by=["validation_score", "avg_validation_goal_score", "avg_validation_sharpe", "total_return_pct"],
        ascending=[False, False, False, False],
    )
    return best_result, ranking.reset_index(drop=True)


def run_pair_screen(
    run_config: RunConfig,
    base_params: StrategyParameters,
    broker: BrokerConfig,
    optimization_profile: str = "consistency",
    max_combinations: int | None = None,
) -> tuple[pd.DataFrame, dict[str, BacktestResult]]:
    rows = []
    best_results: dict[str, BacktestResult] = {}

    for pair in run_config.pairs:
        single_pair_config = replace(
            run_config,
            pairs=(pair,),
            correlation_groups=tuple(),
        )
        print(f"\n[screen] Evaluando {pair}...")
        best_result, ranking = run_grid_search(
            run_config=single_pair_config,
            base_params=base_params,
            broker=broker,
            optimization_profile=optimization_profile,
            max_combinations=max_combinations,
        )
        best_results[pair] = best_result

        top_row = ranking.iloc[0].to_dict()
        rows.append(
            {
                "pair": pair,
                "strategy_family": base_params.strategy_family,
                **top_row,
            }
        )

    ranking_df = pd.DataFrame(rows).sort_values(
        by=[
            "validation_score",
            "avg_validation_goal_score",
            "profit_factor",
            "total_return_pct",
        ],
        ascending=[False, False, False, False],
    )
    return ranking_df.reset_index(drop=True), best_results


def serialize_run_config(run_config: RunConfig) -> dict[str, Any]:
    payload = asdict(run_config)
    payload["data_dir"] = str(run_config.data_dir)
    payload["report_dir"] = str(run_config.report_dir)
    payload["news_file"] = str(run_config.news_file) if run_config.news_file is not None else None
    payload["pairs"] = list(run_config.pairs)
    payload["correlation_groups"] = [list(group) for group in run_config.correlation_groups]
    return payload


def save_prepared_timeframes(run_config: RunConfig, params: StrategyParameters) -> None:
    output_dir = run_config.data_dir / "prepared"
    output_dir.mkdir(parents=True, exist_ok=True)

    for pair in run_config.pairs:
        raw = clip_to_run_window(load_pair_m5(pair, run_config), run_config)
        m5 = raw.copy()
        m15 = resample_ohlcv(m5, "15min")
        h1 = resample_ohlcv(m5, "1h")
        m5.to_csv(output_dir / f"{pair}_M5.csv")
        m15.to_csv(output_dir / f"{pair}_M15.csv")
        h1.to_csv(output_dir / f"{pair}_H1.csv")

    manifest = {
        "run_config": serialize_run_config(run_config),
        "strategy_parameters": asdict(params),
    }
    (output_dir / "prepared_data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def populate_cache(run_config: RunConfig) -> Path:
    cache_rows = []
    for pair in run_config.pairs:
        raw = load_pair_m5(pair, run_config)
        clipped = clip_to_run_window(raw, run_config)
        cache_rows.append(
            {
                "pair": pair,
                "start": str(clipped.index.min()),
                "end": str(clipped.index.max()),
                "bars": int(len(clipped)),
            }
        )

    manifest_path = run_config.data_dir / "cache_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(cache_rows, indent=2), encoding="utf-8")
    return manifest_path


def build_report_dir(base_dir: Path, suffix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = base_dir / f"{timestamp}_{suffix}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_pairs(pairs: Iterable[str]) -> tuple[str, ...]:
    normalized = []
    for pair in pairs:
        pair = pair.upper().strip()
        if pair not in PAIR_META:
            raise ValueError(f"Par no soportado: {pair}")
        normalized.append(pair)
    return tuple(normalized)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtester multi-timeframe para FX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Comandos recomendados:
              python fx_multi_timeframe_backtester.py fetch-news --news-source tradingeconomics
              python fx_multi_timeframe_backtester.py cache-data --download-missing
              python fx_multi_timeframe_backtester.py prepare-data --download-missing
              python fx_multi_timeframe_backtester.py run --pairs EURUSD GBPUSD USDJPY
              python fx_multi_timeframe_backtester.py screen-pairs --strategy-family adaptive_session_reversion
              python fx_multi_timeframe_backtester.py optimize --max-combinations 12
            """
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_flags(target: argparse.ArgumentParser) -> None:
        target.add_argument("--start", default="2020-01-01")
        target.add_argument("--end", default="2025-12-31")
        target.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS))
        target.add_argument("--data-dir", default="data")
        target.add_argument("--report-dir", default="reports")
        target.add_argument("--strategy-family", choices=["trend_pullback", "session_mean_reversion", "adaptive_session_reversion", "core_session_reversion", "session_trend_reclaim", "usdjp_session_playbook"], default="trend_pullback")
        target.add_argument("--source", choices=["auto", "local", "dukascopy"], default="auto")
        target.add_argument("--download-missing", action="store_true")
        target.add_argument("--force-download", action="store_true")
        target.add_argument("--synthetic", action="store_true")
        target.add_argument("--strict-data-quality", action="store_true")
        target.add_argument("--risk-pct", type=float, default=1.0, help="Riesgo por operacion en porcentaje.")
        target.add_argument("--risk-budget-mode", choices=["initial_capital", "equity"], default="initial_capital")
        target.add_argument("--initial-capital", type=float, default=100_000.0)
        target.add_argument("--commission-rate", type=float, default=0.00002)
        target.add_argument("--slippage-pips", type=float, default=0.2)
        target.add_argument("--disable-spread-model", action="store_true")
        target.add_argument("--max-leverage", type=float, default=20.0)
        target.add_argument("--lot-step", type=int, default=1000)
        target.add_argument("--news-source", choices=["none", "csv", "tradingeconomics"], default="none")
        target.add_argument("--news-file", default=None, help="CSV local con columnas datetime,currency,importance,event.")
        target.add_argument("--news-api-key", default="guest:guest")
        target.add_argument("--news-timezone", default="UTC")
        target.add_argument("--news-min-importance", type=int, default=3)
        target.add_argument("--news-no-entry-pre-minutes", type=int, default=45)
        target.add_argument("--news-no-entry-post-minutes", type=int, default=30)
        target.add_argument("--news-flatten-minutes-before", type=int, default=15)
        target.add_argument("--disable-hard-news-veto", action="store_true")
        target.add_argument("--news-hard-no-entry-pre-minutes", type=int, default=90)
        target.add_argument("--news-hard-no-entry-post-minutes", type=int, default=60)
        target.add_argument("--news-hard-flatten-minutes-before", type=int, default=30)
        target.add_argument("--news-hard-keywords", default=",".join(DEFAULT_HARD_NEWS_KEYWORDS))
        target.add_argument("--disable-shock-guard", action="store_true")
        target.add_argument("--shock-no-entry-atr-multiple", type=float, default=2.5)
        target.add_argument("--shock-flatten-atr-multiple", type=float, default=3.0)
        target.add_argument("--shock-cooldown-minutes", type=int, default=30)
        target.add_argument("--context-whitelist-weekdays", default="")
        target.add_argument("--context-whitelist-times", default="")
        target.add_argument("--context-whitelist-regimes", default="")
        target.add_argument("--context-whitelist-extensions", default="")
        target.add_argument("--context-whitelist-directions", default="")

    fetch_news_parser = subparsers.add_parser("fetch-news", help="Descarga o normaliza un calendario economico para usar en el news guard.")
    add_common_flags(fetch_news_parser)
    fetch_news_parser.add_argument("--news-output", default="data/news_events.csv")

    cache_parser = subparsers.add_parser("cache-data", help="Puebla y valida la cache M5 sin exportar M15/H1.")
    add_common_flags(cache_parser)

    prepare_parser = subparsers.add_parser("prepare-data", help="Carga o descarga datos y exporta M5/M15/H1.")
    add_common_flags(prepare_parser)

    run_parser = subparsers.add_parser("run", help="Ejecuta el backtest completo.")
    add_common_flags(run_parser)

    optimize_parser = subparsers.add_parser("optimize", help="Ejecuta un grid search sencillo.")
    add_common_flags(optimize_parser)
    optimize_parser.add_argument("--max-combinations", type=int, default=16)
    optimize_parser.add_argument("--optimization-profile", choices=["consistency", "winrate", "frequency"], default="consistency")

    screen_parser = subparsers.add_parser("screen-pairs", help="Optimiza cada par por separado y arma un ranking de robustez.")
    add_common_flags(screen_parser)
    screen_parser.add_argument("--max-combinations", type=int, default=16)
    screen_parser.add_argument("--optimization-profile", choices=["consistency", "winrate", "frequency"], default="consistency")

    validate_parser = subparsers.add_parser("validate-data", help="Valida calidad de datos M5 antes del backtest.")
    add_common_flags(validate_parser)

    return parser.parse_args()


def namespace_to_configs(args: argparse.Namespace) -> tuple[RunConfig, StrategyParameters, BrokerConfig]:
    pairs = validate_pairs(args.pairs)
    run_config = RunConfig(
        start=args.start,
        end=args.end,
        pairs=pairs,
        data_dir=Path(args.data_dir),
        report_dir=Path(args.report_dir),
        source=args.source,
        download_missing=args.download_missing,
        force_download=args.force_download,
        synthetic=args.synthetic,
        strict_data_quality=args.strict_data_quality,
        news_source=args.news_source,
        news_file=Path(args.news_file) if args.news_file else None,
        news_api_key=args.news_api_key,
        news_timezone=args.news_timezone,
        news_min_importance=args.news_min_importance,
        news_no_entry_pre_minutes=args.news_no_entry_pre_minutes,
        news_no_entry_post_minutes=args.news_no_entry_post_minutes,
        news_flatten_minutes_before=args.news_flatten_minutes_before,
        hard_news_veto_enabled=not args.disable_hard_news_veto,
        news_hard_no_entry_pre_minutes=args.news_hard_no_entry_pre_minutes,
        news_hard_no_entry_post_minutes=args.news_hard_no_entry_post_minutes,
        news_hard_flatten_minutes_before=args.news_hard_flatten_minutes_before,
        news_hard_keywords=normalize_news_keywords(args.news_hard_keywords),
        shock_guard_enabled=not args.disable_shock_guard,
        shock_no_entry_atr_multiple=args.shock_no_entry_atr_multiple,
        shock_flatten_atr_multiple=args.shock_flatten_atr_multiple,
        shock_cooldown_minutes=args.shock_cooldown_minutes,
    )
    params = StrategyParameters(
        strategy_family=args.strategy_family,
        context_whitelist_weekdays=args.context_whitelist_weekdays,
        context_whitelist_times=args.context_whitelist_times,
        context_whitelist_regimes=args.context_whitelist_regimes,
        context_whitelist_extensions=args.context_whitelist_extensions,
        context_whitelist_directions=args.context_whitelist_directions,
    )
    broker = BrokerConfig(
        initial_capital=args.initial_capital,
        risk_fraction=args.risk_pct / 100.0,
        risk_budget_mode=args.risk_budget_mode,
        commission_rate=args.commission_rate,
        slippage_pips=args.slippage_pips,
        use_spread_model=not args.disable_spread_model,
        max_leverage=args.max_leverage,
        lot_step=args.lot_step,
    )
    return run_config, params, broker


def main() -> None:
    args = parse_arguments()
    run_config, params, broker = namespace_to_configs(args)

    if args.command == "fetch-news":
        news_events = load_news_events(run_config)
        output_path = Path(args.news_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        news_events.to_csv(output_path, index=False)
        print(f"Eventos economicos exportados en: {output_path}")
        print(f"Eventos cargados: {len(news_events)}")
        return

    if args.command == "cache-data":
        manifest_path = populate_cache(run_config)
        print(f"Cache poblada y manifest exportado en: {manifest_path}")
        return

    if args.command == "prepare-data":
        save_prepared_timeframes(run_config, params)
        print(f"Datos preparados en: {run_config.data_dir / 'prepared'}")
        return

    if args.command == "validate-data":
        raw_bundle = load_raw_bundle(run_config)
        data_quality = validate_data_bundle(raw_bundle)
        report_dir = build_report_dir(run_config.report_dir, "validate")
        data_quality.to_csv(report_dir / "data_quality.csv", index=False)
        print("\n=== CALIDAD DE DATOS ===")
        print(data_quality.to_string(index=False))
        print(f"\nReporte exportado en: {report_dir}")
        return

    if args.command == "run":
        result = run_backtest(run_config, params, broker)
        report_dir = build_report_dir(run_config.report_dir, "run")
        export_result(result, report_dir)
        print_run_summary(result)
        print(f"\nReportes exportados en: {report_dir}")
        return

    if args.command == "optimize":
        best_result, ranking = run_grid_search(
            run_config=run_config,
            base_params=params,
            broker=broker,
            optimization_profile=args.optimization_profile,
            max_combinations=args.max_combinations,
        )
        report_dir = build_report_dir(run_config.report_dir, "optimize")
        export_result(best_result, report_dir)
        ranking.to_csv(report_dir / "optimization_ranking.csv", index=False)
        print("\n=== TOP PARAMETROS ===")
        print(ranking.head(10).to_string(index=False))
        print("\n=== MEJOR CONFIGURACION ===")
        print_run_summary(best_result)
        print(f"\nReportes exportados en: {report_dir}")
        return

    if args.command == "screen-pairs":
        ranking, best_results = run_pair_screen(
            run_config=run_config,
            base_params=params,
            broker=broker,
            optimization_profile=args.optimization_profile,
            max_combinations=args.max_combinations,
        )
        report_dir = build_report_dir(run_config.report_dir, "screen_pairs")
        ranking.to_csv(report_dir / "pair_screen_ranking.csv", index=False)
        for pair, result in best_results.items():
            pair_dir = report_dir / pair
            pair_dir.mkdir(parents=True, exist_ok=True)
            export_result(result, pair_dir)
        print("\n=== RANKING POR PAR ===")
        print(ranking.to_string(index=False))
        print(f"\nReportes exportados en: {report_dir}")
        return

    raise ValueError(f"Comando no soportado: {args.command}")


if __name__ == "__main__":
    main()
