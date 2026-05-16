from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from research_lab.config import DEFAULT_HIGH_PRECISION_PREPARED_DIR, NY_TZ

warnings.simplefilter("ignore", PerformanceWarning)

FX_REOPEN_MINUTE_NY = 17 * 60
FX_CLOSE_MINUTE_NY = 17 * 60
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
SUPPORTED_PREPARED_TIMEFRAMES = ("M1", "M5", "M15", "H1")
TIMEFRAME_RESAMPLE_RULES: dict[str, str] = {
    "M1": "1min",
    "M3": "3min",
    "M5": "5min",
    "M15": "15min",
    "H1": "1h",
}


@dataclass(frozen=True)
class PreparedDatasetInfo:
    path: str
    timeframe: str
    rows: int
    index_timezone: str
    explicit_timezone: bool
    first_timestamp_ny: str
    last_timestamp_ny: str
    manifest_source: str | None
    manifest_price_type: str | None
    manifest_granularity: str | None


@dataclass(frozen=True)
class BacktestDataBundle:
    frame: pd.DataFrame
    data_source_used: str
    precision_package: dict[str, pd.DataFrame] | None = None


def has_explicit_timezone(index: pd.Index) -> bool:
    if index.empty:
        return True
    values = pd.Series(index.astype(str))
    return bool(values.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", case=False, regex=True, na=False).all())


def parse_prepared_index(index: pd.Index) -> pd.DatetimeIndex:
    if not has_explicit_timezone(index):
        raise ValueError("El indice del CSV preparado no tiene offset timezone explicito; el loader no acepta timestamps naive.")
    dt = pd.to_datetime(index, utc=True, format="ISO8601", errors="raise")
    return dt.tz_convert(NY_TZ)



def fx_market_mask(index: pd.DatetimeIndex) -> np.ndarray:
    minute_values = index.hour * 60 + index.minute
    dow = index.dayofweek
    return (
        ((dow >= 0) & (dow <= 3))
        | ((dow == 4) & (minute_values <= FX_CLOSE_MINUTE_NY))
        | ((dow == 6) & (minute_values > FX_REOPEN_MINUTE_NY))
    )


def fx_session_date(index: pd.DatetimeIndex) -> pd.Series:
    session_date = pd.Series(index.date, index=index, dtype="object")
    sunday_reopen = (index.dayofweek == 6) & ((index.hour * 60 + index.minute) > FX_REOPEN_MINUTE_NY)
    if np.any(sunday_reopen):
        session_date.loc[sunday_reopen] = (index[sunday_reopen] + pd.Timedelta(days=1)).date
    return session_date


def validate_price_frame(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("El dataset quedo vacio despues de cargarlo.")
    if frame.index.duplicated().any():
        raise ValueError("El dataset contiene timestamps duplicados despues de cargarlo.")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("El indice de precios no esta ordenado crecientemente.")
    invalid_ohlc = (frame["high"] < frame[["open", "close", "low"]].max(axis=1)) | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
    if bool(invalid_ohlc.any()):
        raise ValueError("Se detectaron velas OHLC invalidas despues de cargar el dataset.")


def _infer_index_delta(index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(index) < 2:
        return pd.Timedelta(minutes=15)
    deltas = pd.Series(index[1:] - index[:-1])
    mode = deltas.mode()
    if mode.empty:
        return pd.Timedelta(minutes=15)
    return pd.Timedelta(mode.iloc[0])


def _load_manifest_info(data_dir: Path) -> dict[str, Any]:
    manifest_path = data_dir / "prepared_data_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _manifest_entry_for_pair(manifest: dict[str, Any], pair: str) -> dict[str, Any]:
    pairs = manifest.get("pairs")
    if isinstance(pairs, dict):
        value = pairs.get(pair)
        if isinstance(value, dict):
            return value
    return {}


def load_prepared_ohlcv(pair: str, data_dirs: list[Path], timeframe: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for data_dir in data_dirs:
        path = data_dir / f"{pair}_{timeframe}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, index_col=0)
        frame.index = parse_prepared_index(frame.index)
        frame = frame[OHLCV_COLUMNS].copy()
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No encontre datos preparados para {pair} {timeframe} en {data_dirs}")

    merged = pd.concat(frames).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    validate_price_frame(merged)
    return merged


def load_high_precision_package(pair: str, data_dir: Path) -> dict[str, pd.DataFrame]:
    package: dict[str, pd.DataFrame] = {}
    for side in ("BID", "ASK", "MID"):
        path = data_dir / f"{pair}_M1_{side}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, index_col=0)
        frame.index = parse_prepared_index(frame.index)
        missing = [column for column in OHLCV_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"{path} no contiene columnas OHLCV requeridas: {missing}")
        frame = frame[OHLCV_COLUMNS].astype(float)
        validate_price_frame(frame)
        package[side.lower()] = frame
    if not package:
        raise FileNotFoundError(f"No encontre paquete de alta precision M1 para {pair} en {data_dir}")
    return package


def describe_available_price_data(pair: str, data_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for data_dir in data_dirs:
        manifest = _load_manifest_info(data_dir)
        pair_manifest = _manifest_entry_for_pair(manifest, pair)
        for timeframe in SUPPORTED_PREPARED_TIMEFRAMES:
            path = data_dir / f"{pair}_{timeframe}.csv"
            if not path.exists():
                continue
            index_frame = pd.read_csv(path, index_col=0, usecols=[0])
            parsed_index = parse_prepared_index(index_frame.index)
            rows.append(
                asdict(
                    PreparedDatasetInfo(
                        path=str(path),
                        timeframe=timeframe,
                        rows=int(len(index_frame)),
                        index_timezone=str(parsed_index.tz),
                        explicit_timezone=has_explicit_timezone(index_frame.index),
                        first_timestamp_ny=str(parsed_index.min()) if len(parsed_index) else "",
                        last_timestamp_ny=str(parsed_index.max()) if len(parsed_index) else "",
                        manifest_source=str(manifest.get("run_config", {}).get("source")) if manifest else None,
                        manifest_price_type=str(pair_manifest.get("price_type")) if pair_manifest else None,
                        manifest_granularity=str(pair_manifest.get("granularity")) if pair_manifest else None,
                    )
                )
            )
    return rows


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def macd(series: pd.Series, fast: int, slow: int, signal_period: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def atr(frame: pd.DataFrame, period: int) -> pd.Series:
    prev_close = frame["close"].shift(1)
    tr = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def adx(frame: pd.DataFrame, period: int) -> pd.Series:
    high = frame["high"]
    low = frame["low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=frame.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=frame.index)
    atr_series = atr(frame, period)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_series.replace(0.0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_series.replace(0.0, np.nan)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)) * 100
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().fillna(0.0)


def bollinger_bands(series: pd.Series, period: int, std_mult: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(period).mean()
    std = series.rolling(period).std(ddof=0)
    upper = mid + std * std_mult
    lower = mid - std * std_mult
    return mid, upper, lower


def keltner_channels(frame: pd.DataFrame, ema_length: int, atr_mult: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = ema(frame["close"], ema_length)
    atr_series = atr(frame, ema_length)
    upper = mid + atr_series * atr_mult
    lower = mid - atr_series * atr_mult
    return mid, upper, lower


def nr7(frame: pd.DataFrame) -> pd.Series:
    range_series = frame["high"] - frame["low"]
    min_range_7 = range_series.rolling(7).min()
    return (range_series == min_range_7) & (min_range_7 > 0)


def session_vwap(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    # Intraday VWAP that resets with fx_session_date
    session_dates = fx_session_date(frame.index)
    tp = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    v = frame["volume"]
    
    tpv = tp * v
    cum_tpv = tpv.groupby(session_dates).cumsum()
    cum_v = v.groupby(session_dates).cumsum()
    
    vwap = cum_tpv / cum_v.replace(0.0, np.nan)
    
    # Standard Deviation from VWAP (Intraday)
    sq_diff = (tp - vwap) ** 2
    cum_sq_diff_v = (sq_diff * v).groupby(session_dates).cumsum()
    vwap_std = np.sqrt(cum_sq_diff_v / cum_v.replace(0.0, np.nan))
    
    return vwap, vwap_std


def supertrend(frame: pd.DataFrame, atr_period: int, multiplier: float) -> tuple[pd.Series, pd.Series]:
    atr_series = atr(frame, atr_period)
    hl2 = (frame["high"] + frame["low"]) / 2.0
    basic_upper = hl2 + multiplier * atr_series
    basic_lower = hl2 - multiplier * atr_series
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    direction = pd.Series(index=frame.index, dtype=float)
    line = pd.Series(index=frame.index, dtype=float)

    for i in range(1, len(frame)):
        prev_i = i - 1
        prev_upper = final_upper.iat[prev_i]
        prev_lower = final_lower.iat[prev_i]
        if np.isnan(prev_upper) or basic_upper.iat[i] < prev_upper or frame["close"].iat[prev_i] > prev_upper:
            final_upper.iat[i] = basic_upper.iat[i]
        else:
            final_upper.iat[i] = prev_upper

        if np.isnan(prev_lower) or basic_lower.iat[i] > prev_lower or frame["close"].iat[prev_i] < prev_lower:
            final_lower.iat[i] = basic_lower.iat[i]
        else:
            final_lower.iat[i] = prev_lower

        if i == 1:
            direction.iat[i] = 1.0 if frame["close"].iat[i] >= hl2.iat[i] else -1.0
        elif direction.iat[prev_i] == -1 and frame["close"].iat[i] > final_upper.iat[prev_i]:
            direction.iat[i] = 1.0
        elif direction.iat[prev_i] == 1 and frame["close"].iat[i] < final_lower.iat[prev_i]:
            direction.iat[i] = -1.0
        else:
            direction.iat[i] = direction.iat[prev_i]

        line.iat[i] = final_lower.iat[i] if direction.iat[i] == 1 else final_upper.iat[i]

    direction = direction.ffill().fillna(0.0)
    line = line.ffill()
    return line, direction


def load_price_data(pair: str, data_dirs: list[Path], start: str, end: str) -> pd.DataFrame:
    merged = load_prepared_ohlcv(pair, data_dirs, "M5")
    merged = merged[fx_market_mask(merged.index)].copy()
    start_ts = pd.Timestamp(start, tz=NY_TZ)
    end_ts = pd.Timestamp(end, tz=NY_TZ) + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)
    merged = merged.loc[(merged.index >= start_ts) & (merged.index <= end_ts)].copy()
    validate_price_frame(merged)
    return merged


def slice_high_precision_package_to_frame(
    package: dict[str, pd.DataFrame] | None,
    frame_index: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame] | None:
    if package is None or frame_index.empty:
        return None

    bar_delta = _infer_index_delta(frame_index)
    m15_start = frame_index.min()
    m15_end = frame_index.max()
    m1_start = m15_start - bar_delta

    sliced: dict[str, pd.DataFrame] = {}
    for key, value in package.items():
        if key.endswith("_m1"):
            sliced[key] = value.loc[(value.index >= m1_start) & (value.index <= m15_end)].copy()
        elif key.endswith("_m15"):
            sliced[key] = value.loc[(value.index >= m15_start) & (value.index <= m15_end)].copy()
        else:
            sliced[key] = value.copy()
    return sliced


def load_backtest_data_bundle(
    pair: str,
    data_dirs: list[Path],
    start: str,
    end: str,
    execution_mode: str,
    *,
    high_precision_dir: Path = DEFAULT_HIGH_PRECISION_PREPARED_DIR,
    target_timeframe: str = "M15",
) -> BacktestDataBundle:
    normalized_mode = execution_mode.strip().lower()
    if normalized_mode != "high_precision_mode":
        raw_frame = load_price_data(pair, data_dirs, start, end)
        return BacktestDataBundle(
            frame=prepare_common_frame(raw_frame, target_timeframe=target_timeframe),
            data_source_used="prepared_m5_bid",
            precision_package=None,
        )

    package = load_high_precision_package(pair, high_precision_dir)
    start_ts = pd.Timestamp(start, tz=NY_TZ)
    end_ts = pd.Timestamp(end, tz=NY_TZ) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)

    filtered_m1: dict[str, pd.DataFrame] = {}
    for side, source in package.items():
        frame = source.loc[(source.index >= start_ts) & (source.index <= end_ts)].copy()
        frame = frame[fx_market_mask(frame.index)].copy()
        validate_price_frame(frame)
        filtered_m1[f"{side}_m1"] = frame

    mid_frame = filtered_m1["mid_m1"]
    strategy_frame = prepare_common_frame(mid_frame, target_timeframe=target_timeframe)
    bid_exec = resample_ohlcv_to_timeframe(filtered_m1["bid_m1"], target_timeframe)
    ask_exec = resample_ohlcv_to_timeframe(filtered_m1["ask_m1"], target_timeframe)
    mid_exec = resample_ohlcv_to_timeframe(filtered_m1["mid_m1"], target_timeframe)
    bid_m15 = _resample_to_m15(filtered_m1["bid_m1"])
    ask_m15 = _resample_to_m15(filtered_m1["ask_m1"])
    mid_m15 = _resample_to_m15(filtered_m1["mid_m1"])
    common_index = strategy_frame.index.intersection(bid_exec.index).intersection(ask_exec.index).intersection(mid_exec.index)
    if common_index.empty:
        raise ValueError("La fuente M1 BID/ASK no pudo alinearse con el timeframe canonico solicitado.")

    aligned_package = {
        **filtered_m1,
        "bid_exec": bid_exec.loc[common_index].copy(),
        "ask_exec": ask_exec.loc[common_index].copy(),
        "mid_exec": mid_exec.loc[common_index].copy(),
        "bid_m15": bid_m15.loc[common_index].copy() if str(target_timeframe).strip().upper() == "M15" else bid_m15.copy(),
        "ask_m15": ask_m15.loc[common_index].copy() if str(target_timeframe).strip().upper() == "M15" else ask_m15.copy(),
        "mid_m15": mid_m15.loc[common_index].copy() if str(target_timeframe).strip().upper() == "M15" else mid_m15.copy(),
    }
    return BacktestDataBundle(
        frame=strategy_frame.loc[common_index].copy(),
        data_source_used="dukascopy_m1_bid_ask_full",
        precision_package=aligned_package,
    )


def resample_ohlcv_to_timeframe(frame: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
    timeframe = str(target_timeframe).strip().upper()
    rule = TIMEFRAME_RESAMPLE_RULES.get(timeframe)
    if rule is None:
        raise ValueError(f"Timeframe no soportado para resample OHLCV: {target_timeframe!r}")
    if timeframe == "M1":
        return frame.copy()
    return (
        frame.resample(rule, label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )


def _resample_to_m15(frame: pd.DataFrame) -> pd.DataFrame:
    return resample_ohlcv_to_timeframe(frame, "M15")


def _build_h1_context(frame: pd.DataFrame) -> pd.DataFrame:
    h1 = (
        frame.resample("1h", label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )
    h1["h1_atr14"] = atr(h1, 14)
    h1["h1_adx14"] = adx(h1, 14)
    h1["h1_ema50"] = ema(h1["close"], 50)
    h1["h1_ema100"] = ema(h1["close"], 100)
    h1["h1_ema200"] = ema(h1["close"], 200)
    h1["h1_high"] = h1["high"]
    h1["h1_low"] = h1["low"]
    for lookback in (3, 5, 8):
        h1[f"h1_ema200_slope_{lookback}"] = h1["h1_ema200"] - h1["h1_ema200"].shift(lookback)
    return h1[[column for column in h1.columns if column.startswith("h1_")]].reindex(frame.index, method="ffill")


def fixed_session_window_components(
    index: pd.DatetimeIndex,
    start_hhmm: str,
    end_hhmm: str,
) -> tuple[np.ndarray, pd.Series, np.ndarray]:
    start_hour, start_minute = (int(part) for part in start_hhmm.split(":"))
    end_hour, end_minute = (int(part) for part in end_hhmm.split(":"))
    minute_values = (index.hour * 60 + index.minute).to_numpy()
    range_start = start_hour * 60 + start_minute
    range_end = end_hour * 60 + end_minute

    session_dates = pd.Series(index.date, index=index, dtype="object")
    if range_start < range_end:
        in_window = (minute_values >= range_start) & (minute_values < range_end)
        complete_mask = minute_values >= range_end
    else:
        in_window = (minute_values >= range_start) | (minute_values < range_end)
        after_start = minute_values >= range_start
        if np.any(after_start):
            session_dates.loc[after_start] = (index[after_start] + pd.Timedelta(days=1)).date
        complete_mask = (minute_values >= range_end) & (minute_values < range_start)
    return in_window, session_dates, complete_mask


def _fill_fixed_range_columns(frame: pd.DataFrame, start_hhmm: str, end_hhmm: str) -> pd.DataFrame:
    suffix = f"{start_hhmm}_{end_hhmm}".replace(":", "_")
    in_window, session_dates, complete_mask = fixed_session_window_components(frame.index, start_hhmm, end_hhmm)
    
    high_col = f"session_range_high_{suffix}"
    low_col = f"session_range_low_{suffix}"
    complete_col = f"session_range_complete_{suffix}"

    # Pre-calculamos los maximos/minimos por dia COMPLETOS (vectorized)
    sliced = frame.loc[in_window]
    day_res = sliced.groupby(session_dates.loc[sliced.index]).agg({"high": "max", "low": "min"})
    
    if day_res.empty:
         frame[high_col] = np.nan
         frame[low_col] = np.nan
         frame[complete_col] = False
         return frame
         
    frame[high_col] = session_dates.map(day_res["high"])
    frame[low_col] = session_dates.map(day_res["low"])
    frame[complete_col] = complete_mask & frame[high_col].notna()
    frame.loc[~frame[complete_col], [high_col, low_col]] = np.nan
    
    return frame


def prepare_common_frame(raw_frame: pd.DataFrame, target_timeframe: str = "M15") -> pd.DataFrame:
    timeframe = str(target_timeframe).strip().upper()
    if timeframe not in TIMEFRAME_RESAMPLE_RULES:
        raise ValueError(f"Timeframe canonico no soportado: {target_timeframe!r}")
    frame = resample_ohlcv_to_timeframe(raw_frame, timeframe)
    frame["prev_close"] = frame["close"].shift(1)
    frame["prev_high"] = frame["high"].shift(1)
    frame["prev_low"] = frame["low"].shift(1)
    frame["bar_range"] = frame["high"] - frame["low"]
    frame["body_abs"] = (frame["close"] - frame["open"]).abs()
    frame["atr14"] = atr(frame, 14)
    frame["adx14"] = adx(frame, 14)
    frame["range_atr"] = frame["bar_range"] / frame["atr14"].replace(0.0, np.nan)
    for period in (10, 20, 30, 50, 100, 150, 200):
        frame[f"ema{period}"] = ema(frame["close"], period)
    frame["rsi7"] = rsi(frame["close"], 7)
    frame["rsi14"] = rsi(frame["close"], 14)

    for period in (20, 30):
        for std_mult in (1.8, 2.0, 2.2, 2.5):
            mid, upper, lower = bollinger_bands(frame["close"], period, std_mult)
            suffix = f"{period}_{str(std_mult).replace('.', '_')}"
            frame[f"bb_mid_{suffix}"] = mid
            frame[f"bb_upper_{suffix}"] = upper
            frame[f"bb_lower_{suffix}"] = lower
            frame[f"bb_width_atr_{suffix}"] = (upper - lower) / frame["atr14"].replace(0.0, np.nan)

    for bars in (20, 30, 40, 55):
        frame[f"donchian_high_{bars}"] = frame["high"].shift(1).rolling(bars).max()
        frame[f"donchian_low_{bars}"] = frame["low"].shift(1).rolling(bars).min()
        frame[f"donchian_range_atr_{bars}"] = (
            (frame[f"donchian_high_{bars}"] - frame[f"donchian_low_{bars}"]) / frame["atr14"].replace(0.0, np.nan)
        )

    for ema_length in (20, 30):
        for atr_mult in (1.5, 2.0):
            mid, upper, lower = keltner_channels(frame, ema_length, atr_mult)
            suffix = f"{ema_length}_{str(atr_mult).replace('.', '_')}"
            frame[f"kc_mid_{suffix}"] = mid
            frame[f"kc_upper_{suffix}"] = upper
            frame[f"kc_lower_{suffix}"] = lower
            frame[f"kc_width_atr_{suffix}"] = (upper - lower) / frame["atr14"].replace(0.0, np.nan)

    vwap, vwap_std = session_vwap(frame)
    frame["vwap"] = vwap
    frame["vwap_std"] = vwap_std
    frame["vwap_dist_std"] = (frame["close"] - vwap) / vwap_std.replace(0.0, np.nan)
    frame["is_nr7"] = nr7(frame)
    
    # RSIs
    for p in (2, 7, 14):
        frame[f"rsi{p}"] = rsi(frame["close"], p)
        
    # EMAs
    for p in (9, 12, 21, 26, 50, 100, 200):
        frame[f"ema{p}"] = ema(frame["close"], p)
        
    # MACD Variations
    m_fast, m_sig, m_hist = macd(frame["close"], 12, 26, 9)
    frame["macd_main_hist"] = m_hist
    
    m_fast2, m_sig2, m_hist2 = macd(frame["close"], 24, 52, 18)
    frame["macd_slow_hist"] = m_hist2
    
    m_fast3, m_sig3, m_hist3 = macd(frame["close"], 6, 13, 5)
    frame["macd_fast_hist"] = m_hist3

    # for atr_period in (7, 10, 14):
    #     for mult in (2.0, 2.5, 3.0, 3.5):
    #         line, direction = supertrend(frame, atr_period, mult)
    #         suffix = f"{atr_period}_{str(mult).replace('.', '_')}"
    #         frame[f"supertrend_line_{suffix}"] = line
    #         frame[f"supertrend_dir_{suffix}"] = direction

    session_dates = fx_session_date(frame.index)
    day_high = frame.groupby(session_dates)["high"].transform("max")
    day_low = frame.groupby(session_dates)["low"].transform("min")
    day_levels = pd.DataFrame({"session_date": session_dates, "day_high": day_high, "day_low": day_low}).drop_duplicates("session_date")
    day_levels["prev_day_high"] = day_levels["day_high"].shift(1)
    day_levels["prev_day_low"] = day_levels["day_low"].shift(1)
    prev_map_high = day_levels.set_index("session_date")["prev_day_high"]
    prev_map_low = day_levels.set_index("session_date")["prev_day_low"]
    frame["prev_day_high"] = session_dates.map(prev_map_high)
    frame["prev_day_low"] = session_dates.map(prev_map_low)
    
    # Inyeccion de Apertura Diaria (Daily Open)
    day_opens = frame.groupby(session_dates)["open"].transform("first")
    frame["daily_open"] = day_opens

    frame = frame.join(_build_h1_context(frame))
    frame["day_running_high"] = frame.groupby(session_dates)["high"].cummax()
    frame["day_running_low"] = frame.groupby(session_dates)["low"].cummin()
    frame["day_running_range"] = frame["day_running_high"] - frame["day_running_low"]
    frame["day_range_m15_atr"] = frame["day_running_range"] / frame["atr14"].replace(0.0, np.nan)
    frame["day_range_h1_atr"] = frame["day_running_range"] / frame["h1_atr14"].replace(0.0, np.nan)
    frame = _fill_fixed_range_columns(frame, "11:00", "13:00")
    required_columns = [
        "atr14",
        "adx14",
        "range_atr",
        "ema20",
        "ema50",
        "ema100",
        "ema150",
        "ema200",
        "rsi7",
        "rsi14",
        "prev_day_high",
        "prev_day_low",
        "daily_open",
        "h1_atr14",
        "h1_adx14",
        "h1_ema50",
        "h1_ema100",
        "h1_ema200",
        "h1_high",
        "h1_low",
        "h1_ema200_slope_3",
        "h1_ema200_slope_5",
        "h1_ema200_slope_8",
        "day_range_m15_atr",
        "day_range_h1_atr",
    ]
    # Inyeccion de AM Range (07:00 - 11:00 NY) para LS-SR
    frame = _fill_fixed_range_columns(frame, "07:00", "11:00")
    # Inyeccion de Asia (00:00 - 07:00 NY)
    frame = _fill_fixed_range_columns(frame, "00:00", "07:00")
    # Inyeccion de Londres (03:00 - 11:00 NY)
    frame = _fill_fixed_range_columns(frame, "03:00", "11:00")
    # Inyeccion de Asia exacta manual (19:00 - 03:00 NY)
    frame = _fill_fixed_range_columns(frame, "19:00", "03:00")
    # Inyeccion de Londres exacta manual (03:00 - 07:00 NY)
    frame = _fill_fixed_range_columns(frame, "03:00", "07:00")
    # Inyeccion de SB Anchor (03:00 - 08:30 NY)
    frame = _fill_fixed_range_columns(frame, "03:00", "08:30")
    # Inyeccion de Midday Range (11:00 - 13:00)
    frame = _fill_fixed_range_columns(frame, "11:00", "13:00")
    
    frame = frame.dropna(subset=required_columns).copy()

    # Keep ICT primitives on the canonical frame path so objective ICT setups can
    # reuse the same loader contract as the rest of the lab.
    from research_lab.ict_primitives import add_ict_primitives

    frame = add_ict_primitives(frame)
    return frame.copy()
