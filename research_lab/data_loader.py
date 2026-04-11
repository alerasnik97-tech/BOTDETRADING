from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from research_lab.config import NY_TZ

warnings.simplefilter("ignore", PerformanceWarning)

FX_REOPEN_MINUTE_NY = 17 * 60
FX_CLOSE_MINUTE_NY = 17 * 60


def has_explicit_timezone(index: pd.Index) -> bool:
    if index.empty:
        return True
    values = pd.Series(index.astype(str))
    return bool(values.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", case=False, regex=True, na=False).all())


def parse_prepared_index(index: pd.Index) -> pd.DatetimeIndex:
    if not has_explicit_timezone(index):
        raise ValueError("El índice del CSV preparado no tiene offset timezone explícito; el loader no acepta timestamps naive.")
    return pd.to_datetime(index.astype(str), utc=True, errors="raise").tz_convert(NY_TZ)


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
        raise ValueError("El dataset quedó vacío después de cargarlo.")
    if frame.index.duplicated().any():
        raise ValueError("El dataset contiene timestamps duplicados después de cargarlo.")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("El índice de precios no está ordenado crecientemente.")
    invalid_ohlc = (frame["high"] < frame[["open", "close", "low"]].max(axis=1)) | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
    if bool(invalid_ohlc.any()):
        raise ValueError("Se detectaron velas OHLC inválidas después de cargar el dataset.")


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


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
    frames: list[pd.DataFrame] = []
    for data_dir in data_dirs:
        path = data_dir / f"{pair}_M5.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        frame.index = parse_prepared_index(frame.index)
        frame = frame[["open", "high", "low", "close", "volume"]].copy()
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No encontré datos preparados para {pair} en {data_dirs}")

    merged = pd.concat(frames).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged[fx_market_mask(merged.index)].copy()
    start_ts = pd.Timestamp(start, tz=NY_TZ)
    end_ts = pd.Timestamp(end, tz=NY_TZ) + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)
    merged = merged.loc[(merged.index >= start_ts) & (merged.index <= end_ts)].copy()
    validate_price_frame(merged)
    return merged


def _resample_to_m15(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.resample("15min", label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )


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
    for lookback in (3, 5, 8):
        h1[f"h1_ema200_slope_{lookback}"] = h1["h1_ema200"] - h1["h1_ema200"].shift(lookback)
    return h1[[column for column in h1.columns if column.startswith("h1_")]].reindex(frame.index, method="ffill")


def _fill_fixed_range_columns(frame: pd.DataFrame, start_hhmm: str, end_hhmm: str) -> pd.DataFrame:
    start_hour, start_minute = (int(part) for part in start_hhmm.split(":"))
    end_hour, end_minute = (int(part) for part in end_hhmm.split(":"))
    suffix = end_hhmm.replace(":", "_")
    date_key = frame.index.date
    minute_values = frame.index.hour * 60 + frame.index.minute
    range_start = start_hour * 60 + start_minute
    range_end = end_hour * 60 + end_minute
    high_col = f"session_range_high_{suffix}"
    low_col = f"session_range_low_{suffix}"
    complete_col = f"session_range_complete_{suffix}"
    frame[high_col] = np.nan
    frame[low_col] = np.nan
    frame[complete_col] = False

    for session_date in pd.Index(date_key).unique():
        mask_day = date_key == session_date
        in_window = mask_day & (minute_values >= range_start) & (minute_values < range_end)
        if not np.any(in_window):
            continue
        session_high = float(frame.loc[in_window, "high"].max())
        session_low = float(frame.loc[in_window, "low"].min())
        active_mask = mask_day & (minute_values >= range_end)
        frame.loc[active_mask, high_col] = session_high
        frame.loc[active_mask, low_col] = session_low
        frame.loc[active_mask, complete_col] = True
    return frame


def prepare_common_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    frame = _resample_to_m15(raw_frame).copy()
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

    for atr_period in (7, 10, 14):
        for mult in (2.0, 2.5, 3.0, 3.5):
            line, direction = supertrend(frame, atr_period, mult)
            suffix = f"{atr_period}_{str(mult).replace('.', '_')}"
            frame[f"supertrend_line_{suffix}"] = line
            frame[f"supertrend_dir_{suffix}"] = direction

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
        "h1_atr14",
        "h1_adx14",
        "h1_ema50",
        "h1_ema100",
        "h1_ema200",
        "h1_ema200_slope_3",
        "h1_ema200_slope_5",
        "h1_ema200_slope_8",
        "day_range_m15_atr",
        "day_range_h1_atr",
    ]
    frame = frame.dropna(subset=required_columns).copy()
    return frame.copy()
