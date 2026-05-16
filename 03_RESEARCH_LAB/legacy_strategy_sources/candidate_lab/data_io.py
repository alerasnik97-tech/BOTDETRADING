from __future__ import annotations

from pathlib import Path

import pandas as pd

from research_lab.config import DEFAULT_RAW_NEWS_FILE_OBSOLETE, NewsConfig, PAIR_CANONICAL_NEWS_FILES
from research_lab.data_loader import load_prepared_ohlcv, validate_price_frame
from research_lab.news_filter import load_news_events

from .config import LabPaths


def _trim_frame(frame: pd.DataFrame, *, start_date: str, end_date: str) -> pd.DataFrame:
    start_day = pd.Timestamp(start_date).date()
    end_day = pd.Timestamp(end_date).date()
    trimmed = frame.loc[(frame.index.date >= start_day) & (frame.index.date <= end_day)].copy()
    if trimmed.empty:
        raise ValueError(f"El dataset quedo vacio luego del recorte {start_date} -> {end_date}")
    return trimmed


def load_price_frames(
    paths: LabPaths,
    *,
    start_date: str,
    end_date: str,
    pad_before_days: int = 1,
    pad_after_days: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    padded_start = (pd.Timestamp(start_date) - pd.Timedelta(days=pad_before_days)).strftime("%Y-%m-%d")
    padded_end = (pd.Timestamp(end_date) + pd.Timedelta(days=pad_after_days)).strftime("%Y-%m-%d")

    h1 = load_prepared_ohlcv("EURUSD", list(paths.price_dirs), "H1")
    m5 = load_prepared_ohlcv("EURUSD", list(paths.price_dirs), "M5")
    validate_price_frame(h1)
    validate_price_frame(m5)

    h1 = _trim_frame(h1, start_date=padded_start, end_date=padded_end)
    m5 = _trim_frame(m5, start_date=padded_start, end_date=padded_end)

    coverage = {
        "price_dirs": [str(path.relative_to(paths.project_root)) for path in paths.price_dirs],
        "h1_rows": int(len(h1)),
        "m5_rows": int(len(m5)),
        "h1_first_timestamp_ny": str(h1.index.min()),
        "h1_last_timestamp_ny": str(h1.index.max()),
        "m5_first_timestamp_ny": str(m5.index.min()),
        "m5_last_timestamp_ny": str(m5.index.max()),
    }
    return h1, m5, coverage


def load_news_frame(paths: LabPaths, *, start_date: str, end_date: str) -> tuple[pd.DataFrame, dict[str, object]]:
    settings = NewsConfig(
        file_path=Path(paths.project_root) / PAIR_CANONICAL_NEWS_FILES["EURUSD"],
        raw_file_path=Path(paths.project_root) / DEFAULT_RAW_NEWS_FILE_OBSOLETE,
    )
    result = load_news_events("EURUSD", settings)
    if not result.enabled or result.events.empty:
        return pd.DataFrame(), {
            "enabled": False,
            "source_path": str(paths.news_file.relative_to(paths.project_root)),
            "coverage_start_date": None,
            "coverage_end_date": None,
            "rows": 0,
        }

    events = result.events.copy()
    events["timestamp_ny"] = pd.to_datetime(events["timestamp_ny"], utc=True).dt.tz_convert("US/Eastern")
    events["event_name_normalized"] = events.get("event_name_normalized", "").astype(str)
    events["event_date"] = events["timestamp_ny"].dt.date
    padded_start = pd.Timestamp(start_date, tz="US/Eastern") - pd.Timedelta(days=1)
    padded_end = pd.Timestamp(end_date, tz="US/Eastern") + pd.Timedelta(days=1)
    events = events.loc[(events["timestamp_ny"] >= padded_start) & (events["timestamp_ny"] <= padded_end)].copy()
    events = events.sort_values("timestamp_ny").reset_index(drop=True)

    all_events = result.events.copy()
    all_events["timestamp_ny"] = pd.to_datetime(all_events["timestamp_ny"], utc=True).dt.tz_convert("US/Eastern")
    coverage_start_date = str(all_events["timestamp_ny"].dt.date.min()) if not all_events.empty else None
    coverage_end_date = str(all_events["timestamp_ny"].dt.date.max()) if not all_events.empty else None
    events.attrs["coverage_start_date"] = coverage_start_date
    events.attrs["coverage_end_date"] = coverage_end_date

    coverage = {
        "enabled": True,
        "source_path": str(paths.news_file.relative_to(paths.project_root)),
        "coverage_start_date": coverage_start_date,
        "coverage_end_date": coverage_end_date,
        "rows": int(len(events)),
    }
    return events, coverage
