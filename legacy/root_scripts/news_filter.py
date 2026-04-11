from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import NY_TZ, NewsConfig, PAIR_META


def relevant_currencies(pair: str) -> set[str]:
    meta = PAIR_META[pair]
    return {meta["base"], meta["quote"]}


def load_news_events(pair: str, settings: NewsConfig) -> tuple[pd.DataFrame, bool]:
    if not settings.enabled:
        return pd.DataFrame(), False

    path = Path(settings.file_path)
    if not path.exists():
        return pd.DataFrame(), False

    news = pd.read_csv(path)
    if news.empty or "DateTime" not in news.columns:
        return pd.DataFrame(), False

    frame = news.copy()
    frame["DateTime"] = pd.to_datetime(frame["DateTime"], utc=True, errors="coerce").dt.tz_convert(NY_TZ)
    frame = frame.dropna(subset=["DateTime"])
    impact = frame.get("Impact", pd.Series("", index=frame.index)).astype(str).str.lower()
    frame = frame[impact.str.contains("high")].copy()
    if frame.empty:
        return pd.DataFrame(), False

    frame["Currency"] = frame["Currency"].astype(str).str.upper()
    frame = frame[frame["Currency"].isin(relevant_currencies(pair))].copy()
    frame = frame.sort_values("DateTime").reset_index(drop=True)
    return frame, not frame.empty


def build_entry_block(index: pd.DatetimeIndex, news_events: pd.DataFrame, settings: NewsConfig) -> np.ndarray:
    mask = np.zeros(len(index), dtype=bool)
    if news_events.empty:
        return mask

    for event_ts in news_events["DateTime"]:
        block_start = event_ts - pd.Timedelta(minutes=settings.pre_minutes)
        block_end = event_ts + pd.Timedelta(minutes=settings.post_minutes)
        left = index.searchsorted(block_start, side="left")
        right = index.searchsorted(block_end, side="right")
        mask[left:right] = True
    return mask
