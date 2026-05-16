from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import HarnessPaths

PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


def _load_single_price_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert("US/Eastern")
    missing_columns = [column for column in PRICE_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Faltan columnas {missing_columns} en {path}")
    return frame[list(PRICE_COLUMNS)].sort_index()


def load_merged_price_frame(
    paths: HarnessPaths,
    *,
    pair: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    pad_before_days: int = 0,
    pad_after_days: int = 0,
) -> pd.DataFrame:
    filename = f"{pair}_{timeframe}.csv"
    frames: list[pd.DataFrame] = []
    for price_dir in paths.price_dirs:
        candidate = price_dir / filename
        if candidate.exists():
            frames.append(_load_single_price_file(candidate))
    if not frames:
        raise FileNotFoundError(f"No se encontro {filename} en {paths.price_dirs}")

    merged = pd.concat(frames).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    start_day = (pd.Timestamp(start_date) - pd.Timedelta(days=pad_before_days)).date()
    end_day = (pd.Timestamp(end_date) + pd.Timedelta(days=pad_after_days)).date()
    merged = merged.loc[(merged.index.date >= start_day) & (merged.index.date <= end_day)].copy()
    if merged.empty:
        raise ValueError(f"{filename} quedo vacio luego del recorte {start_date} -> {end_date}")
    return merged


def load_news_frame(paths: HarnessPaths, *, start_date: str, end_date: str) -> pd.DataFrame:
    if not paths.news_file.exists():
        raise FileNotFoundError(f"No existe el calendario de noticias: {paths.news_file}")

    frame = pd.read_csv(paths.news_file, dtype=str, keep_default_na=False, low_memory=False)
    if "timestamp_ny" not in frame.columns:
        raise ValueError("El CSV de noticias no tiene timestamp_ny")

    if "impact_level" in frame.columns:
        frame = frame.loc[frame["impact_level"].str.upper() == "HIGH"].copy()
    if "validation_status" in frame.columns:
        frame = frame.loc[frame["validation_status"].str.startswith("approved")].copy()

    frame["timestamp_ny"] = pd.to_datetime(frame["timestamp_ny"], utc=True).dt.tz_convert("US/Eastern")
    frame["event_name_normalized"] = frame.get("event_name_normalized", "").astype(str)
    if frame.empty:
        raise ValueError("El CSV de noticias quedo vacio luego del filtrado canonico")

    coverage_start_date = frame["timestamp_ny"].dt.date.min()
    coverage_end_date = frame["timestamp_ny"].dt.date.max()

    start_bound = pd.Timestamp(start_date, tz="US/Eastern") - pd.Timedelta(days=1)
    end_bound = pd.Timestamp(end_date, tz="US/Eastern") + pd.Timedelta(days=1)
    frame = frame.loc[(frame["timestamp_ny"] >= start_bound) & (frame["timestamp_ny"] <= end_bound)].copy()
    frame = frame.sort_values("timestamp_ny").reset_index(drop=True)
    frame.attrs["coverage_start_date"] = str(coverage_start_date)
    frame.attrs["coverage_end_date"] = str(coverage_end_date)
    frame.attrs["source_path"] = str(paths.news_file)
    return frame


def load_expected_internal_baseline(paths: HarnessPaths) -> dict[str, object]:
    summary_path = paths.core_root / "results" / "SCBI_2020_2025_DURABILITY" / "summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))
