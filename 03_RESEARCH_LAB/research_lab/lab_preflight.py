from __future__ import annotations

import os
import pandas as pd
from pathlib import Path
from research_lab.config import DEFAULT_DATA_DIRS, DEFAULT_NEWS_ENABLED

def assert_train_data_no_holdout(df: pd.DataFrame, cutoff: str = "2025-01-01") -> None:
    """Fail-closed check for 2025+ rows in training data."""
    if len(df) == 0:
        return
    
    # Ensure index is datetime and has timezone (UTC preferred)
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("Dataset index must be datetime for preflight check.")
    
    cutoff_ts = pd.Timestamp(cutoff, tz="UTC")
    idx_utc = df.index.tz_convert("UTC") if df.index.tz else df.index.tz_localize("UTC")
    
    # Fail-closed explicit iteration
    leaky_count = 0
    for ts in idx_utc:
        if ts >= cutoff_ts:
            leaky_count += 1
            
    if leaky_count > 0:
        max_ts = idx_utc.max()
        raise RuntimeError(f"FAIL_CLOSED: Detected {leaky_count} rows with timestamp >= {cutoff}. Max found: {max_ts}. This is a critical holdout leakage.")

def assert_default_data_dirs_no_holdout() -> None:
    """Ensures default data dirs do not contain holdout or 2025_2026 tokens."""
    forbidden = ["holdout", "2025", "2026", "sealed", "quarantine"]
    for path in DEFAULT_DATA_DIRS:
        p_str = str(path).lower()
        for token in forbidden:
            if token in p_str:
                raise RuntimeError(f"FAIL_CLOSED: Forbidden token '{token}' found in DEFAULT_DATA_DIRS: {path}")

def assert_output_dir_not_quarantine(output_path: str | Path) -> None:
    """Ensures output directory is not in a quarantine or legacy folder."""
    forbidden = ["quarantine", "legacy_archive", "07_backups", "do_not_use"]
    p_str = str(output_path).lower()
    for token in forbidden:
        if token in p_str:
            raise RuntimeError(f"FAIL_CLOSED: Forbidden token '{token}' found in output path: {output_path}")

def assert_news_disabled_unless_certified() -> None:
    """Ensures news is disabled unless explicitly certified (currently blocked)."""
    if DEFAULT_NEWS_ENABLED:
        raise RuntimeError("FAIL_CLOSED: News is enabled in config but not yet certified for this research phase.")

def run_institutional_preflight() -> None:
    """Executes all structural preflight checks."""
    assert_default_data_dirs_no_holdout()
    assert_news_disabled_unless_certified()
