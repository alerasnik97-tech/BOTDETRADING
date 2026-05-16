from __future__ import annotations

import os
import pandas as pd
from pathlib import Path
from research_lab.config import DEFAULT_DATA_DIRS, DEFAULT_NEWS_ENABLED

def assert_train_data_no_holdout(df: pd.DataFrame, cutoff: str = "2025-01-01") -> None:
    """Fail-closed check for 2025+ rows in training data."""
    if len(df) == 0:
        return
    
    cutoff_ts = pd.Timestamp(cutoff, tz="UTC")
    
    # Identify timestamp source
    idx_to_check = None
    if pd.api.types.is_datetime64_any_dtype(df.index):
        idx_to_check = df.index
    elif "timestamp" in df.columns:
        idx_to_check = pd.to_datetime(df["timestamp"])
    elif "timestamp_utc" in df.columns:
        idx_to_check = pd.to_datetime(df["timestamp_utc"])
    
    if idx_to_check is None:
        raise ValueError("FAIL_CLOSED: No detectable timestamp found for preflight check.")

    # Convert to UTC aware
    if idx_to_check.tz is None:
        idx_to_check = idx_to_check.tz_localize("UTC")
    else:
        idx_to_check = idx_to_check.tz_convert("UTC")
        
    leaky_mask = idx_to_check >= cutoff_ts
    leaky_count = int(leaky_mask.sum())
            
    if leaky_count > 0:
        max_ts = idx_to_check.max()
        raise RuntimeError(
            f"FAIL_CLOSED: Detected {leaky_count} rows with timestamp >= {cutoff}. "
            f"Max found: {max_ts}. This is a critical holdout leakage."
        )

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
