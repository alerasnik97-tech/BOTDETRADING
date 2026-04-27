from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import HarnessPaths, TruthModelConfig, default_paths
from .data_io import load_merged_price_frame, load_news_frame
from .reporting import build_variant_row, rank_variants
from .strategy import run_truth_model


def resolve_paths(workspace_root: str | None = None) -> HarnessPaths:
    if workspace_root is None:
        return default_paths()
    return default_paths(Path(workspace_root))


def load_research_inputs(paths: HarnessPaths, *, start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    h1 = load_merged_price_frame(
        paths,
        pair="EURUSD",
        timeframe="H1",
        start_date=start_date,
        end_date=end_date,
        pad_before_days=1,
        pad_after_days=0,
    )
    m5 = load_merged_price_frame(
        paths,
        pair="EURUSD",
        timeframe="M5",
        start_date=start_date,
        end_date=end_date,
        pad_before_days=0,
        pad_after_days=1,
    )
    news = load_news_frame(paths, start_date=start_date, end_date=end_date)
    return h1, m5, news


def execute_variant(
    config: TruthModelConfig,
    *,
    h1: pd.DataFrame,
    m5: pd.DataFrame,
    news: pd.DataFrame,
) -> tuple[dict[str, object], dict[str, object]]:
    run_result = run_truth_model(config, h1=h1, m5=m5, news=news)
    row = build_variant_row(config, run_result)
    return run_result, row


def execute_variant_matrix(
    configs: list[TruthModelConfig],
    *,
    h1: pd.DataFrame,
    m5: pd.DataFrame,
    news: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    run_results: dict[str, dict[str, object]] = {}
    total = len(configs)

    for position, config in enumerate(configs, start=1):
        print(f"[{position}/{total}] Ejecutando {config.variant_id}")
        run_result, row = execute_variant(config, h1=h1, m5=m5, news=news)
        rows.append(row)
        run_results[config.variant_id] = run_result

    results = pd.DataFrame(rows)
    ranked = rank_variants(results)
    return ranked, run_results
