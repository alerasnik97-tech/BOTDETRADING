from __future__ import annotations

from pathlib import Path

import pandas as pd

from .baseline_truth_model import run_baseline_truth_model
from .candidate_matrix import build_baseline_config, build_candidate_matrix
from .config import CandidateConfig, LabPaths, default_paths
from .data_io import load_news_frame, load_price_frames
from .reporting import build_baseline_payload, build_variant_row


def resolve_paths(project_root: str | None = None) -> LabPaths:
    return default_paths(Path(project_root)) if project_root else default_paths()


def load_inputs(paths: LabPaths, *, start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    h1, m5, price_coverage = load_price_frames(paths, start_date=start_date, end_date=end_date)
    news, news_coverage = load_news_frame(paths, start_date=start_date, end_date=end_date)
    return h1, m5, news, {"price": price_coverage, "news": news_coverage}


def execute_candidate(config: CandidateConfig, *, h1: pd.DataFrame, m5: pd.DataFrame, news: pd.DataFrame) -> tuple[dict[str, object], dict[str, object]]:
    result = run_baseline_truth_model(config, h1=h1, m5=m5, news=news)
    row = build_variant_row(config, result)
    return result, row


def execute_baseline(paths: LabPaths, *, start_date: str, end_date: str) -> tuple[CandidateConfig, dict[str, object], dict[str, object]]:
    h1, m5, news, coverage = load_inputs(paths, start_date=start_date, end_date=end_date)
    config = build_baseline_config(start_date, end_date)
    result, _ = execute_candidate(config, h1=h1, m5=m5, news=news)
    payload = build_baseline_payload(config, result, coverage)
    return config, result, payload


def execute_matrix(paths: LabPaths, *, start_date: str, end_date: str, profile: str, max_variants: int | None = None) -> tuple[pd.DataFrame, dict[str, dict[str, object]], dict[str, object]]:
    h1, m5, news, coverage = load_inputs(paths, start_date=start_date, end_date=end_date)
    configs = build_candidate_matrix(start_date=start_date, end_date=end_date, profile=profile, max_variants=max_variants)
    rows: list[dict[str, object]] = []
    results: dict[str, dict[str, object]] = {}
    total = len(configs)
    for idx, config in enumerate(configs, start=1):
        print(f"[{idx}/{total}] Ejecutando {config.variant_id}")
        result, row = execute_candidate(config, h1=h1, m5=m5, news=news)
        rows.append(row)
        results[config.variant_id] = result
    ranked = pd.DataFrame(rows)
    baseline_config = build_baseline_config(start_date, end_date)
    baseline_result = results[baseline_config.variant_id]
    baseline_payload = build_baseline_payload(baseline_config, baseline_result, coverage)
    return ranked, results, baseline_payload
