from __future__ import annotations

import math

from research_lab.config import MIN_TOTAL_TRADES, MIN_TRADES_PER_MONTH, TARGET_TRADES_PER_MONTH_MAX, TARGET_TRADES_PER_MONTH_MIN


def sample_penalty_meta(summary: dict) -> tuple[bool, bool, float]:
    total_trades = float(summary.get("total_trades", 0.0))
    trades_per_month = float(summary.get("avg_trades_per_month", 0.0))
    insufficient_sample = total_trades < MIN_TOTAL_TRADES or trades_per_month < MIN_TRADES_PER_MONTH
    penalty = 0.0
    if total_trades < MIN_TOTAL_TRADES:
        penalty += 2000.0 + (MIN_TOTAL_TRADES - total_trades) * 1.5
    if trades_per_month < MIN_TRADES_PER_MONTH:
        penalty += 1500.0 + (MIN_TRADES_PER_MONTH - trades_per_month) * 500.0
    return insufficient_sample, penalty > 0.0, penalty


def score_is_summary(summary: dict) -> float:
    profit_factor = float(summary.get("profit_factor", 0.0))
    expectancy_r = float(summary.get("expectancy_r", 0.0))
    max_drawdown_pct = float(summary.get("max_drawdown_pct", 0.0))
    negative_months = float(summary.get("negative_months", 0.0))
    trades_per_month = float(summary.get("avg_trades_per_month", 0.0))

    pf_capped = 2.0 if not math.isfinite(profit_factor) else min(profit_factor, 2.0)
    score = pf_capped * 200.0 + expectancy_r * 500.0 - max_drawdown_pct * 4.0 - negative_months * 2.0
    if profit_factor < 1.0:
        score -= 100.0
    if expectancy_r < 0.0:
        score -= 100.0
    if trades_per_month < 0.75:
        score -= (0.75 - trades_per_month) * 200.0
    return score


def frequency_penalty(trades_per_month: float) -> float:
    if TARGET_TRADES_PER_MONTH_MIN <= trades_per_month <= TARGET_TRADES_PER_MONTH_MAX:
        return 0.0
    if trades_per_month < TARGET_TRADES_PER_MONTH_MIN:
        return (TARGET_TRADES_PER_MONTH_MIN - trades_per_month) * 4.0
    return (trades_per_month - TARGET_TRADES_PER_MONTH_MAX) * 2.0


def compute_final_score(
    *,
    full_summary: dict,
    oos_summary: dict,
    plateau_index: float,
    top10_median_gap: float,
    positive_years_full: int,
    share_best_year: float,
) -> float:
    pf_oos = float(oos_summary.get("profit_factor", 0.0))
    exp_oos = float(oos_summary.get("expectancy_r", 0.0))
    dd_oos = float(oos_summary.get("max_drawdown_pct", 0.0))
    neg_months_oos = float(oos_summary.get("negative_months", 0.0))
    neg_years_oos = float(oos_summary.get("negative_years", 0.0))
    trades_pm_oos = float(oos_summary.get("avg_trades_per_month", 0.0))

    pf_full = float(full_summary.get("profit_factor", 0.0))
    exp_full = float(full_summary.get("expectancy_r", 0.0))
    dd_full = float(full_summary.get("max_drawdown_pct", 0.0))

    oos_component = (
        min(pf_oos if math.isfinite(pf_oos) else 2.0, 2.0) * 45.0
        + exp_oos * 500.0 * 0.45
        - dd_oos * 2.5 * 0.45
        - neg_months_oos * 1.5 * 0.45
        - neg_years_oos * 12.0 * 0.45
    )
    consistency_component = positive_years_full * 4.0 - float(full_summary.get("negative_years", 0.0)) * 12.0
    stability_component = plateau_index * 15.0 - top10_median_gap * 8.0
    frequency_component = -frequency_penalty(trades_pm_oos) * 1.0
    support_component = min(pf_full if math.isfinite(pf_full) else 2.0, 2.0) * 10.0 + exp_full * 500.0 * 0.10 - dd_full * 0.8

    score = oos_component + consistency_component + stability_component + frequency_component + support_component
    if pf_oos < 1.0:
        score -= 80.0
    if exp_oos <= 0.0:
        score -= 80.0
    if positive_years_full < 4:
        score -= (4 - positive_years_full) * 40.0
    if share_best_year > 0.60:
        score -= (share_best_year - 0.60) * 200.0
    insufficient_sample, _, penalty = sample_penalty_meta(oos_summary)
    if insufficient_sample:
        score -= penalty
    return score
