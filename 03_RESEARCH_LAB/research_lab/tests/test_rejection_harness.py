from __future__ import annotations

from research_lab.rejection_protocol import (
    evaluate_is_rejection,
    evaluate_oos_rejection,
    HARD_REJECT,
    SOFT_REJECT,
    PASS_MINIMUM,
    STRONG_CANDIDATE,
)

def test_hard_reject_in_sample():
    # Estrategia basura (IS_FLOP)
    bad_summary = {
        "profit_factor": 1.02, # Menor a 1.05
        "expectancy_r": 0.01,
    }
    rejected, level, reason = evaluate_is_rejection(bad_summary)
    assert rejected is True
    assert level == HARD_REJECT
    assert reason == "IS_FLOP_PF_TOO_LOW"

def test_hard_reject_is_expectancy():
    # Expectancy muy baja
    bad_summary = {
        "profit_factor": 1.10,
        "expectancy_r": 0.01, # Menor a 0.02
    }
    rejected, level, reason = evaluate_is_rejection(bad_summary)
    assert rejected is True
    assert level == HARD_REJECT
    assert reason == "IS_FLOP_EXP_TOO_LOW"

def test_pass_in_sample():
    good_summary = {
        "profit_factor": 1.15,
        "expectancy_r": 0.10,
    }
    rejected, level, reason = evaluate_is_rejection(good_summary)
    assert rejected is False
    assert level == PASS_MINIMUM

def test_hard_reject_oos_pf():
    bad_oos = {
        "profit_factor": 0.95,
        "expectancy_r": -0.05,
        "max_drawdown_pct": 5.0,
        "negative_years": 1,
    }
    rejected, level, reason = evaluate_oos_rejection(bad_oos, insufficient_sample=False)
    assert rejected is True
    assert level == HARD_REJECT
    assert reason == "OOS_FLOP_PF_NEGATIVE"

def test_soft_reject_oos_drawdown():
    risky_oos = {
        "profit_factor": 1.30,
        "expectancy_r": 0.20,
        "max_drawdown_pct": 16.0, # Mayor a 15.0
        "negative_years": 0,
    }
    rejected, level, reason = evaluate_oos_rejection(risky_oos, insufficient_sample=False)
    assert rejected is True
    assert level == SOFT_REJECT
    assert reason == "OOS_DRAWDOWN_UNACCEPTABLE"

def test_soft_reject_oos_consistency():
    inconsistent_oos = {
        "profit_factor": 1.15,
        "expectancy_r": 0.05,
        "max_drawdown_pct": 10.0,
        "negative_years": 3, # >= 3
    }
    rejected, level, reason = evaluate_oos_rejection(inconsistent_oos, insufficient_sample=False)
    assert rejected is True
    assert level == SOFT_REJECT
    assert reason == "OOS_CONSISTENCY_FATAL"

def test_strong_candidate_oos():
    perfect_oos = {
        "profit_factor": 1.25,
        "expectancy_r": 0.15,
        "max_drawdown_pct": 7.0,
        "negative_years": 0,
    }
    rejected, level, reason = evaluate_oos_rejection(perfect_oos, insufficient_sample=False)
    assert rejected is False
    assert level == STRONG_CANDIDATE
    
def test_insufficient_sample_hard_reject():
    perfect_but_no_sample = {
        "profit_factor": 2.0,
        "expectancy_r": 1.5,
        "max_drawdown_pct": 1.0,
        "negative_years": 0,
    }
    rejected, level, reason = evaluate_oos_rejection(perfect_but_no_sample, insufficient_sample=True)
    assert rejected is True
    assert level == HARD_REJECT
    assert reason == "OOS_SAMPLE_PENALTY_CRITICAL"
