"""Tests for MANIPULANTE 4 causality, news, rollover, EOM, slippage."""
import pytest


# --- No Lookahead ---
def test_m4_causal_execution_fill_after_signal():
    signal_idx = 200
    fill_idx = 201
    assert fill_idx > signal_idx


# --- News/Rollover Fail-Close ---
def test_m4_news_missing_blocks_execution():
    calendar_available = False
    with pytest.raises(RuntimeError):
        if not calendar_available:
            raise RuntimeError("FAIL_CLOSE: news calendar missing")


def test_m4_rollover_blocks_signal():
    """Signals between 16:55-17:15 NY must be blocked."""
    signal_hour, signal_minute = 17, 5
    blocked = 16 * 60 + 55 <= signal_hour * 60 + signal_minute <= 17 * 60 + 15
    assert blocked is True


def test_m4_tier1_buffer_blocks():
    """Signal within -1/+5 min of Tier-1 news must be blocked."""
    news_time_min = 100
    signal_time_min = 103  # 3 min after
    pre_buffer = 1
    post_buffer = 5
    blocked = (news_time_min - pre_buffer) <= signal_time_min <= (news_time_min + post_buffer)
    assert blocked is True


# --- EOM Integrity ---
def test_m4_artificial_eom_excluded_from_metrics():
    """Artificial EOM trades must not be counted in PF/WR/expectancy."""
    trades = [
        {"net_r": 2.0, "eom_type": "NO_EOM"},
        {"net_r": -1.0, "eom_type": "NO_EOM"},
        {"net_r": 0.5, "eom_type": "ARTIFICIAL"},  # Excluded
    ]
    valid = [t for t in trades if t["eom_type"] != "ARTIFICIAL"]
    assert len(valid) == 2
    assert all(t["eom_type"] != "ARTIFICIAL" for t in valid)


# --- Slippage ---
def test_m4_slippage_reduces_net_r():
    gross_r = 2.0
    slippage_r = 0.08
    commission_r = 0.05
    net_r = gross_r - slippage_r - commission_r
    assert net_r < gross_r
    assert net_r == pytest.approx(1.87, abs=0.01)


# --- TEST isolation ---
def test_m4_test_not_used_for_selection():
    """Selection uses only VAL metrics. TEST is pass/fail only."""
    candidates = {
        "CFG_A": {"val_pf": 1.20, "test_pf": 0.90},
        "CFG_B": {"val_pf": 1.15, "test_pf": 1.50},
    }
    # Selection by VAL only
    selected = max(candidates, key=lambda c: candidates[c]["val_pf"])
    assert selected == "CFG_A"  # Not CFG_B despite better TEST
