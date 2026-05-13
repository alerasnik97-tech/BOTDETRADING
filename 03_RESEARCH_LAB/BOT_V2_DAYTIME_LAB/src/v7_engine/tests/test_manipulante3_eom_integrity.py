from __future__ import annotations

import csv
from pathlib import Path

from src.v7_engine.eom_integrity import (
    actual_window_before_intended,
    classify_eom,
    compute_net_r_metrics,
    metric_inclusion,
)


ROOT = Path(__file__).resolve().parents[5]
EOM_FIXED_TRADES = (
    ROOT
    / "03_RESEARCH_LAB"
    / "BOT_V2_DAYTIME_LAB"
    / "reports"
    / "v38_manipulante3_htf_ltf"
    / "eom_blocker_resolution"
    / "MAX_CONFIRMATION_EOM_FIXED_TRADES.csv"
)
EOM_FIXED_AUDIT = EOM_FIXED_TRADES.with_name("MAX_CONFIRMATION_EOM_FIXED_EOM_AUDIT.csv")
EOM_FIXED_RUNNER = EOM_FIXED_TRADES.with_name("max_confirmation_eom_fixed_rerun.py")


def base_record(**overrides):
    record = {
        "trade_id": "T1",
        "exit_reason": "EOM",
        "eom_type": "REAL_DATA_END",
        "tick_window_complete": True,
        "valid_closed_trade": True,
        "intended_position_end": "2026-01-02 20:00:00+00:00",
        "actual_tick_window_end": "2026-01-02 20:00:00.500000+00:00",
        "net_r": "1.0",
    }
    record.update(overrides)
    return record


def test_artificial_eom_is_excluded_from_metrics():
    artificial = base_record(eom_type="ARTIFICIAL_TRUNCATION", tick_window_complete=False, net_r="2.0")
    valid = base_record(trade_id="T2", net_r="-1.0")
    included, reason = metric_inclusion(artificial)
    assert included is False
    assert reason == "ARTIFICIAL_TRUNCATION"
    assert compute_net_r_metrics([artificial, valid])["N"] == 1
    assert compute_net_r_metrics([artificial, valid])["total_net_r"] == -1.0


def test_session_forced_exit_is_valid_eom_when_window_complete():
    record = base_record(exit_reason="TIME", eom_type="SESSION_FORCED_EXIT", net_r="0.2")
    artificial, reason = classify_eom(record)
    assert artificial is False
    assert reason == ""
    assert metric_inclusion(record)[0] is True


def test_incomplete_tick_window_marks_artificial_eom():
    record = base_record(tick_window_complete=False, eom_type="REAL_DATA_END")
    artificial, reason = classify_eom(record)
    assert artificial is True
    assert reason == "TICK_WINDOW_INCOMPLETE"


def test_actual_tick_window_before_intended_end_blocks_trade():
    record = base_record(actual_tick_window_end="2026-01-02 19:59:59.999000+00:00")
    assert actual_window_before_intended(record) is True
    included, reason = metric_inclusion(record)
    assert included is False
    assert reason == "ACTUAL_BEFORE_INTENDED"


def test_no_head_based_silent_truncation():
    if EOM_FIXED_RUNNER.exists():
        source = EOM_FIXED_RUNNER.read_text(encoding="utf-8")
        assert ".head(3000)" not in source
        assert ".head(" not in source[source.find("ticks_during") : source.find("trade = engine.close_position_with_costs")]


def test_eom_audit_counts_match_trades():
    if not EOM_FIXED_TRADES.exists() or not EOM_FIXED_AUDIT.exists():
        records = [base_record(), base_record(eom_type="ARTIFICIAL_TRUNCATION", tick_window_complete=False)]
        assert sum(1 for row in records if classify_eom(row)[0]) == 1
        return

    with EOM_FIXED_TRADES.open("r", encoding="utf-8", newline="") as fh:
        trades = list(csv.DictReader(fh))
    with EOM_FIXED_AUDIT.open("r", encoding="utf-8", newline="") as fh:
        audit_rows = list(csv.DictReader(fh))

    artificial_trades = sum(1 for row in trades if row.get("artificial_eom") == "True")
    artificial_audit = sum(int(row["artificial_eom_total"]) for row in audit_rows)
    assert artificial_audit == artificial_trades


def test_max_confirmation_has_zero_artificial_eom_in_metrics():
    if not EOM_FIXED_TRADES.exists():
        artificial = base_record(eom_type="ARTIFICIAL_TRUNCATION", tick_window_complete=False)
        assert metric_inclusion(artificial)[0] is False
        return

    with EOM_FIXED_TRADES.open("r", encoding="utf-8", newline="") as fh:
        trades = list(csv.DictReader(fh))
    offenders = [
        row
        for row in trades
        if row.get("artificial_eom") == "True" and row.get("included_in_metrics") == "True"
    ]
    assert offenders == []
