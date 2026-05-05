import json
from pathlib import Path

import pandas as pd


LAB_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = LAB_ROOT.parent
BOT_ROOT = PROJECT_ROOT.parent
MARKET_DATA_ROOT = BOT_ROOT / "BOT_MARKET_DATA"
TICK_FILE = MARKET_DATA_ROOT / "tick" / "EURUSD" / "monthly" / "EURUSD_ticks_2025_01.parquet"
TRADES_FILE = LAB_ROOT / "outputs" / "phase38_manipulante_deep_explainer" / "csv" / "phase38_raw_trades_enriched.csv"
PREVIOUS_20_FILE = LAB_ROOT / "reports" / "manipulante_tick_historical" / "PHASE50C_B_20_TRADES_FORENSIC_SAMPLE.csv"
DEBUG_DIR = LAB_ROOT / "reports" / "manipulante_tick_historical" / "debug"
HIST_DIR = LAB_ROOT / "reports" / "manipulante_tick_historical"
REPORTS_DIR = LAB_ROOT / "reports"
PHASE49_REPORT_CANDIDATES = [
    BOT_ROOT / "BOT_V2_DAYTIME_LAB" / "reports" / "PHASE49F1B_TIMEZONE_REPAIR_REPORT.json",
    LAB_ROOT / "reports" / "PHASE49F1B_TIMEZONE_REPAIR_REPORT.json",
]


def load_phase49_report():
    for path in PHASE49_REPORT_CANDIDATES:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                report = json.load(f)
            return path, report
    raise FileNotFoundError("PHASE49F1B_TIMEZONE_REPAIR_REPORT.json not found")


def assert_phase49_ready():
    path, report = load_phase49_report()
    impact = report.get("phase50c_impact", {})
    cert = report.get("certification", {})
    ok = (
        report.get("verdict") == "TICK_TIMEZONE_REPAIR_OK_RECERTIFIED"
        and cert.get("timezone_consistent") is True
        and cert.get("timestamp_ny_timezone") == "America/New_York"
        and impact.get("phase50c_status") == "TEMPORARILY_INVALIDATED_BY_TICK_TIMEZONE_BUG"
    )
    if not ok:
        raise RuntimeError("ABORT: Phase49F1B timezone repair is not fully confirmed")
    return path, report


def pip_diff(price, reference):
    return round((float(price) - float(reference)) * 10000, 4)


def exact_price_stats(ticks, entry_time_ny, historical_entry):
    windows = {}
    for label, start, end in [
        ("entry_minute", entry_time_ny, entry_time_ny + pd.Timedelta(minutes=1)),
        ("plus_minus_5m", entry_time_ny - pd.Timedelta(minutes=5), entry_time_ny + pd.Timedelta(minutes=5)),
        ("plus_minus_10m", entry_time_ny - pd.Timedelta(minutes=10), entry_time_ny + pd.Timedelta(minutes=10)),
    ]:
        if label == "entry_minute":
            window = ticks[(ticks["timestamp_ny"] >= start) & (ticks["timestamp_ny"] < end)]
        else:
            window = ticks[(ticks["timestamp_ny"] >= start) & (ticks["timestamp_ny"] <= end)]
        if window.empty:
            windows[label] = {"rows": 0, "exists_bid": False, "exists_ask": False}
            continue
        bid_abs = (window["bid"] - historical_entry).abs()
        ask_abs = (window["ask"] - historical_entry).abs()
        windows[label] = {
            "rows": int(len(window)),
            "exists_bid": bool((bid_abs < 1e-9).any()),
            "exists_ask": bool((ask_abs < 1e-9).any()),
            "nearest_bid_diff_pips": round(float(bid_abs.min() * 10000), 4),
            "nearest_ask_diff_pips": round(float(ask_abs.min() * 10000), 4),
        }
    return windows


def trade_levels(trade):
    direction = trade["type"]
    entry = float(trade["entry_price"])
    risk = float(trade["risk"])
    tp = float(trade["tp"])
    initial_sl = entry - risk if direction == "LONG" else entry + risk
    be_trigger = entry + (0.4 * risk if direction == "LONG" else -0.4 * risk)
    return {
        "entry": entry,
        "risk": risk,
        "tp": tp,
        "initial_sl": initial_sl,
        "be_trigger": be_trigger,
        "be_level": entry,
    }


def first_touch(trade, ticks):
    direction = trade["type"]
    entry_time = pd.Timestamp(trade["entry_time"])
    exit_time = pd.Timestamp(trade["exit_time"])
    levels = trade_levels(trade)
    window = ticks[
        (ticks["timestamp_ny"] >= entry_time)
        & (ticks["timestamp_ny"] <= exit_time + pd.Timedelta(minutes=5))
    ]
    be_active = False
    be_activated_time = None
    for row in window.itertuples(index=False):
        bid = float(row.bid)
        ask = float(row.ask)
        t_ny = row.timestamp_ny
        t_utc = row.timestamp_utc
        if direction == "LONG":
            if not be_active and bid >= levels["be_trigger"]:
                be_active = True
                be_activated_time = t_ny
            current_sl = levels["be_level"] if be_active else levels["initial_sl"]
            if bid <= current_sl:
                reason = "BE" if be_active else "SL"
                return reason, t_ny, t_utc, bid, ask, be_activated_time
            if bid >= levels["tp"]:
                return "TP", t_ny, t_utc, bid, ask, be_activated_time
        else:
            if not be_active and ask <= levels["be_trigger"]:
                be_active = True
                be_activated_time = t_ny
            current_sl = levels["be_level"] if be_active else levels["initial_sl"]
            if ask >= current_sl:
                reason = "BE" if be_active else "SL"
                return reason, t_ny, t_utc, bid, ask, be_activated_time
            if ask <= levels["tp"]:
                return "TP", t_ny, t_utc, bid, ask, be_activated_time
    return "NONE", None, None, None, None, be_activated_time


def nearest_ticks(ticks, entry_time_utc):
    idx = int(ticks["timestamp_utc"].searchsorted(entry_time_utc))
    before = ticks.iloc[idx - 1] if idx > 0 else None
    after = ticks.iloc[idx] if idx < len(ticks) else None
    if before is None:
        nearest = after
    elif after is None:
        nearest = before
    else:
        nearest = before if abs(before["timestamp_utc"] - entry_time_utc) <= abs(after["timestamp_utc"] - entry_time_utc) else after
    return before, after, nearest


def summarize_trade(trade_id, trade, ticks):
    entry_ny = pd.Timestamp(trade["entry_time"])
    entry_utc = entry_ny.tz_convert("UTC")
    historical_entry = float(trade["entry_price"])
    before, after, nearest = nearest_ticks(ticks, entry_utc)
    tick_outcome, touch_ny, touch_utc, touch_bid, touch_ask, be_time = first_touch(trade, ticks)
    match = bool(str(trade["outcome"]) == tick_outcome)
    entry_leg = "ask" if trade["type"] == "LONG" else "bid"
    nearest_bid = float(nearest["bid"])
    nearest_ask = float(nearest["ask"])
    result = {
        "trade_id": int(trade_id),
        "direction": trade["type"],
        "entry_time_original": str(trade["entry_time"]),
        "entry_time_ny": entry_ny.isoformat(),
        "entry_time_utc": entry_utc.isoformat(),
        "historical_entry": historical_entry,
        "entry_leg": entry_leg,
        "nearest_bid": nearest_bid,
        "nearest_ask": nearest_ask,
        "diff_bid_pips": pip_diff(nearest_bid, historical_entry),
        "diff_ask_pips": pip_diff(nearest_ask, historical_entry),
        "spread_entry": round(float(nearest_ask - nearest_bid), 8),
        "spread_entry_pips": round(float((nearest_ask - nearest_bid) * 10000), 4),
        "nearest_before": None if before is None else {
            "timestamp_ny": before["timestamp_ny"].isoformat(),
            "timestamp_utc": before["timestamp_utc"].isoformat(),
            "bid": float(before["bid"]),
            "ask": float(before["ask"]),
        },
        "nearest_after": None if after is None else {
            "timestamp_ny": after["timestamp_ny"].isoformat(),
            "timestamp_utc": after["timestamp_utc"].isoformat(),
            "bid": float(after["bid"]),
            "ask": float(after["ask"]),
        },
        "price_presence": exact_price_stats(ticks, entry_ny, historical_entry),
        "levels": trade_levels(trade),
        "bar_outcome": str(trade["outcome"]),
        "tick_outcome_after_repair": tick_outcome,
        "first_touch": tick_outcome,
        "first_touch_time": None if touch_ny is None else touch_ny.isoformat(),
        "first_touch_time_utc": None if touch_utc is None else touch_utc.isoformat(),
        "first_touch_bid": touch_bid,
        "first_touch_ask": touch_ask,
        "be_activated_time": None if be_time is None else be_time.isoformat(),
        "match_status": "MATCH" if match else "MISMATCH",
        "match": match,
    }
    large_mismatch_fixed = max(abs(result["diff_bid_pips"]), abs(result["diff_ask_pips"])) < 13
    result["conclusion"] = (
        "Timezone repair removes the prior 13-pip timing mismatch; tick first-touch matches bar outcome."
        if match and large_mismatch_fixed else
        "Mismatch remains after timezone repair; do not expand sample yet."
    )
    return result


def write_trade_2310_reports(result):
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DEBUG_DIR / "PHASE50C_C_TRADE_2310_REAUDIT.json"
    md_path = DEBUG_DIR / "PHASE50C_C_TRADE_2310_REAUDIT.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    md = f"""# PHASE50C-C TRADE 2310 REAUDIT

- trade_id: {result['trade_id']}
- direction: {result['direction']}
- entry_time_ny: {result['entry_time_ny']}
- entry_time_utc: {result['entry_time_utc']}
- historical_entry: {result['historical_entry']}
- nearest_bid: {result['nearest_bid']}
- nearest_ask: {result['nearest_ask']}
- diff_bid_pips: {result['diff_bid_pips']}
- diff_ask_pips: {result['diff_ask_pips']}
- spread_entry_pips: {result['spread_entry_pips']}
- bar_outcome: {result['bar_outcome']}
- tick_outcome_after_repair: {result['tick_outcome_after_repair']}
- first_touch: {result['first_touch']}
- first_touch_time: {result['first_touch_time']}
- match_status: {result['match_status']}
- conclusion: {result['conclusion']}
"""
    with md_path.open("w", encoding="utf-8") as f:
        f.write(md)
    return md_path, json_path


def write_20_trade_csv(results):
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    previous = pd.read_csv(PREVIOUS_20_FILE) if PREVIOUS_20_FILE.exists() else pd.DataFrame()
    previous_map = {}
    if not previous.empty:
        previous_map = previous.set_index("trade_id").to_dict(orient="index")
    for result in results:
        prev = previous_map.get(result["trade_id"], {})
        rows.append({
            "trade_id": result["trade_id"],
            "entry_time_ny": result["entry_time_ny"],
            "direction": result["direction"],
            "bar_outcome": result["bar_outcome"],
            "tick_outcome_before_repair": prev.get("tick_outcome"),
            "tick_outcome_after_repair": result["tick_outcome_after_repair"],
            "match_before_repair": prev.get("match"),
            "match_after_repair": result["match"],
            "diff_bid_pips": result["diff_bid_pips"],
            "diff_ask_pips": result["diff_ask_pips"],
            "first_touch_time": result["first_touch_time"],
            "reclassified": prev.get("tick_outcome") != result["tick_outcome_after_repair"],
        })
    out = HIST_DIR / "PHASE50C_C_20_TRADES_REAUDIT_AFTER_TZ_REPAIR.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out, pd.DataFrame(rows)


def main():
    phase49_path, _ = assert_phase49_ready()
    trades = pd.read_csv(TRADES_FILE)
    ticks = pd.read_parquet(TICK_FILE, columns=["timestamp_utc", "timestamp_ny", "bid", "ask", "spread_pips"])
    ticks["timestamp_utc"] = pd.to_datetime(ticks["timestamp_utc"]).dt.tz_convert("UTC")
    ticks["timestamp_ny"] = pd.to_datetime(ticks["timestamp_ny"])

    trade_2310 = trades.loc[2310]
    trade_2310_result = summarize_trade(2310, trade_2310, ticks)
    write_trade_2310_reports(trade_2310_result)

    reaudited_20 = False
    results_20 = []
    csv_20 = None
    df_20 = pd.DataFrame()
    if trade_2310_result["match"] and max(abs(trade_2310_result["diff_bid_pips"]), abs(trade_2310_result["diff_ask_pips"])) < 13:
        sample_ids = list(pd.read_csv(PREVIOUS_20_FILE)["trade_id"].astype(int)) if PREVIOUS_20_FILE.exists() else list(trades[trades["year_month"] == "2025-01"].index[:20])
        results_20 = [summarize_trade(tid, trades.loc[tid], ticks) for tid in sample_ids]
        csv_20, df_20 = write_20_trade_csv(results_20)
        reaudited_20 = True

    before_match_rate = 0.10
    before_mismatches = 18
    after_match_rate = None
    after_mismatches = None
    reclassified_count = None
    outcome_changes = {}
    if reaudited_20:
        after_match_rate = round(float(df_20["match_after_repair"].mean()), 4)
        after_mismatches = int((~df_20["match_after_repair"]).sum())
        reclassified_count = int(df_20["reclassified"].sum())
        outcome_changes = {
            f"{before}->{after}": int(count)
            for (before, after), count in df_20.groupby(["tick_outcome_before_repair", "tick_outcome_after_repair"]).size().items()
        }

    if not reaudited_20:
        verdict = "TRADE_2310_REAUDIT_STILL_MISMATCH_REQUIRES_REVIEW"
    elif after_match_rate is not None and after_match_rate > before_match_rate:
        verdict = "JAN2025_20_TRADES_REAUDIT_RECOVERED"
    elif after_match_rate is not None:
        verdict = "JAN2025_20_TRADES_REAUDIT_STILL_DEGRADED"
    else:
        verdict = "JAN2025_REAUDIT_INCONCLUSIVE"

    final = {
        "phase": "PHASE50C-C",
        "verdict": verdict,
        "phase49_report": str(phase49_path),
        "scope": {
            "read_only_data": True,
            "manipulante_modified": False,
            "strategy_modified": False,
            "mt5_opened": False,
            "orders_sent": False,
            "real_or_exness_touched": False,
            "backtest_run": False,
            "optimization_run": False,
            "advanced_to_2024": False,
        },
        "trade_2310": trade_2310_result,
        "twenty_trade_reaudit": {
            "executed": reaudited_20,
            "csv_path": None if csv_20 is None else str(csv_20),
            "match_rate_before_repair": before_match_rate,
            "match_rate_after_repair": after_match_rate,
            "mismatches_before": before_mismatches,
            "mismatches_after": after_mismatches,
            "trades_reclassified": reclassified_count,
            "outcome_changes": outcome_changes,
            "conclusion": (
                "Match rate improved after timezone repair, but evidence remains degraded until broader reaudits confirm durability."
                if reaudited_20 and after_match_rate and after_match_rate > before_match_rate else
                "Trade 2310 did not clear expansion gate." if not reaudited_20 else
                "Timezone repair did not recover the 20-trade sample."
            ),
        },
        "phase50c_prior_10pct_invalidated": True,
        "do_not_touch": ["MANIPULANTE", "strategy logic", "TP/BE/BF", "MT5", "orders", "real", "Exness", "2024"],
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    final_json = REPORTS_DIR / "PHASE50C_C_POST_TIMEZONE_REPAIR_TICK_REAUDIT_REPORT.json"
    final_md = REPORTS_DIR / "PHASE50C_C_POST_TIMEZONE_REPAIR_TICK_REAUDIT_REPORT.md"
    with final_json.open("w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    md = f"""# PHASE50C-C POST TIMEZONE REPAIR TICK REAUDIT

Verdict: {verdict}

Trade 2310:
- Bar outcome: {trade_2310_result['bar_outcome']}
- Tick outcome after repair: {trade_2310_result['tick_outcome_after_repair']}
- Match: {trade_2310_result['match_status']}
- Entry NY/UTC: {trade_2310_result['entry_time_ny']} / {trade_2310_result['entry_time_utc']}
- Historical entry: {trade_2310_result['historical_entry']}
- Nearest bid/ask: {trade_2310_result['nearest_bid']} / {trade_2310_result['nearest_ask']}
- Diff bid/ask pips: {trade_2310_result['diff_bid_pips']} / {trade_2310_result['diff_ask_pips']}
- First touch: {trade_2310_result['first_touch']} at {trade_2310_result['first_touch_time']}

20-trade sample:
- Executed: {reaudited_20}
- Match before repair: {before_match_rate}
- Match after repair: {after_match_rate}
- Mismatches before/after: {before_mismatches} / {after_mismatches}
- Trades reclassified: {reclassified_count}

Interpretation:
- Prior 10% match rate is invalidated as a conclusion.
- MANIPULANTE remains unchanged.
- No strategy, TP, BE, BF, schedule, MT5, real, Exness, or 2024 scope was touched.
"""
    with final_md.open("w", encoding="utf-8") as f:
        f.write(md)
    print(verdict)


if __name__ == "__main__":
    main()
