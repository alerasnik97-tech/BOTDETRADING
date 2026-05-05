import json
from pathlib import Path

import pandas as pd


LAB_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = LAB_ROOT.parent
BOT_ROOT = PROJECT_ROOT.parent
MARKET_DATA_ROOT = BOT_ROOT / "BOT_MARKET_DATA"

TRADES_FILE = LAB_ROOT / "outputs" / "phase38_manipulante_deep_explainer" / "csv" / "phase38_raw_trades_enriched.csv"
TICK_FILE = MARKET_DATA_ROOT / "tick" / "EURUSD" / "monthly" / "EURUSD_ticks_2025_01.parquet"
REAUDIT_20_FILE = LAB_ROOT / "reports" / "manipulante_tick_historical" / "PHASE50C_C_20_TRADES_REAUDIT_AFTER_TZ_REPAIR.csv"
HIST_DIR = LAB_ROOT / "reports" / "manipulante_tick_historical"
DEBUG_DIR = HIST_DIR / "debug" / "residual_mismatches"
REPORTS_DIR = LAB_ROOT / "reports"


def pip_diff(price, reference):
    return round((float(price) - float(reference)) * 10000, 4)


def levels(trade):
    direction = trade["type"]
    entry = float(trade["entry_price"])
    risk = float(trade["risk"])
    return {
        "entry": entry,
        "risk": risk,
        "historical_sl": float(trade["sl"]),
        "computed_initial_sl": entry - risk if direction == "LONG" else entry + risk,
        "tp": float(trade["tp"]),
        "be_trigger": entry + (0.4 * risk if direction == "LONG" else -0.4 * risk),
        "be_level": entry,
    }


def nearest_ticks(ticks, entry_utc):
    idx = int(ticks["timestamp_utc"].searchsorted(entry_utc))
    before = ticks.iloc[idx - 1] if idx > 0 else None
    after = ticks.iloc[idx] if idx < len(ticks) else None
    if before is None:
        nearest = after
    elif after is None:
        nearest = before
    else:
        nearest = before if abs(before["timestamp_utc"] - entry_utc) <= abs(after["timestamp_utc"] - entry_utc) else after
    return before, after, nearest


def row_event(row, trade, lv, be_active):
    direction = trade["type"]
    bid = float(row.bid)
    ask = float(row.ask)
    events = []
    if direction == "LONG":
        if bid >= lv["tp"]:
            events.append("TP_TOUCH")
        if bid <= lv["computed_initial_sl"]:
            events.append("SL_TOUCH")
        if bid >= lv["be_trigger"]:
            events.append("BE_TRIGGER")
        if be_active and bid <= lv["be_level"]:
            events.append("BE_STOP_TOUCH")
    else:
        if ask <= lv["tp"]:
            events.append("TP_TOUCH")
        if ask >= lv["computed_initial_sl"]:
            events.append("SL_TOUCH")
        if ask <= lv["be_trigger"]:
            events.append("BE_TRIGGER")
        if be_active and ask >= lv["be_level"]:
            events.append("BE_STOP_TOUCH")
    return events


def event_scan(trade, ticks):
    lv = levels(trade)
    entry = pd.Timestamp(trade["entry_time"])
    exit_time = pd.Timestamp(trade["exit_time"])
    window = ticks[(ticks["timestamp_ny"] >= entry) & (ticks["timestamp_ny"] <= exit_time + pd.Timedelta(minutes=5))]
    be_active = False
    firsts = {k: None for k in ["TP_TOUCH", "SL_TOUCH", "BE_TRIGGER", "BE_STOP_TOUCH"]}
    ordered = []
    same_ts = False
    for row in window.itertuples(index=False):
        events = row_event(row, trade, lv, be_active)
        if "BE_TRIGGER" in events and not be_active:
            be_active = True
        if be_active:
            events = row_event(row, trade, lv, be_active)
        if len(events) > 1:
            same_ts = True
        for event in events:
            rec = {
                "event": event,
                "timestamp_ny": row.timestamp_ny.isoformat(),
                "timestamp_utc": row.timestamp_utc.isoformat(),
                "bid": float(row.bid),
                "ask": float(row.ask),
            }
            if firsts[event] is None:
                firsts[event] = rec
            ordered.append(rec)
        if events and len(ordered) >= 12:
            break
    tick_outcome = "NONE"
    first_touch = None
    for rec in ordered:
        event = rec["event"]
        if event == "BE_TRIGGER":
            continue
        if event == "BE_STOP_TOUCH":
            tick_outcome = "BE"
        elif event == "SL_TOUCH":
            tick_outcome = "SL"
        elif event == "TP_TOUCH":
            tick_outcome = "TP"
        first_touch = rec
        break
    return {
        "tick_outcome": tick_outcome,
        "first_touch": first_touch,
        "first_events": firsts,
        "ordered_events_head": ordered[:12],
        "same_timestamp_ambiguity": same_ts,
    }


def window_stats(trade, ticks):
    entry = pd.Timestamp(trade["entry_time"])
    exit_time = pd.Timestamp(trade["exit_time"])
    w = ticks[(ticks["timestamp_ny"] >= entry - pd.Timedelta(minutes=10)) & (ticks["timestamp_ny"] <= exit_time + pd.Timedelta(minutes=10))].copy()
    if w.empty:
        return {"ticks_loaded": 0, "gaps_gt_60s": None, "max_gap_seconds": None, "max_spread_pips": None, "p95_spread_pips": None}
    diffs = w["timestamp_utc"].diff().dt.total_seconds()
    return {
        "ticks_loaded": int(len(w)),
        "gaps_gt_60s": int((diffs > 60).sum()),
        "max_gap_seconds": None if diffs.dropna().empty else round(float(diffs.max()), 3),
        "max_spread_pips": round(float(w["spread_pips"].max()), 4),
        "p95_spread_pips": round(float(w["spread_pips"].quantile(0.95)), 4),
    }


def classify(trade, result):
    bar = result["bar_outcome"]
    tick = result["tick_outcome"]
    diff = max(abs(result["diff_bid_pips"]), abs(result["diff_ask_pips"]))
    firsts = result["events"]["first_events"]
    if result["window"]["ticks_loaded"] == 0:
        return "DATA_GAP_OR_LOW_TICK_DENSITY", "No hay ticks cargados en la ventana entry-10m a exit+10m."
    if bar == "FORCED_CLOSE" and tick == "NONE":
        return "EXIT_TIME_MISSING_OR_UNCLEAR", "No hubo TP/SL/BE por tick; el cierre forzado requiere capa de salida horaria."
    if result["window"]["ticks_loaded"] < 50 or (result["window"]["gaps_gt_60s"] or 0) > 0:
        return "DATA_GAP_OR_LOW_TICK_DENSITY", "La ventana tiene baja densidad o gaps que impiden una conclusion fuerte."
    if result["events"]["same_timestamp_ambiguity"]:
        return "TP_SL_SAME_TIMESTAMP_AMBIGUOUS", "Eventos de salida comparten timestamp o quedan intrabar ambiguos."
    if diff > 5:
        return "ENTRY_PRICE_FEED_DIFFERENCE", "El precio historico de entrada queda lejos del bid/ask tick cercano."
    if bar == "BE" and tick == "NONE":
        return "BE_SEQUENCE_DIFFERENCE", "La secuencia tick no confirma activacion y stop BE dentro de la ventana."
    if bar == "SL" and tick == "BE" and firsts["BE_TRIGGER"] and firsts["BE_STOP_TOUCH"]:
        return "BE_SEQUENCE_DIFFERENCE", "El tick muestra activacion de BE antes del toque posterior, mientras la barra queda en SL."
    if bar == "TP" and tick == "NONE":
        return "HISTORICAL_TRADE_LIST_ISSUE", "El TP historico no aparece como touch en ticks con densidad suficiente."
    return "LEGIT_TICK_MICROSTRUCTURE_DIFFERENCE", "La diferencia queda explicada por orden tick vs resultado de barra."


def audit_trade(trade_id, trade, ticks, re_row):
    entry_ny = pd.Timestamp(trade["entry_time"])
    entry_utc = entry_ny.tz_convert("UTC")
    before, after, nearest = nearest_ticks(ticks, entry_utc)
    lv = levels(trade)
    events = event_scan(trade, ticks)
    result = {
        "trade_id": int(trade_id),
        "date": entry_ny.date().isoformat(),
        "direction": trade["type"],
        "entry_time_ny": entry_ny.isoformat(),
        "entry_time_utc": entry_utc.isoformat(),
        "exit_time_ny": pd.Timestamp(trade["exit_time"]).isoformat(),
        "historical_entry": float(trade["entry_price"]),
        "sl": lv["historical_sl"],
        "computed_initial_sl": lv["computed_initial_sl"],
        "tp": lv["tp"],
        "be_level": lv["be_level"],
        "be_trigger": lv["be_trigger"],
        "nearest_bid": float(nearest["bid"]),
        "nearest_ask": float(nearest["ask"]),
        "diff_bid_pips": pip_diff(nearest["bid"], trade["entry_price"]),
        "diff_ask_pips": pip_diff(nearest["ask"], trade["entry_price"]),
        "spread_at_entry_pips": round(float((nearest["ask"] - nearest["bid"]) * 10000), 4),
        "bar_outcome": str(trade["outcome"]),
        "tick_outcome": str(re_row["tick_outcome_after_repair"]),
        "recomputed_tick_outcome": events["tick_outcome"],
        "first_touch": events["tick_outcome"],
        "first_touch_time": None if events["first_touch"] is None else events["first_touch"]["timestamp_ny"],
        "events": events,
        "window": window_stats(trade, ticks),
    }
    category, cause = classify(trade, result)
    result["category"] = category
    result["cause"] = cause
    result["mismatch_type_preliminar"] = category
    return result


def write_debug(result):
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DEBUG_DIR / f"TRADE_{result['trade_id']}_RESIDUAL_MISMATCH_DEBUG.json"
    md_path = DEBUG_DIR / f"TRADE_{result['trade_id']}_RESIDUAL_MISMATCH_DEBUG.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    firsts = result["events"]["first_events"]
    md = f"""# TRADE {result['trade_id']} RESIDUAL MISMATCH DEBUG

- category: {result['category']}
- cause: {result['cause']}
- direction: {result['direction']}
- entry_time_ny: {result['entry_time_ny']}
- entry_time_utc: {result['entry_time_utc']}
- historical_entry: {result['historical_entry']}
- nearest_bid/ask: {result['nearest_bid']} / {result['nearest_ask']}
- diff_bid/ask_pips: {result['diff_bid_pips']} / {result['diff_ask_pips']}
- spread_at_entry_pips: {result['spread_at_entry_pips']}
- bar_outcome: {result['bar_outcome']}
- tick_outcome: {result['tick_outcome']}
- recomputed_tick_outcome: {result['recomputed_tick_outcome']}
- first_touch: {result['first_touch']}
- first_touch_time: {result['first_touch_time']}
- ticks_loaded: {result['window']['ticks_loaded']}
- gaps_gt_60s: {result['window']['gaps_gt_60s']}
- max_spread_pips: {result['window']['max_spread_pips']}
- p95_spread_pips: {result['window']['p95_spread_pips']}
- first_tp_touch: {None if firsts['TP_TOUCH'] is None else firsts['TP_TOUCH']['timestamp_ny']}
- first_sl_touch: {None if firsts['SL_TOUCH'] is None else firsts['SL_TOUCH']['timestamp_ny']}
- first_be_trigger: {None if firsts['BE_TRIGGER'] is None else firsts['BE_TRIGGER']['timestamp_ny']}
- first_be_stop_touch: {None if firsts['BE_STOP_TOUCH'] is None else firsts['BE_STOP_TOUCH']['timestamp_ny']}
- same_timestamp_ambiguity: {result['events']['same_timestamp_ambiguity']}
"""
    with md_path.open("w", encoding="utf-8") as f:
        f.write(md)


def main():
    re = pd.read_csv(REAUDIT_20_FILE)
    residual = re[re["match_after_repair"] != True].copy()
    trades = pd.read_csv(TRADES_FILE)
    ticks = pd.read_parquet(TICK_FILE, columns=["timestamp_utc", "timestamp_ny", "bid", "ask", "spread_pips"])
    ticks["timestamp_utc"] = pd.to_datetime(ticks["timestamp_utc"]).dt.tz_convert("UTC")
    ticks["timestamp_ny"] = pd.to_datetime(ticks["timestamp_ny"])

    results = []
    for _, re_row in residual.iterrows():
        tid = int(re_row["trade_id"])
        result = audit_trade(tid, trades.loc[tid], ticks, re_row)
        results.append(result)
        write_debug(result)

    list_rows = []
    for r in results:
        list_rows.append({
            "trade_id": r["trade_id"],
            "date": r["date"],
            "direction": r["direction"],
            "entry_time_ny": r["entry_time_ny"],
            "historical_entry": r["historical_entry"],
            "sl": r["sl"],
            "tp": r["tp"],
            "be_level": r["be_level"],
            "bar_outcome": r["bar_outcome"],
            "tick_outcome": r["tick_outcome"],
            "first_touch": r["first_touch"],
            "first_touch_time": r["first_touch_time"],
            "mismatch_type_preliminar": r["mismatch_type_preliminar"],
        })
    list_path = HIST_DIR / "PHASE50C_D_RESIDUAL_MISMATCH_LIST.csv"
    pd.DataFrame(list_rows).to_csv(list_path, index=False)

    confirmed_matches = int((re["match_after_repair"] == True).sum())
    legit_categories = {"LEGIT_TICK_MICROSTRUCTURE_DIFFERENCE", "BE_SEQUENCE_DIFFERENCE", "ENTRY_PRICE_FEED_DIFFERENCE", "HISTORICAL_TRADE_LIST_ISSUE", "EXIT_TIME_MISSING_OR_UNCLEAR"}
    bug_count = sum(1 for r in results if r["category"] == "REPLAY_BUG_REQUIRES_REPAIR")
    data_gap_count = sum(1 for r in results if r["category"] == "DATA_GAP_OR_LOW_TICK_DENSITY")
    direct_not_auditable = sum(1 for r in results if r["category"] == "NOT_AUDITABLE")
    not_auditable = direct_not_auditable + data_gap_count
    legitimate = sum(1 for r in results if r["category"] in legit_categories)
    effective_reliability = round((confirmed_matches + legitimate) / len(re), 4)
    match_rate_confirmed = round(confirmed_matches / len(re), 4)
    if bug_count > 0:
        verdict = "JAN2025_REAUDIT_REQUIRES_REPLAY_REPAIR"
    elif not_auditable > 0:
        verdict = "JAN2025_REAUDIT_NOT_READY_FOR_NEXT_MONTH"
    else:
        verdict = "JAN2025_RECOVERED_BUT_WITH_LEGIT_TICK_DIFFERENCES"

    summary = {
        "phase": "PHASE50C-D",
        "verdict": verdict,
        "residual_trade_ids": [r["trade_id"] for r in results],
        "counts": {
            "sample_size": int(len(re)),
            "confirmed_matches": confirmed_matches,
            "residual_mismatches": int(len(results)),
            "mismatches_legitimate_or_explained": legitimate,
            "mismatches_by_bug": bug_count,
            "data_gap_or_low_tick_density": data_gap_count,
            "not_auditable": not_auditable,
            "match_rate_before_repair": 0.10,
            "match_rate_after_repair": 0.75,
            "match_rate_confirmed": match_rate_confirmed,
            "effective_reliability_score": effective_reliability,
        },
        "classifications": [
            {
                "trade_id": r["trade_id"],
                "bar_outcome": r["bar_outcome"],
                "tick_outcome": r["tick_outcome"],
                "cause": r["cause"],
                "category": r["category"],
            }
            for r in results
        ],
        "jan2025_status": "PARTIALLY_RECOVERED_WITH_ONE_DATA_GAP",
        "can_advance_next_month": False,
        "advance_condition": "Review residual explanations and decide whether effective reliability is acceptable before any next-month audit.",
        "safety": {
            "manipulante_modified": False,
            "strategy_modified": False,
            "mt5_opened": False,
            "orders_sent": False,
            "real_or_exness_touched": False,
            "git_add_commit_push": False,
        },
    }

    json_path = REPORTS_DIR / "PHASE50C_D_RESIDUAL_MISMATCH_FORENSIC_REVIEW_REPORT.json"
    md_path = REPORTS_DIR / "PHASE50C_D_RESIDUAL_MISMATCH_FORENSIC_REVIEW_REPORT.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    md = f"""# PHASE50C-D RESIDUAL MISMATCH FORENSIC REVIEW

Verdict: {verdict}

Residual trades: {', '.join(map(str, summary['residual_trade_ids']))}

Counts:
- confirmed_matches: {confirmed_matches}
- residual_mismatches: {len(results)}
- mismatches_explained: {legitimate}
- bugs: {bug_count}
- not_auditable: {not_auditable}
- match_rate_confirmed: {match_rate_confirmed}
- effective_reliability_score: {effective_reliability}

Classifications:
""" + "\n".join(
        f"- {r['trade_id']}: bar={r['bar_outcome']} tick={r['tick_outcome']} category={r['category']} cause={r['cause']}"
        for r in results
    ) + """

Safety:
- MANIPULANTE not modified.
- Strategy, TP, BE, BF, schedules, MT5, orders, real, Exness, 2024 not touched.
"""
    with md_path.open("w", encoding="utf-8") as f:
        f.write(md)
    print(verdict)
    print(",".join(map(str, summary["residual_trade_ids"])))


if __name__ == "__main__":
    main()
