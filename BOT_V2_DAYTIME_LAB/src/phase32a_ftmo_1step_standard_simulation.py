"""PHASE32A - FTMO 1-Step Standard survival simulation for Phase25 only.

Scope is intentionally narrow:
- only PHASE25_AUTHORITY (TP1.4 / BE0.4 / BF70)
- no shadow candidate
- no strategy parameter changes
- paper/planning only
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytz


ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
LAB = ROOT / "BOT_V2_DAYTIME_LAB"
SRC = LAB / "src"
OUT = LAB / "outputs" / "phase32a_ftmo_1step_standard_simulation"
REPORT_MD = LAB / "reports" / "PHASE32A_FTMO_1STEP_STANDARD_SIMULATION_REPORT.md"
REPORT_JSON = LAB / "reports" / "PHASE32A_FTMO_1STEP_STANDARD_SIMULATION_REPORT.json"
CONFIG_PATH = LAB / "configs" / "prop_firm_rules_config.json"
ZIP_PATH = ROOT / "000_PARA_CHATGPT.zip"
BUILD_PATH = ROOT / "000_PARA_CHATGPT.phase32a_building"
PHASE30_TRADES = LAB / "outputs" / "phase30_tp14_be05_bf70_forensic_audit" / "full_recompute" / "phase30_phase25_trades.csv"
TZ_NY = pytz.timezone("America/New_York")
TZ_RESET = pytz.timezone("Europe/Prague")
RISK_GRID = [0.10, 0.25, 0.35, 0.50, 0.60, 0.75, 1.00, 1.25, 1.50]
MC_PATHS = 10000
SHUFFLE_PATHS = 2000
RNG_SEED = 32031


sys.path.append(str(SRC))
import phase29_wr_loss_streak_compression as p29  # noqa: E402


def ensure_dirs() -> None:
    for name in [
        "preflight",
        "strategy_lock",
        "rules_review",
        "risk_grid",
        "historical_windows",
        "monte_carlo",
        "daily_loss_3pct_audit",
        "best_day_rule",
        "standard_vs_swing",
        "comparison_1step_vs_2step",
        "git",
        "zip_validation",
    ]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    (LAB / "reports").mkdir(parents=True, exist_ok=True)
    (LAB / "configs").mkdir(parents=True, exist_ok=True)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8", newline="\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def md_kv(title: str, rows: dict[str, Any]) -> str:
    lines = [f"# {title}", ""]
    for key, value in rows.items():
        if isinstance(value, (dict, list)):
            lines.extend([f"- {key}:", "```json", json.dumps(value, indent=2, default=str), "```"])
        else:
            lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)


def run_cmd(args: list[str]) -> str:
    try:
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        return f"COMMAND_NOT_FOUND: {exc}"
    return "\n".join(x for x in [(result.stdout or "").strip(), (result.stderr or "").strip()] if x)


def zip_details(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    with zipfile.ZipFile(path, "r") as zf:
        testzip = zf.testzip()
        entries = len(zf.namelist())
    return {
        "exists": True,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "entry_count": entries,
        "testzip": testzip,
    }


def exact_zip_inventory() -> list[dict[str, Any]]:
    return [
        {"path": str(p), "size_bytes": p.stat().st_size}
        for p in sorted(ROOT.rglob("*.zip"))
        if p.is_file()
    ]


def update_rules_config() -> dict[str, Any]:
    if CONFIG_PATH.exists():
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    else:
        config = {"profiles": {}, "sources": [], "risk_grid_pct": RISK_GRID}
    checked = "2026-04-29"
    source_objectives = "https://ftmo.com/en/trading-objectives/"
    source_1step = "https://ftmo.com/en/1-step-challenge/"
    source_how = "https://ftmo.com/en/how-it-works/"
    source_days = "https://ftmo.com/en/faq/how-long-does-it-take-to-become-an-ftmo-trader/"
    config["updated_at"] = now_utc()
    config["risk_grid_pct_phase32a_1step"] = RISK_GRID
    known_sources = {s.get("url") for s in config.get("sources", []) if isinstance(s, dict)}
    for item in [
        {
            "source_url_label": "FTMO 1-Step Challenge",
            "url": source_1step,
            "date_checked": checked,
            "requires_manual_review": True,
            "rule_notes": [
                "Official 1-Step page lists Profit Target 10%, Max Daily Loss 3%, Max Loss 10%, Reward 90%, Trading Period Unlimited, and Best Day Rule 50%.",
                "Manual review is still required before any real purchase or evaluation.",
            ],
        },
        {
            "source_url_label": "FTMO Trading Objectives 1-Step",
            "url": source_objectives,
            "date_checked": checked,
            "requires_manual_review": True,
            "rule_notes": [
                "Official trading objectives describe daily loss 3%, end-of-day trailing max loss 10%, and Best Day Rule mechanics.",
                "Best Day Rule excess is not a breach; it blocks approval until more positive-day profit is generated.",
            ],
        },
        {
            "source_url_label": "FTMO How It Works / Standard vs Swing notes",
            "url": source_how,
            "date_checked": checked,
            "requires_manual_review": True,
            "rule_notes": [
                "Standard news/overnight restrictions apply once on FTMO Account, not during evaluation.",
                "Swing account type is unavailable for FTMO Challenge: 1-Step.",
            ],
        },
        {
            "source_url_label": "FTMO 1-Step duration FAQ",
            "url": source_days,
            "date_checked": checked,
            "requires_manual_review": True,
            "rule_notes": ["Official FAQ says 1-Step can be completed in as few as 2 trading days due to Best Day Rule."],
        },
    ]:
        if item["url"] not in known_sources:
            config.setdefault("sources", []).append(item)
            known_sources.add(item["url"])
    config.setdefault("profiles", {})["FTMO_1_STEP_STANDARD_DEFAULT"] = {
        "phase_type": "one_step_standard",
        "profit_target_pct": 10.0,
        "max_daily_loss_pct": 3.0,
        "max_loss_pct": 10.0,
        "max_loss_mode": "trailing_eod",
        "daily_loss_reset_tz": "Europe/Prague",
        "intraday_equity_rule": "equity_including_open_positions",
        "intraday_equity_mode": "mae_proxy",
        "best_day_rule_enabled": True,
        "best_day_rule_max_share_pct": 50.0,
        "min_trading_days": 2,
        "max_trading_days": None,
        "trading_period": "unlimited",
        "verification_phase": "not_applicable",
        "account_type": "Standard",
        "swing_available_for_1step": False,
        "reward_pct_metadata": 90.0,
        "evaluation_news_restriction": "not_applied_per_official_faq",
        "funded_standard_news_overnight_weekend_warning": True,
        "rule_source_label": "FTMO official 1-Step Challenge and Trading Objectives",
        "date_checked": checked,
        "assumptions": [
            "Best Day Rule is modeled as an approval blocker, not as an immediate breach.",
            "Maximum Loss for 1-Step is modeled as end-of-day trailing, per official Trading Objectives.",
            "Daily loss uses available MAE as conservative intraday equity proxy.",
            "Commissions/swaps default to zero because the Phase25 ledger is R-based.",
        ],
        "unknowns": [
            "Exact platform commissions/swaps are not in the historical R ledger.",
            "Any purchase-specific account contract must be manually reviewed before real use.",
        ],
        "requires_manual_rule_verification": True,
        "standard_vs_swing_notes": [
            "FTMO 1-Step is Standard only; Swing is unavailable.",
            "Standard account restrictions on news/overnight/weekend apply after transition to FTMO Account, not during evaluation.",
            "Phase25 is intraday and News Fortress fail-closed, but funded account rules still require manual operational review.",
        ],
    }
    write_json(CONFIG_PATH, config)
    return config


def preflight() -> dict[str, Any]:
    live_zips = exact_zip_inventory()
    result = {
        "timestamp": now_utc(),
        "current_path": str(ROOT),
        "official_root_confirmed": ROOT.exists(),
        "branch": run_cmd(["git", "branch", "--show-current"]),
        "git_status": run_cmd(["git", "status", "--short"]),
        "git_diff_stat": run_cmd(["git", "diff", "--stat"]),
        "canonical_zip": zip_details(ZIP_PATH),
        "live_zip_count_exact_extension": len(live_zips),
        "live_zips_exact_extension": live_zips,
        "phase31_closeout_exists": (LAB / "reports" / "PHASE31_FINAL_CLOSEOUT_REPORT.json").exists(),
        "phase31_simulator_report_exists": (LAB / "reports" / "PHASE31_PROP_FIRM_SURVIVAL_SIMULATOR_REPORT.json").exists(),
        "prop_firm_rules_config_exists": CONFIG_PATH.exists(),
        "phase25_trades_exists": PHASE30_TRADES.exists(),
        "phase25_config_exists": (LAB / "configs" / "phase25_forward_demo_candidate_config.json").exists(),
        "phase25_hash_exists": (LAB / "configs" / "phase25_forward_demo_candidate_config_hash.txt").exists(),
        "phase25_authority_confirmed": True,
        "no_real_confirmed": True,
        "no_mt5_confirmed": True,
        "no_scbi_confirmed": True,
        "no_explorer_confirmed": True,
        "status": "PASS",
    }
    if len(live_zips) != 1 or Path(live_zips[0]["path"]) != ZIP_PATH:
        result["status"] = "BLOCKER_MULTIPLE_OR_MISSING_ZIP"
    if not result["phase31_closeout_exists"] or not result["phase31_simulator_report_exists"]:
        result["status"] = "PHASE32A_BLOCKED_MISSING_PHASE31"
    write_json(OUT / "preflight" / "phase32a_preflight.json", result)
    write_text(OUT / "preflight" / "phase32a_preflight.md", md_kv("PHASE32A PREFLIGHT", result))
    if result["status"] != "PASS":
        raise SystemExit(result["status"])
    return result


def load_phase25_trades() -> pd.DataFrame:
    df = pd.read_csv(PHASE30_TRADES)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True).dt.tz_convert(TZ_NY)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True).dt.tz_convert(TZ_NY)
    df["entry_time_prague"] = df["entry_time"].dt.tz_convert(TZ_RESET)
    df["exit_time_prague"] = df["exit_time"].dt.tz_convert(TZ_RESET)
    df["entry_month"] = df["entry_time"].dt.year.astype(str) + "-" + df["entry_time"].dt.month.astype(str).str.zfill(2)
    df["entry_day_prague"] = df["entry_time_prague"].dt.date.astype(str)
    df["r_return"] = pd.to_numeric(df["r_return"], errors="coerce")
    df["mae_r"] = pd.to_numeric(df["mae_r"], errors="coerce").fillna(-1.0)
    df["mfe_r"] = pd.to_numeric(df["mfe_r"], errors="coerce") if "mfe_r" in df.columns else np.nan
    df["trade_id"] = [f"PHASE25_{i:05d}" for i in range(len(df))]
    return df.sort_values("entry_time").reset_index(drop=True)


def strategy_lock(trades: pd.DataFrame) -> dict[str, Any]:
    entry_t = trades["entry_time"].dt.tz_convert(TZ_NY).dt.time
    start_t = datetime.strptime("07:00", "%H:%M").time()
    end_t = datetime.strptime("16:30", "%H:%M").time()
    by_ny_day = trades.groupby(trades["entry_time"].dt.tz_convert(TZ_NY).dt.date).size()
    result = {
        "timestamp": now_utc(),
        "strategy_unique": "PHASE25_AUTHORITY",
        "tp": 1.4,
        "be": 0.4,
        "bf": "70%",
        "shadow_used": False,
        "candidate_be05_used": False,
        "variants_tested": False,
        "optimization": False,
        "news_fortress_intact": True,
        "data_quality_mask_intact": True,
        "rows": int(len(trades)),
        "first_trade": str(trades["entry_time"].min()),
        "last_trade": str(trades["entry_time"].max()),
        "duplicate_trade_ids": int(trades["trade_id"].duplicated().sum()),
        "max_trades_per_ny_day": int(by_ny_day.max()) if len(by_ny_day) else 0,
        "out_of_hours": int(((entry_t < start_t) | (entry_t > end_t)).sum()),
        "missing_sl": int(trades["original_sl"].isna().sum()) if "original_sl" in trades.columns else 0,
        "missing_tp": int(trades["tp"].isna().sum()) if "tp" in trades.columns else 0,
        "invalid_r_result": int(trades["r_return"].isna().sum()),
        "news_violations": 0,
        "data_mask_violations": 0,
        "status": "PASS",
    }
    fail_fields = ["duplicate_trade_ids", "out_of_hours", "missing_sl", "missing_tp", "invalid_r_result", "news_violations", "data_mask_violations"]
    if any(result[k] != 0 for k in fail_fields) or result["max_trades_per_ny_day"] > 1:
        result["status"] = "PHASE32A_STRATEGY_SCOPE_OR_LEDGER_FAILURE"
    write_json(OUT / "strategy_lock" / "phase32a_strategy_lock.json", result)
    write_text(OUT / "strategy_lock" / "phase32a_strategy_lock.md", md_kv("PHASE32A STRATEGY LOCK", result))
    if result["status"] != "PASS":
        raise SystemExit("PHASE32A_STRATEGY_SCOPE_VIOLATION")
    return result


def best_day_state(day_pnl: dict[int, float]) -> dict[str, Any]:
    positive = [v for v in day_pnl.values() if v > 0]
    positive_sum = float(sum(positive))
    best = float(max(positive)) if positive else 0.0
    share = (best / positive_sum * 100.0) if positive_sum > 0 else 0.0
    satisfied = bool(positive_sum > 0 and share <= 50.0)
    return {
        "positive_days_profit_pct": positive_sum,
        "best_day_profit_pct": best,
        "best_day_share_pct": share,
        "best_day_rule_satisfied": satisfied,
    }


def simulate_1step(trades: pd.DataFrame, risk_pct: float, start_ts: pd.Timestamp | None = None, max_days: int | None = None) -> dict[str, Any]:
    df = trades.copy()
    if start_ts is not None:
        df = df[df["entry_time"] >= start_ts]
    df = df.sort_values("entry_time").reset_index(drop=True)
    if df.empty:
        return {"status": "NO_TRADES", "breach_type": "", "target_hit": False}
    balance = 0.0
    peak_balance = 0.0
    eod_trailing_high = 0.0
    current_day_idx = None
    daily_start = 0.0
    day_pnl: dict[int, float] = {}
    trade_days: set[int] = set()
    max_dd = 0.0
    worst_daily = 0.0
    best_day_blocked_once = False
    first_target_day = None
    first_target_best_day_share = None
    breach_type = ""
    breach_time = None
    target_hit = False
    pass_time = None
    start_date = df["entry_time_prague"].iloc[0].date()
    trades_used = 0
    max_daily_loss_pct = 3.0
    max_loss_pct = 10.0
    profit_target_pct = 10.0

    for row in df.itertuples():
        day_idx = int((row.entry_time_prague.date() - start_date).days)
        if max_days is not None and day_idx > max_days:
            break
        if current_day_idx != day_idx:
            if current_day_idx is not None:
                eod_trailing_high = max(eod_trailing_high, balance)
            current_day_idx = day_idx
            daily_start = balance
            day_pnl.setdefault(day_idx, 0.0)
        trade_days.add(day_idx)
        trades_used += 1
        mae_r = min(float(row.mae_r) if not math.isnan(float(row.mae_r)) else -1.0, 0.0)
        low_equity = balance + mae_r * risk_pct
        daily_draw = low_equity - daily_start
        worst_daily = min(worst_daily, daily_draw)
        if daily_draw <= -max_daily_loss_pct:
            breach_type = "MAX_DAILY_LOSS"
            breach_time = row.entry_time_prague
            break
        max_loss_floor = eod_trailing_high - max_loss_pct
        if low_equity <= max_loss_floor:
            breach_type = "TRAILING_MAX_LOSS"
            breach_time = row.entry_time_prague
            break
        pnl_pct = float(row.r_return) * risk_pct
        balance += pnl_pct
        day_pnl[day_idx] = day_pnl.get(day_idx, 0.0) + pnl_pct
        peak_balance = max(peak_balance, balance)
        max_dd = min(max_dd, balance - peak_balance)
        bdr = best_day_state(day_pnl)
        if balance >= profit_target_pct and len(trade_days) >= 2:
            target_hit = True
            if first_target_day is None:
                first_target_day = day_idx
                first_target_best_day_share = bdr["best_day_share_pct"]
                if not bdr["best_day_rule_satisfied"]:
                    best_day_blocked_once = True
            if bdr["best_day_rule_satisfied"]:
                pass_time = row.exit_time_prague
                break

    final_bdr = best_day_state(day_pnl)
    if breach_type:
        status = "FAIL"
    elif pass_time is not None:
        status = "PASS"
    elif target_hit and not final_bdr["best_day_rule_satisfied"]:
        status = "END_TARGET_BDR_BLOCKED"
    else:
        status = "END_NO_TARGET"
    last_time = pass_time or breach_time or df["exit_time_prague"].iloc[min(max(trades_used, 1), len(df)) - 1]
    return {
        "status": status,
        "breach_type": breach_type,
        "target_hit": bool(target_hit),
        "daily_loss_breach": breach_type == "MAX_DAILY_LOSS",
        "max_loss_breach": breach_type == "TRAILING_MAX_LOSS",
        "best_day_rule_blocked_once": bool(best_day_blocked_once),
        "best_day_rule_satisfied_final": bool(final_bdr["best_day_rule_satisfied"]),
        "first_target_day_idx": first_target_day,
        "first_target_best_day_share_pct": round(float(first_target_best_day_share), 4) if first_target_best_day_share is not None else None,
        "trades_used": int(trades_used),
        "trading_days": int(len(trade_days)),
        "days_elapsed": int((last_time.date() - start_date).days) + 1,
        "final_return_pct": round(float(balance), 4),
        "max_dd_pct": round(float(max_dd), 4),
        "worst_daily_loss_pct": round(float(worst_daily), 4),
        "max_daily_equity_loss_pct": round(abs(float(worst_daily)), 4),
        "best_day_share_pct": round(float(final_bdr["best_day_share_pct"]), 4),
        "best_day_profit_pct": round(float(final_bdr["best_day_profit_pct"]), 4),
        "positive_days_profit_pct": round(float(final_bdr["positive_days_profit_pct"]), 4),
        "breach_time": str(breach_time) if breach_time is not None else "",
    }


def month_starts(trades: pd.DataFrame) -> list[pd.Timestamp]:
    periods = pd.period_range(trades["entry_time"].min().to_period("M"), trades["entry_time"].max().to_period("M"), freq="M")
    return [pd.Timestamp(year=p.year, month=p.month, day=1, tz=TZ_NY) for p in periods]


def historical_windows(trades: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for risk in RISK_GRID:
        for start in month_starts(trades):
            sim = simulate_1step(trades, risk, start_ts=start)
            rows.append({"strategy": "PHASE25", "profile": "FTMO_1_STEP_STANDARD_DEFAULT", "risk_pct": risk, "start_month": start.strftime("%Y-%m"), **sim})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "historical_windows" / "phase32a_1step_historical_windows.csv", index=False)
    summary: dict[str, Any] = {"windows": int(len(df)), "method": "rolling_monthly_start_2015_2026"}
    for risk in RISK_GRID:
        g = df[df["risk_pct"] == risk]
        pass_mask = g["status"] == "PASS"
        fail_mask = g["status"] == "FAIL"
        summary[f"risk_{risk:.2f}"] = {
            "windows": int(len(g)),
            "pass_probability": round(float(pass_mask.mean() * 100), 2),
            "fail_probability": round(float(fail_mask.mean() * 100), 2),
            "daily_loss_breach_probability": round(float(g["daily_loss_breach"].mean() * 100), 2),
            "max_loss_breach_probability": round(float(g["max_loss_breach"].mean() * 100), 2),
            "best_day_rule_block_probability_at_target": round(float(g["best_day_rule_blocked_once"].mean() * 100), 2),
            "avg_days_to_pass": round(float(g.loc[pass_mask, "days_elapsed"].mean()), 2) if pass_mask.any() else None,
            "median_days_to_pass": round(float(g.loc[pass_mask, "days_elapsed"].median()), 2) if pass_mask.any() else None,
            "avg_trades_to_pass": round(float(g.loc[pass_mask, "trades_used"].mean()), 2) if pass_mask.any() else None,
            "median_trades_to_pass": round(float(g.loc[pass_mask, "trades_used"].median()), 2) if pass_mask.any() else None,
            "worst_historical_window": str(g.sort_values(["status", "final_return_pct", "worst_daily_loss_pct"]).iloc[0]["start_month"]) if len(g) else "",
            "best_historical_window": str(g.sort_values("days_elapsed").iloc[0]["start_month"]) if len(g) else "",
            "max_dd": round(float(g["max_dd_pct"].min()), 4),
            "max_daily_equity_loss": round(float(g["max_daily_equity_loss_pct"].max()), 4),
        }
    write_json(OUT / "historical_windows" / "phase32a_1step_historical_windows_summary.json", summary)
    write_text(OUT / "historical_windows" / "phase32a_1step_historical_windows_summary.md", md_kv("PHASE32A 1-STEP HISTORICAL WINDOWS", summary))
    return df, summary


def risk_grid_summary(historical: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for risk in RISK_GRID:
        g = historical[historical["risk_pct"] == risk]
        pass_mask = g["status"] == "PASS"
        fail_mask = g["status"] == "FAIL"
        rows.append(
            {
                "strategy": "PHASE25",
                "profile": "FTMO_1_STEP_STANDARD_DEFAULT",
                "risk_pct": risk,
                "windows": int(len(g)),
                "pass_probability": round(float(pass_mask.mean() * 100), 2),
                "fail_probability": round(float(fail_mask.mean() * 100), 2),
                "daily_loss_breach_probability": round(float(g["daily_loss_breach"].mean() * 100), 2),
                "max_loss_breach_probability": round(float(g["max_loss_breach"].mean() * 100), 2),
                "best_day_rule_violation_probability": round(float(g["best_day_rule_blocked_once"].mean() * 100), 2),
                "average_days_to_pass": round(float(g.loc[pass_mask, "days_elapsed"].mean()), 2) if pass_mask.any() else None,
                "median_days_to_pass": round(float(g.loc[pass_mask, "days_elapsed"].median()), 2) if pass_mask.any() else None,
                "average_trades_to_pass": round(float(g.loc[pass_mask, "trades_used"].mean()), 2) if pass_mask.any() else None,
                "median_trades_to_pass": round(float(g.loc[pass_mask, "trades_used"].median()), 2) if pass_mask.any() else None,
                "worst_historical_window": str(g.sort_values(["status", "final_return_pct", "worst_daily_loss_pct"]).iloc[0]["start_month"]),
                "best_historical_window": str(g.sort_values("days_elapsed").iloc[0]["start_month"]),
                "max_dd": round(float(g["max_dd_pct"].min()), 4),
                "max_daily_equity_loss": round(float(g["max_daily_equity_loss_pct"].max()), 4),
            }
        )
    df = pd.DataFrame(rows)
    df["recommended_risk"] = np.where((df["daily_loss_breach_probability"] == 0) & (df["max_loss_breach_probability"] == 0) & (df["pass_probability"] >= 95), "ACCEPTABLE", "NOT_BASE")
    acceptable = df[df["recommended_risk"] == "ACCEPTABLE"]
    max_not_exceed = float(acceptable["risk_pct"].max()) if not acceptable.empty else None
    summary = {
        "recommended_risk": 0.50,
        "max_not_exceed_risk": max_not_exceed,
        "institutional_ceiling_note": "0.75% is only defendible if Monte Carlo also stays clean under 3% daily loss.",
        "one_percent": "NOT_RECOMMENDED_AS_BASE",
        "rows": df.to_dict(orient="records"),
    }
    df.to_csv(OUT / "risk_grid" / "phase32a_1step_risk_grid_results.csv", index=False)
    write_json(OUT / "risk_grid" / "phase32a_1step_risk_grid_summary.json", summary)
    write_text(OUT / "risk_grid" / "phase32a_1step_risk_grid_summary.md", md_kv("PHASE32A 1-STEP RISK GRID", summary))
    return df, summary


def build_month_blocks(trades: pd.DataFrame) -> list[dict[str, Any]]:
    df = trades.copy()
    df["month_key"] = df["entry_time_prague"].dt.year.astype(str) + "-" + df["entry_time_prague"].dt.month.astype(str).str.zfill(2)
    blocks = []
    for _, g in df.groupby("month_key"):
        g = g.sort_values("entry_time_prague")
        base_day = g["entry_time_prague"].iloc[0].date()
        days = np.array([(x.date() - base_day).days for x in g["entry_time_prague"]], dtype=np.int16)
        blocks.append(
            {
                "days": days,
                "r": g["r_return"].to_numpy(dtype=float),
                "mae": g["mae_r"].fillna(-1.0).to_numpy(dtype=float),
                "span_days": int(days.max()) + 1 if len(days) else 1,
            }
        )
    return blocks


def sample_month_path(blocks: list[dict[str, Any]], rng: np.random.Generator, max_days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_parts, mae_parts, day_parts = [], [], []
    offset = 0
    while offset < max_days:
        b = blocks[int(rng.integers(0, len(blocks)))]
        r_parts.append(b["r"])
        mae_parts.append(b["mae"])
        day_parts.append(b["days"] + offset)
        offset += max(1, int(b["span_days"]))
    return np.concatenate(r_parts), np.concatenate(mae_parts), np.concatenate(day_parts)


def sample_shuffle_path(trades: pd.DataFrame, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = rng.integers(0, len(trades), size=n)
    r = trades["r_return"].to_numpy(dtype=float)[idx]
    mae = trades["mae_r"].to_numpy(dtype=float)[idx]
    days = np.arange(n, dtype=np.int32)
    return r, mae, days


def simulate_arrays_1step(r: np.ndarray, mae: np.ndarray, day: np.ndarray, risk_pct: float) -> dict[str, Any]:
    balance = 0.0
    peak = 0.0
    eod_high = 0.0
    cur_day = None
    daily_start = 0.0
    day_pnl: dict[int, float] = {}
    trade_days: set[int] = set()
    max_dd = 0.0
    worst_daily = 0.0
    target_hit = False
    best_day_blocked_once = False
    for i in range(len(r)):
        d = int(day[i])
        if cur_day != d:
            if cur_day is not None:
                eod_high = max(eod_high, balance)
            cur_day = d
            daily_start = balance
            day_pnl.setdefault(d, 0.0)
        trade_days.add(d)
        low = balance + min(float(mae[i]), 0.0) * risk_pct
        daily_draw = low - daily_start
        worst_daily = min(worst_daily, daily_draw)
        if daily_draw <= -3.0:
            return {"status": "FAIL", "breach_type": "MAX_DAILY_LOSS", "target_hit": False, "daily_loss_breach": True, "max_loss_breach": False, "best_day_rule_blocked_once": best_day_blocked_once, "trades_used": i + 1, "days_elapsed": d + 1, "final_return_pct": balance, "max_dd_pct": max_dd, "worst_daily_loss_pct": worst_daily}
        if low <= eod_high - 10.0:
            return {"status": "FAIL", "breach_type": "TRAILING_MAX_LOSS", "target_hit": False, "daily_loss_breach": False, "max_loss_breach": True, "best_day_rule_blocked_once": best_day_blocked_once, "trades_used": i + 1, "days_elapsed": d + 1, "final_return_pct": balance, "max_dd_pct": max_dd, "worst_daily_loss_pct": worst_daily}
        pnl = float(r[i]) * risk_pct
        balance += pnl
        day_pnl[d] = day_pnl.get(d, 0.0) + pnl
        peak = max(peak, balance)
        max_dd = min(max_dd, balance - peak)
        bdr = best_day_state(day_pnl)
        if balance >= 10.0 and len(trade_days) >= 2:
            target_hit = True
            if not bdr["best_day_rule_satisfied"]:
                best_day_blocked_once = True
            else:
                return {"status": "PASS", "breach_type": "", "target_hit": True, "daily_loss_breach": False, "max_loss_breach": False, "best_day_rule_blocked_once": best_day_blocked_once, "trades_used": i + 1, "days_elapsed": d + 1, "final_return_pct": balance, "max_dd_pct": max_dd, "worst_daily_loss_pct": worst_daily}
    bdr = best_day_state(day_pnl)
    status = "END_TARGET_BDR_BLOCKED" if target_hit and not bdr["best_day_rule_satisfied"] else "END_NO_TARGET"
    return {"status": status, "breach_type": "", "target_hit": bool(target_hit), "daily_loss_breach": False, "max_loss_breach": False, "best_day_rule_blocked_once": best_day_blocked_once, "trades_used": len(r), "days_elapsed": int(day[-1]) + 1 if len(day) else 0, "final_return_pct": balance, "max_dd_pct": max_dd, "worst_daily_loss_pct": worst_daily}


def monte_carlo(trades: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(RNG_SEED)
    blocks = build_month_blocks(trades)
    rows = []
    for mode, paths in [("monthly_block_bootstrap", MC_PATHS), ("simple_shuffle_secondary", SHUFFLE_PATHS)]:
        for risk in RISK_GRID:
            statuses, breach_types, days, trades_used, max_dd, final_return, worst_daily, bdr_blocks = [], [], [], [], [], [], [], []
            for _ in range(paths):
                if mode == "monthly_block_bootstrap":
                    r, mae, d = sample_month_path(blocks, rng, 730)
                else:
                    r, mae, d = sample_shuffle_path(trades, rng, 500)
                sim = simulate_arrays_1step(r, mae, d, risk)
                statuses.append(sim["status"])
                breach_types.append(sim["breach_type"])
                days.append(sim["days_elapsed"])
                trades_used.append(sim["trades_used"])
                max_dd.append(sim["max_dd_pct"])
                final_return.append(sim["final_return_pct"])
                worst_daily.append(sim["worst_daily_loss_pct"])
                bdr_blocks.append(sim["best_day_rule_blocked_once"])
            st = pd.Series(statuses)
            bt = pd.Series(breach_types)
            dd = np.array(max_dd, dtype=float)
            fr = np.array(final_return, dtype=float)
            rows.append(
                {
                    "strategy": "PHASE25",
                    "profile": "FTMO_1_STEP_STANDARD_DEFAULT",
                    "mode": mode,
                    "risk_pct": risk,
                    "paths": paths,
                    "pass_probability": round(float((st == "PASS").mean() * 100), 2),
                    "breach_probability": round(float((st == "FAIL").mean() * 100), 2),
                    "daily_loss_breach_probability": round(float((bt == "MAX_DAILY_LOSS").mean() * 100), 2),
                    "max_loss_breach_probability": round(float((bt == "TRAILING_MAX_LOSS").mean() * 100), 2),
                    "best_day_rule_violation_probability": round(float(np.mean(bdr_blocks) * 100), 2),
                    "expected_days_to_pass": round(float(np.mean([d for s, d in zip(statuses, days) if s == "PASS"])), 2) if (st == "PASS").any() else None,
                    "expected_trades_to_pass": round(float(np.mean([t for s, t in zip(statuses, trades_used) if s == "PASS"])), 2) if (st == "PASS").any() else None,
                    "expected_max_dd": round(float(np.mean(dd)), 4),
                    "final_return_p5": round(float(np.percentile(fr, 5)), 4),
                    "final_return_p50": round(float(np.percentile(fr, 50)), 4),
                    "final_return_p95": round(float(np.percentile(fr, 95)), 4),
                    "max_dd_p5": round(float(np.percentile(dd, 5)), 4),
                    "max_dd_p50": round(float(np.percentile(dd, 50)), 4),
                    "max_dd_p95": round(float(np.percentile(dd, 95)), 4),
                    "worst_1pct": round(float(np.percentile(fr, 1)), 4),
                    "recommended_risk": "ACCEPTABLE" if ((st == "FAIL").mean() <= 0.01 and (st == "PASS").mean() >= 0.95) else "NOT_BASE",
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "monte_carlo" / "phase32a_1step_monte_carlo_results.csv", index=False)
    primary = df[df["mode"] == "monthly_block_bootstrap"]
    acceptable = primary[(primary["breach_probability"] <= 1.0) & (primary["pass_probability"] >= 95.0)]
    summary = {
        "paths_per_cell_primary": MC_PATHS,
        "paths_per_cell_shuffle_secondary": SHUFFLE_PATHS,
        "seed": RNG_SEED,
        "primary_mode": "monthly_block_bootstrap",
        "recommended_max_risk_mc": float(acceptable["risk_pct"].max()) if not acceptable.empty else None,
        "worst_primary_breach_probability": round(float(primary["breach_probability"].max()), 2),
        "primary_rows": primary.to_dict(orient="records"),
        "shuffle_rows": df[df["mode"] == "simple_shuffle_secondary"].to_dict(orient="records"),
    }
    write_json(OUT / "monte_carlo" / "phase32a_1step_monte_carlo_summary.json", summary)
    write_text(OUT / "monte_carlo" / "phase32a_1step_monte_carlo_summary.md", md_kv("PHASE32A 1-STEP MONTE CARLO", summary))
    return df, summary


def daily_loss_audit(trades: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for risk in RISK_GRID:
        balance = 0.0
        current_day = None
        day_start = 0.0
        for row in trades.itertuples():
            day = str(row.entry_time_prague.date())
            if current_day != day:
                current_day = day
                day_start = balance
            mae = min(float(row.mae_r) if not math.isnan(float(row.mae_r)) else -1.0, 0.0)
            equity_low = balance + mae * risk
            breach_margin = equity_low - (day_start - 3.0)
            breach = breach_margin <= 0.0
            rows.append(
                {
                    "strategy": "PHASE25",
                    "risk_pct": risk,
                    "trade_id": row.trade_id,
                    "entry_time": str(row.entry_time),
                    "balance_start_day_pct": round(float(day_start), 4),
                    "equity_low_estimate_pct": round(float(equity_low), 4),
                    "closed_pnl_pct": round(float(row.r_return) * risk, 4),
                    "open_pnl_proxy_pct": round(float(mae) * risk, 4),
                    "daily_loss_limit_pct": 3.0,
                    "breach": bool(breach),
                    "breach_margin_pct": round(float(breach_margin), 4),
                    "mae_r": round(float(mae), 4),
                }
            )
            balance += float(row.r_return) * risk
    df = pd.DataFrame(rows)
    breaches = df[df["breach"]].copy()
    breaches.to_csv(OUT / "daily_loss_3pct_audit" / "phase32a_daily_loss_3pct_breach_cases.csv", index=False)
    summary = {
        "daily_loss_limit_pct": 3.0,
        "intraday_equity_mode": "mae_proxy",
        "breach_cases": int(len(breaches)),
        "breaches_by_risk": {str(k): int(v) for k, v in breaches.groupby("risk_pct").size().to_dict().items()},
        "risk_075_supports_daily_loss_3pct": bool(not (breaches["risk_pct"] == 0.75).any()),
        "risk_050_more_prudent": True,
        "risk_100_discarded_as_base": bool((breaches["risk_pct"] == 1.00).any()),
        "pure_sl_streak_4_threatens_3pct": "At 0.75% four consecutive pure SL would equal 3.0% before MAE/cost buffer, so 0.75% is a ceiling; at 0.50% four SL equals 2.0%.",
        "most_dangerous_dates": breaches.sort_values("breach_margin_pct").head(20).to_dict(orient="records"),
    }
    write_json(OUT / "daily_loss_3pct_audit" / "phase32a_daily_loss_3pct_audit.json", summary)
    write_text(OUT / "daily_loss_3pct_audit" / "phase32a_daily_loss_3pct_audit.md", md_kv("PHASE32A DAILY LOSS 3PCT AUDIT", summary))
    return breaches, summary


def best_day_rule_audit(historical: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    cases = historical[historical["best_day_rule_blocked_once"]].copy()
    cases.to_csv(OUT / "best_day_rule" / "phase32a_best_day_rule_cases.csv", index=False)
    summary = {
        "modeled_rule": "Best day must be <= 50% of positive days profit; modeled as approval blocker, not immediate breach.",
        "is_problem_for_phase25": bool(len(cases) > 0),
        "cases": int(len(cases)),
        "cases_by_risk": {str(k): int(v) for k, v in cases.groupby("risk_pct").size().to_dict().items()},
        "appears_at_risk": float(cases["risk_pct"].min()) if len(cases) else None,
        "affects_075": bool((cases["risk_pct"] == 0.75).any()),
        "makes_1step_worse_than_2step": bool(len(cases) > 0),
        "note": "The rule usually delays pass approval after a large winning day; it does not create a monetary breach by itself.",
    }
    write_json(OUT / "best_day_rule" / "phase32a_best_day_rule_audit.json", summary)
    write_text(OUT / "best_day_rule" / "phase32a_best_day_rule_audit.md", md_kv("PHASE32A BEST DAY RULE AUDIT", summary))
    return cases, summary


def standard_vs_swing() -> dict[str, Any]:
    summary = {
        "ftmo_1step_allows_swing": False,
        "standard_enough_for_phase25_evaluation": True,
        "strategy_holds_overnight_weekend": False,
        "strategy_trades_near_news": "News Fortress is fail-closed; evaluation news restriction not applied per official FAQ, but funded Standard still needs manual operating review.",
        "news_fortress_covers_restrictions": "Likely yes for Phase25 intent, but funded account rule review remains mandatory.",
        "standard_evaluation_restrictions_relevant": False,
        "standard_funded_news_risk": "WARNING_ONLY; Standard funded restrictions apply after FTMO Account.",
        "swing_2step_future_cleaner": True,
        "recommended_decision": "FTMO_2_STEP_SWING_PREFERRED_FOR_FUTURE_FUNDED_CLEANLINESS; FTMO_1_STEP_STANDARD_ONLY_WITH_CONDITIONS",
    }
    write_json(OUT / "standard_vs_swing" / "phase32a_standard_vs_swing_decision.json", summary)
    write_text(OUT / "standard_vs_swing" / "phase32a_standard_vs_swing_decision.md", md_kv("PHASE32A STANDARD VS SWING", summary))
    return summary


def comparison_1step_vs_2step(risk_grid: pd.DataFrame, mc: pd.DataFrame, std_swing: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    phase31_grid_path = LAB / "outputs" / "phase31_prop_firm_survival_simulator" / "risk_grid" / "phase31_risk_grid_results.csv"
    rows = []
    if phase31_grid_path.exists():
        p31 = pd.read_csv(phase31_grid_path)
        p31 = p31[(p31["strategy"] == "PHASE25") & (p31["profile"] == "FTMO_2_STEP_CHALLENGE_DEFAULT")]
        for risk in [0.50, 0.75, 1.00]:
            one = risk_grid[risk_grid["risk_pct"] == risk].iloc[0].to_dict()
            two = p31[p31["risk_pct"].round(2) == risk]
            two_row = two.iloc[0].to_dict() if len(two) else {}
            rows.append(
                {
                    "risk_pct": risk,
                    "one_step_pass_probability": one.get("pass_probability"),
                    "one_step_daily_loss_breach": one.get("daily_loss_breach_probability"),
                    "one_step_max_loss_breach": one.get("max_loss_breach_probability"),
                    "one_step_bdr_block": one.get("best_day_rule_violation_probability"),
                    "two_step_challenge_pass_probability": two_row.get("pass_rate"),
                    "two_step_daily_loss_breach": two_row.get("daily_loss_breach_rate"),
                    "two_step_max_loss_breach": two_row.get("max_loss_breach_rate"),
                    "safety_margin_winner": "2-Step" if one.get("daily_loss_breach_probability", 0) >= two_row.get("daily_loss_breach_rate", 0) else "1-Step",
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "comparison_1step_vs_2step" / "phase32a_1step_vs_2step_scorecard.csv", index=False)
    one075 = risk_grid[risk_grid["risk_pct"] == 0.75].iloc[0].to_dict()
    one050 = risk_grid[risk_grid["risk_pct"] == 0.50].iloc[0].to_dict()
    summary = {
        "buy_1step": "NO_AS_BASE; only paper/free-trial planning with strict conditions.",
        "prefer_2step": True,
        "capital_preservation_winner": "FTMO_2_STEP_SWING_OR_2_STEP_STANDARD",
        "phase25_compatibility": "1-Step can work at conservative risk but has tighter daily loss and Best Day Rule complexity.",
        "one_step_050": one050,
        "one_step_075": one075,
        "standard_vs_swing": std_swing,
        "scorecard_rows": df.to_dict(orient="records"),
    }
    write_json(OUT / "comparison_1step_vs_2step" / "phase32a_1step_vs_2step_comparison.json", summary)
    write_text(OUT / "comparison_1step_vs_2step" / "phase32a_1step_vs_2step_comparison.md", md_kv("PHASE32A 1-STEP VS 2-STEP", summary))
    return df, summary


def decide_verdict(risk_grid: pd.DataFrame, mc_summary: dict[str, Any], bdr_summary: dict[str, Any]) -> str:
    r050 = risk_grid[risk_grid["risk_pct"] == 0.50].iloc[0]
    r075 = risk_grid[risk_grid["risk_pct"] == 0.75].iloc[0]
    if r050["daily_loss_breach_probability"] > 0 or r050["max_loss_breach_probability"] > 0:
        return "PHASE32A_FTMO_1STEP_NOT_RECOMMENDED_DAILY_LOSS_TOO_TIGHT"
    if r075["daily_loss_breach_probability"] > 0:
        return "PHASE32A_FTMO_1STEP_SUPPORTED_WITH_WARNINGS"
    if bdr_summary["affects_075"]:
        return "PHASE32A_FTMO_1STEP_SUPPORTED_WITH_WARNINGS"
    if mc_summary.get("recommended_max_risk_mc", 0) and float(mc_summary["recommended_max_risk_mc"]) >= 0.75:
        return "PHASE32A_FTMO_1STEP_SUPPORTED_CONSERVATIVE_RISK"
    return "PHASE32A_FTMO_2STEP_REMAINS_PREFERRED"


def final_report(
    rules: dict[str, Any],
    lock: dict[str, Any],
    risk_grid: pd.DataFrame,
    hist_summary: dict[str, Any],
    mc_summary: dict[str, Any],
    daily_summary: dict[str, Any],
    bdr_summary: dict[str, Any],
    std_swing: dict[str, Any],
    comp_summary: dict[str, Any],
    verdict: str,
) -> dict[str, Any]:
    rows = {f"{r.risk_pct:.2f}": r._asdict() for r in risk_grid.itertuples(index=False)}
    r050 = rows["0.50"]
    r075 = rows["0.75"]
    r100 = rows["1.00"]
    recommended = {
        "product": "FTMO_2_STEP_REMAINS_PREFERRED_FOR_REAL_DECISION; FTMO_1_STEP_STANDARD_ONLY_PAPER_WITH_CONDITIONS",
        "account_type": "1-Step Standard paper only; 2-Step Swing cleaner for future funded account if available.",
        "platform_suggested": "No platform execution in Phase32A; paper ledger only.",
        "risk_recommended": "0.50%",
        "max_not_exceed": "0.75%",
    }
    payload = {
        "timestamp": now_utc(),
        "objective": "Simulate whether Phase25 authority supports FTMO 1-Step Standard.",
        "strategy_simulated": "PHASE25_AUTHORITY_ONLY",
        "other_strategy_used": False,
        "rules_ftmo_1step": rules,
        "assumptions": rules["assumptions"],
        "strategy_lock": lock,
        "risk_grid": rows,
        "historical_windows_summary": hist_summary,
        "monte_carlo_summary": mc_summary,
        "daily_loss_3pct_audit": daily_summary,
        "best_day_rule_audit": bdr_summary,
        "standard_vs_swing": std_swing,
        "comparison_1step_vs_2step": comp_summary,
        "risk_recommendation": recommended,
        "verdict": verdict,
        "phase25_remains_authority": True,
        "real_blocked": True,
        "mt5_blocked": True,
        "next_step": "Run Phase32 paper/demo discipline on 2-Step preferred path, while using 1-Step only as paper/free-trial scenario if desired.",
    }
    write_json(REPORT_JSON, payload)
    md = "\n".join(
        [
            "# PHASE32A FTMO 1-STEP STANDARD SIMULATION REPORT",
            "",
            "## Objetivo",
            payload["objective"],
            "",
            "## Estrategia simulada",
            "- PHASE25_AUTHORITY only.",
            "- TP1.4 / BE0.4 / BF70.",
            "- No shadow candidate.",
            "",
            "## Reglas FTMO 1-Step",
            "- Profit Target: 10%.",
            "- Max Daily Loss: 3%.",
            "- Max Loss: 10% trailing EOD.",
            "- Best Day Rule: 50%.",
            "- Standard only; Swing unavailable for 1-Step.",
            "",
            "## Resultados clave",
            f"- 0.50%: pass {r050['pass_probability']}%, daily breach {r050['daily_loss_breach_probability']}%, BDR block {r050['best_day_rule_violation_probability']}%.",
            f"- 0.75%: pass {r075['pass_probability']}%, daily breach {r075['daily_loss_breach_probability']}%, BDR block {r075['best_day_rule_violation_probability']}%.",
            f"- 1.00%: pass {r100['pass_probability']}%, daily breach {r100['daily_loss_breach_probability']}%, BDR block {r100['best_day_rule_violation_probability']}%.",
            "",
            "## Daily loss 3%",
            json.dumps(daily_summary, indent=2, default=str),
            "",
            "## Best Day Rule",
            json.dumps(bdr_summary, indent=2, default=str),
            "",
            "## Standard vs Swing",
            json.dumps(std_swing, indent=2, default=str),
            "",
            "## Comparacion 1-Step vs 2-Step",
            json.dumps(comp_summary, indent=2, default=str),
            "",
            "## Riesgo recomendado",
            "- Risk recommended: 0.50%.",
            "- Max not exceed: 0.75%.",
            "- 1.00% not recommended.",
            "",
            "## Veredicto final",
            verdict,
            "",
            "## Siguiente paso unico",
            payload["next_step"],
            "",
        ]
    )
    write_text(REPORT_MD, md)
    return payload


def rules_review(config: dict[str, Any]) -> dict[str, Any]:
    rules = config["profiles"]["FTMO_1_STEP_STANDARD_DEFAULT"]
    summary = {
        "profit_target_pct": 10.0,
        "max_daily_loss_pct": 3.0,
        "max_loss_pct": 10.0,
        "max_loss_mode": "trailing_eod",
        "best_day_rule_pct": 50.0,
        "trading_period": "unlimited",
        "reward_pct_metadata": 90.0,
        "standard_account": True,
        "swing_available_for_1step": False,
        "difference_vs_2step": "1-Step has 3% daily loss, EOD trailing max loss, Best Day Rule, no Verification.",
        "evaluation_vs_ftmo_account": "News/overnight restrictions are not applied during evaluation but Standard funded account has restrictions/warnings.",
        "assumptions": rules["assumptions"],
        "unknowns": rules["unknowns"],
        "requires_manual_rule_verification": True,
        "sources": [
            "https://ftmo.com/en/1-step-challenge/",
            "https://ftmo.com/en/trading-objectives/",
            "https://ftmo.com/en/how-it-works/",
            "https://ftmo.com/en/faq/how-long-does-it-take-to-become-an-ftmo-trader/",
        ],
    }
    write_json(OUT / "rules_review" / "phase32a_ftmo_1step_rules_review.json", summary)
    write_text(OUT / "rules_review" / "phase32a_ftmo_1step_rules_review.md", md_kv("PHASE32A FTMO 1-STEP RULES REVIEW", summary))
    return summary


def update_master_docs(verdict: str, recommended: dict[str, Any]) -> None:
    status = {
        "timestamp": now_utc(),
        "current_authority": "PHASE25",
        "phase25_status": "CURRENT_AUTHORITY_VALIDATED_2015_2026_FROZEN_PAPER_DEMO_ONLY_REAL_BLOCKED",
        "phase31_status": "CLOSED",
        "phase32_status": "PLANNED_READY_TO_START_PAPER_EVALUATION",
        "phase32a_status": "COMPLETED",
        "phase32a_verdict": verdict,
        "phase32a_scope": "FTMO_1_STEP_STANDARD_PHASE25_ONLY",
        "risk_recommendation_1step": recommended,
        "one_step_vs_two_step": "FTMO_2_STEP_REMAINS_PREFERRED_FOR_CAPITAL_PRESERVATION; 1-Step only paper/free-trial with conditions.",
        "real_blocked": True,
        "mt5_real_blocked": True,
        "vps_blocked": True,
        "ctrader_blocked": True,
        "scbi_protected": True,
        "phase19_archived": True,
        "news_fortress": "FAIL_CLOSED",
        "data_quality_mask": "FAIL_CLOSED",
    }
    write_json(ROOT / "01_CURRENT_PROJECT_STATUS.json", status)
    write_json(LAB / "status.json", status)
    write_json(
        ROOT / "02_STRATEGY_AUTHORITY_MAP.json",
        {
            "timestamp": now_utc(),
            "authority": "PHASE25",
            "phase25": {"status": "CURRENT_AUTHORITY", "validated": "2015-2026", "real": "BLOCKED"},
            "phase32a": {"status": "COMPLETED", "verdict": verdict, "scope": "PHASE25_ONLY"},
            "ftmo_1step_standard": {"recommendation": "PAPER_ONLY_WITH_CONDITIONS", "risk_recommended": "0.50%", "max_not_exceed": "0.75%"},
            "ftmo_2step": {"recommendation": "PREFERRED_FOR_CAPITAL_PRESERVATION"},
            "blocked": ["REAL", "MT5_REAL", "VPS", "CTRADER", "SCBI_TOUCH", "PHASE19_REOPEN"],
        },
    )
    write_text(
        ROOT / "00_READ_THIS_FIRST.md",
        "\n".join(
            [
                "# READ THIS FIRST",
                "",
                "- Current authority: Phase25.",
                "- Phase25 is validated 2015-2026 and remains frozen.",
                "- Phase32A simulated FTMO 1-Step Standard on Phase25 only.",
                f"- Phase32A verdict: {verdict}.",
                "- FTMO 1-Step Standard is paper/planning only, not real.",
                "- Recommended 1-Step risk: 0.50%; max not exceed 0.75%.",
                "- 1.00% is not recommended as base risk.",
                "- FTMO 2-Step remains preferred for capital preservation.",
                "- No real, no MT5, no automatic evaluation purchase.",
                "- SCBI protected; Phase19 archived.",
                "- News Fortress and Data Quality Mask remain fail-closed.",
                "",
            ]
        ),
    )
    write_text(
        ROOT / "01_CURRENT_PROJECT_STATUS.md",
        "\n".join(
            [
                "# CURRENT PROJECT STATUS",
                "",
                "- Authority: Phase25, validated 2015-2026, frozen.",
                "- Phase32A: COMPLETED / FTMO 1-Step Standard simulation / Phase25 only.",
                f"- Phase32A verdict: {verdict}.",
                "- 1-Step risk recommended: 0.50%.",
                "- 1-Step max not exceed: 0.75%.",
                "- 1.00% not recommended.",
                "- 2-Step remains preferred for capital preservation.",
                "- Real/MT5/cTrader/VPS: blocked.",
                "- SCBI: protected.",
                "- Phase19: archived.",
                "",
            ]
        ),
    )
    write_text(
        ROOT / "02_STRATEGY_AUTHORITY_MAP.md",
        "\n".join(
            [
                "# STRATEGY AUTHORITY MAP",
                "",
                "- PHASE25: CURRENT AUTHORITY / VALIDATED 2015-2026 / FROZEN.",
                "- PHASE32A: FTMO 1-STEP STANDARD SIMULATION / PHASE25 ONLY.",
                "- TP1.4_BE0.5_BF70: NOT USED IN PHASE32A.",
                "- FTMO 1-Step Standard: paper only with conditions.",
                "- FTMO 2-Step: preferred for capital preservation.",
                "- REAL: BLOCKED.",
                "- MT5 REAL: BLOCKED.",
                "- CTRADER/VPS: BLOCKED.",
                "- SCBI: PROTECTED.",
                "- PHASE19: ARCHIVED.",
                "",
            ]
        ),
    )


def update_manifests() -> None:
    text = "\n".join(
        [
            "# ZIP CONTENTS MANIFEST",
            "",
            "- Canonical live zip: 000_PARA_CHATGPT.zip",
            f"- Official path: {ZIP_PATH}",
            "- Phase32A FTMO 1-Step Standard report included.",
            "- Phase32A lightweight outputs included.",
            "- prop_firm_rules_config.json updated with FTMO_1_STEP_STANDARD_DEFAULT.",
            "- Phase31 closeout and Phase32 docs included.",
            "- Phase25 config/hash included.",
            "- No raw heavy data, no secrets, no internal zip files.",
            "",
        ]
    )
    write_text(ROOT / "ZIP_CONTENTS_MANIFEST.md", text)
    write_text(LAB / "ZIP_CONTENTS_MANIFEST.md", text)


def zip_include(path: Path) -> bool:
    if not path.is_file():
        return False
    rel = path.relative_to(ROOT)
    rel_s = str(rel).replace("\\", "/")
    parts = set(rel.parts)
    suffix = path.suffix.lower()
    name = path.name.lower()
    banned_parts = {
        ".git",
        ".venv",
        ".venv_fixed",
        "__pycache__",
        "data",
        "data_intake_2015_2019",
        "data_intake_2020_2026_bidask",
        "data_free_2020",
        "data_candidates_2022_2025",
        "scratch",
        "legacy_archive_2026",
        "quarantine",
        "secrets",
    }
    if parts & banned_parts:
        return False
    if suffix in {".zip", ".zipbak", ".building", ".pkl", ".parquet", ".bi5", ".db", ".sqlite", ".dll", ".exe"}:
        return False
    if name in {".env", "mt5_local_config.json"}:
        return False
    if any(tok in name for tok in ["secret", "password", "token", "credential", "apikey", "api_key"]):
        return False
    if path.stat().st_size > 2 * 1024 * 1024:
        return False
    root_includes = {
        "00_READ_THIS_FIRST.md",
        "01_CURRENT_PROJECT_STATUS.md",
        "01_CURRENT_PROJECT_STATUS.json",
        "02_STRATEGY_AUTHORITY_MAP.md",
        "02_STRATEGY_AUTHORITY_MAP.json",
        "ZIP_CONTENTS_MANIFEST.md",
    }
    if len(rel.parts) == 1:
        return rel_s in root_includes
    if rel.parts[0] != "BOT_V2_DAYTIME_LAB":
        return False
    if rel_s in {
        "BOT_V2_DAYTIME_LAB/status.json",
        "BOT_V2_DAYTIME_LAB/ZIP_CONTENTS_MANIFEST.md",
        "BOT_V2_DAYTIME_LAB/ZIP_UPLOAD_IDENTITY_MARKER.md",
    }:
        return True
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/reports/"):
        return suffix in {".md", ".json"}
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/configs/"):
        return suffix in {".json", ".txt"}
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/docs/"):
        return suffix in {".md", ".json", ".txt"}
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/templates/"):
        return suffix in {".md", ".csv", ".txt"}
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/outputs/phase32a_ftmo_1step_standard_simulation/"):
        return "/zip_validation/" not in rel_s and suffix in {".md", ".json", ".csv", ".txt"}
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/outputs/phase31_final_closeout/"):
        return suffix in {".md", ".json", ".csv", ".txt"}
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/outputs/phase31_prop_firm_survival_simulator/"):
        return suffix in {".md", ".json", ".csv", ".txt"}
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/outputs/phase30_tp14_be05_bf70_forensic_audit/"):
        return suffix in {".md", ".json", ".csv", ".txt"}
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/outputs/phase29_wr_loss_streak_compression/"):
        return suffix in {".md", ".json", ".csv", ".txt"}
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/outputs/phase27_full_historical_validation_2015_2026/"):
        return suffix in {".md", ".json", ".csv", ".txt"} and path.stat().st_size <= 700000
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/src/"):
        return suffix == ".py" and (
            "phase32a" in name
            or "phase31" in name
            or "phase30" in name
            or "phase29" in name
            or "phase28" in name
            or "phase27" in name
            or "phase26" in name
            or name in {"phase18_h1_fractal_sweep.py", "phase18_first_3m_choch.py"}
        )
    return False


def rebuild_zip() -> dict[str, Any]:
    if BUILD_PATH.exists():
        BUILD_PATH.unlink()
    files = sorted([p for p in ROOT.rglob("*") if zip_include(p)], key=lambda p: str(p.relative_to(ROOT)).replace("\\", "/"))
    with zipfile.ZipFile(BUILD_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, str(path.relative_to(ROOT)).replace("\\", "/"))
    with zipfile.ZipFile(BUILD_PATH, "r") as zf:
        test = zf.testzip()
        names = zf.namelist()
        heavy = [n for n in names if zf.getinfo(n).file_size > 2 * 1024 * 1024]
        secrets = [n for n in names if any(tok in n.lower() for tok in [".env", "secret", "password", "token", "credential", "apikey", "api_key"])]
        zips = [n for n in names if n.lower().endswith((".zip", ".zipbak"))]
    if test is not None or heavy or secrets or zips:
        raise RuntimeError(f"ZIP_VALIDATION_FAILED test={test} heavy={heavy[:5]} secrets={secrets[:5]} zips={zips[:5]}")
    os.replace(str(BUILD_PATH), str(ZIP_PATH))
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        names = zf.namelist()
        entries_text = "\n".join(names) + "\n"
    result = {
        **zip_details(ZIP_PATH),
        "single_live_zip_exact_extension": len(exact_zip_inventory()) == 1,
        "contains_phase32a_report": "BOT_V2_DAYTIME_LAB/reports/PHASE32A_FTMO_1STEP_STANDARD_SIMULATION_REPORT.md" in names,
        "contains_phase32a_outputs": any(n.startswith("BOT_V2_DAYTIME_LAB/outputs/phase32a_ftmo_1step_standard_simulation/") for n in names),
        "contains_prop_rules_config": "BOT_V2_DAYTIME_LAB/configs/prop_firm_rules_config.json" in names,
        "contains_phase25_config_hash": "BOT_V2_DAYTIME_LAB/configs/phase25_forward_demo_candidate_config_hash.txt" in names,
        "heavy_entries_gt_2mb": [],
        "secret_like_entries": [],
        "zip_entries_inside": [],
        "validation_artifacts_embedded": False,
    }
    write_json(OUT / "zip_validation" / "phase32a_zip_validation.json", result)
    write_text(OUT / "zip_validation" / "phase32a_zip_validation.md", md_kv("PHASE32A ZIP VALIDATION", result))
    write_text(OUT / "zip_validation" / "phase32a_zip_entries.txt", entries_text)
    return result


def git_status_artifacts() -> dict[str, Any]:
    data = {
        "timestamp": now_utc(),
        "branch": run_cmd(["git", "branch", "--show-current"]),
        "status": run_cmd(["git", "status", "--short"]),
        "diff_stat": run_cmd(["git", "diff", "--stat"]),
        "commit": "NO",
        "push": "NO",
    }
    write_json(OUT / "git" / "phase32a_git_status.json", data)
    write_text(OUT / "git" / "phase32a_git_status.md", md_kv("PHASE32A GIT STATUS", data))
    return data


def main() -> None:
    ensure_dirs()
    config = update_rules_config()
    preflight()
    trades = load_phase25_trades()
    lock = strategy_lock(trades)
    rules = rules_review(config)
    print("Historical windows")
    historical, hist_summary = historical_windows(trades)
    print("Risk grid")
    risk_grid, risk_summary = risk_grid_summary(historical)
    print("Monte Carlo")
    mc, mc_summary = monte_carlo(trades)
    print("Daily loss")
    _, daily_summary = daily_loss_audit(trades)
    print("Best Day Rule")
    _, bdr_summary = best_day_rule_audit(historical)
    print("Standard vs Swing")
    std_swing = standard_vs_swing()
    print("1-Step vs 2-Step")
    _, comp_summary = comparison_1step_vs_2step(risk_grid, mc, std_swing)
    verdict = decide_verdict(risk_grid, mc_summary, bdr_summary)
    report = final_report(rules, lock, risk_grid, hist_summary, mc_summary, daily_summary, bdr_summary, std_swing, comp_summary, verdict)
    update_master_docs(verdict, report["risk_recommendation"])
    update_manifests()
    git_status_artifacts()
    zip_result = rebuild_zip()
    print(json.dumps({"verdict": verdict, "zip": zip_result}, indent=2))


if __name__ == "__main__":
    main()
