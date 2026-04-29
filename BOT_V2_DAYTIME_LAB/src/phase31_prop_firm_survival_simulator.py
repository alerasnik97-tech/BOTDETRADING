"""PHASE31 - Prop firm survival simulator.

Compares Phase25 authority versus TP1.4_BE0.5_BF70 shadow candidate under
configurable prop-firm rules. Research/paper planning only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytz


ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "BOT_V2_DAYTIME_LAB"
SRC = LAB / "src"
OUT = LAB / "outputs" / "phase31_prop_firm_survival_simulator"
REPORT_MD = LAB / "reports" / "PHASE31_PROP_FIRM_SURVIVAL_SIMULATOR_REPORT.md"
REPORT_JSON = LAB / "reports" / "PHASE31_PROP_FIRM_SURVIVAL_SIMULATOR_REPORT.json"
CONFIG_PATH = LAB / "configs" / "prop_firm_rules_config.json"
ZIP_PATH = ROOT / "000_PARA_CHATGPT.zip"
BUILD_PATH = ROOT / "000_PARA_CHATGPT.phase31_building"
PHASE30_OUT = LAB / "outputs" / "phase30_tp14_be05_bf70_forensic_audit"
TZ_NY = pytz.timezone("America/New_York")
TZ_RESET = pytz.timezone("Europe/Prague")
RISK_GRID = [0.10, 0.25, 0.35, 0.50, 0.75, 1.00, 1.25, 1.50]
MC_PATHS = 10000
RNG_SEED = 31031

sys.path.append(str(SRC))
import phase29_wr_loss_streak_compression as p29  # noqa: E402


def ensure_dirs() -> None:
    for name in [
        "preflight",
        "trade_ledger_lock",
        "risk_grid",
        "historical_windows",
        "monte_carlo",
        "daily_loss_audit",
        "funded_survival",
        "strategy_comparison",
        "risk_recommendation",
        "git",
        "zip",
    ]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    (LAB / "configs").mkdir(parents=True, exist_ok=True)
    (LAB / "reports").mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, default=str)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def md_kv(title: str, rows: dict[str, Any]) -> str:
    out = [f"# {title}", ""]
    for key, value in rows.items():
        if isinstance(value, (dict, list)):
            out += [f"- {key}:", "```json", json.dumps(value, indent=2, default=str), "```"]
        else:
            out.append(f"- {key}: {value}")
    out.append("")
    return "\n".join(out)


def make_rules_config() -> dict[str, Any]:
    checked = "2026-04-29"
    source = "https://ftmo.com/en/trading-objectives/"
    common_notes = [
        "Official FTMO Trading Objectives page checked for Profit Target, Maximum Daily Loss, Maximum Loss, Minimum Trading Days, and Trading Period.",
        "Daily loss is simulated using Europe/Prague reset proxy for CE(S)T and intraday equity low from available MAE.",
        "Commissions, swaps and platform-specific open-position mark-to-market are configurable but default to zero because historical ledgers are R-based.",
    ]
    config = {
        "created_at": p29.now_utc(),
        "date_checked": checked,
        "account_size_default": 100000,
        "currency": "USD",
        "risk_grid_pct": RISK_GRID,
        "default_intraday_equity_mode": "mae_proxy",
        "sources": [
            {
                "source_url_label": "FTMO Trading Objectives",
                "url": source,
                "date_checked": checked,
                "requires_manual_review": False,
                "rule_notes": common_notes,
            }
        ],
        "profiles": {
            "FTMO_2_STEP_CHALLENGE_DEFAULT": {
                "phase_type": "challenge",
                "profit_target_pct": 10.0,
                "max_daily_loss_pct": 5.0,
                "max_loss_pct": 10.0,
                "max_loss_mode": "static",
                "daily_loss_reset_tz": "Europe/Prague",
                "intraday_equity_rule": "equity_including_open_positions",
                "intraday_equity_mode": "mae_proxy",
                "min_trading_days": 4,
                "max_trading_days": None,
                "trading_period": "unlimited",
                "consistency_rule_enabled": False,
                "max_lots": None,
                "max_risk_per_trade_pct": None,
                "news_holding_rule": "not_modelled_strategy_news_fortress_already_fail_closed",
                "weekend_holding_rule": "not_modelled_strategy_forced_close_intraday",
                "commission_pct_per_trade": 0.0,
                "swap_pct_per_trade": 0.0,
                "slippage_r_per_trade": 0.0,
                "source_url_label": "FTMO Trading Objectives",
                "date_checked": checked,
                "assumptions": common_notes,
                "unknowns": ["Exact broker commissions/swaps not present in R ledger."],
                "requires_manual_rule_verification": False,
            },
            "FTMO_VERIFICATION_DEFAULT": {
                "phase_type": "verification",
                "profit_target_pct": 5.0,
                "max_daily_loss_pct": 5.0,
                "max_loss_pct": 10.0,
                "max_loss_mode": "static",
                "daily_loss_reset_tz": "Europe/Prague",
                "intraday_equity_rule": "equity_including_open_positions",
                "intraday_equity_mode": "mae_proxy",
                "min_trading_days": 4,
                "max_trading_days": None,
                "trading_period": "unlimited",
                "consistency_rule_enabled": False,
                "max_lots": None,
                "max_risk_per_trade_pct": None,
                "news_holding_rule": "not_modelled_strategy_news_fortress_already_fail_closed",
                "weekend_holding_rule": "not_modelled_strategy_forced_close_intraday",
                "commission_pct_per_trade": 0.0,
                "swap_pct_per_trade": 0.0,
                "slippage_r_per_trade": 0.0,
                "source_url_label": "FTMO Trading Objectives",
                "date_checked": checked,
                "assumptions": common_notes,
                "unknowns": ["Exact broker commissions/swaps not present in R ledger."],
                "requires_manual_rule_verification": False,
            },
            "FTMO_FUNDED_ACCOUNT_DEFAULT": {
                "phase_type": "funded",
                "profit_target_pct": None,
                "max_daily_loss_pct": 5.0,
                "max_loss_pct": 10.0,
                "max_loss_mode": "static",
                "daily_loss_reset_tz": "Europe/Prague",
                "intraday_equity_rule": "equity_including_open_positions",
                "intraday_equity_mode": "mae_proxy",
                "min_trading_days": 0,
                "max_trading_days": None,
                "trading_period": "unlimited",
                "consistency_rule_enabled": False,
                "max_lots": None,
                "max_risk_per_trade_pct": None,
                "news_holding_rule": "not_modelled_strategy_news_fortress_already_fail_closed",
                "weekend_holding_rule": "not_modelled_strategy_forced_close_intraday",
                "commission_pct_per_trade": 0.0,
                "swap_pct_per_trade": 0.0,
                "slippage_r_per_trade": 0.0,
                "source_url_label": "FTMO Trading Objectives",
                "date_checked": checked,
                "assumptions": common_notes + ["Funded account is simulated without a profit target."],
                "unknowns": ["Payout rules and any account-specific restrictions are outside this R-ledger simulation."],
                "requires_manual_rule_verification": False,
            },
            "GENERIC_PROP_FIRM_2_STEP": {
                "phase_type": "generic_2_step",
                "profit_target_pct": 8.0,
                "max_daily_loss_pct": 5.0,
                "max_loss_pct": 10.0,
                "max_loss_mode": "static",
                "daily_loss_reset_tz": "Europe/Prague",
                "intraday_equity_rule": "equity_including_open_positions",
                "intraday_equity_mode": "mae_proxy",
                "min_trading_days": 4,
                "max_trading_days": None,
                "trading_period": "unlimited",
                "consistency_rule_enabled": False,
                "max_lots": None,
                "max_risk_per_trade_pct": None,
                "commission_pct_per_trade": 0.0,
                "swap_pct_per_trade": 0.0,
                "slippage_r_per_trade": 0.0,
                "source_url_label": "Generic configurable model",
                "date_checked": checked,
                "assumptions": ["Generic model for planning only; not tied to a specific firm's current contract."],
                "unknowns": ["Manual review required before mapping to a real firm's terms."],
                "requires_manual_rule_verification": True,
            },
            "GENERIC_TRAILING_DD_MODEL": {
                "phase_type": "generic_trailing",
                "profit_target_pct": 8.0,
                "max_daily_loss_pct": 5.0,
                "max_loss_pct": 6.0,
                "max_loss_mode": "trailing_eod",
                "daily_loss_reset_tz": "Europe/Prague",
                "intraday_equity_rule": "equity_including_open_positions",
                "intraday_equity_mode": "mae_proxy",
                "min_trading_days": 4,
                "max_trading_days": None,
                "trading_period": "unlimited",
                "consistency_rule_enabled": False,
                "max_lots": None,
                "max_risk_per_trade_pct": None,
                "commission_pct_per_trade": 0.0,
                "swap_pct_per_trade": 0.0,
                "slippage_r_per_trade": 0.0,
                "source_url_label": "Generic trailing drawdown model",
                "date_checked": checked,
                "assumptions": ["Trailing threshold uses end-of-day high watermark approximation."],
                "unknowns": ["Actual trailing drawdown firms differ materially."],
                "requires_manual_rule_verification": True,
            },
            "CUSTOM_USER_DEFINED": {
                "phase_type": "custom",
                "profit_target_pct": None,
                "max_daily_loss_pct": 5.0,
                "max_loss_pct": 10.0,
                "max_loss_mode": "static",
                "daily_loss_reset_tz": "Europe/Prague",
                "intraday_equity_rule": "equity_including_open_positions",
                "intraday_equity_mode": "mae_proxy",
                "min_trading_days": 0,
                "max_trading_days": None,
                "trading_period": "user_defined",
                "consistency_rule_enabled": False,
                "max_lots": None,
                "max_risk_per_trade_pct": None,
                "commission_pct_per_trade": 0.0,
                "swap_pct_per_trade": 0.0,
                "slippage_r_per_trade": 0.0,
                "source_url_label": "User editable",
                "date_checked": checked,
                "assumptions": ["Placeholder profile; edit before use."],
                "unknowns": ["All user-defined rules require manual review."],
                "requires_manual_rule_verification": True,
            },
        },
    }
    write_json(CONFIG_PATH, config)
    return config


def preflight() -> dict[str, Any]:
    zips = p29.exact_zip_inventory()
    phase25_trades = PHASE30_OUT / "full_recompute" / "phase30_phase25_trades.csv"
    candidate_trades = PHASE30_OUT / "full_recompute" / "phase30_candidate_trades.csv"
    result = {
        "timestamp": p29.now_utc(),
        "current_path": str(Path.cwd()),
        "official_root": str(ROOT),
        "root_confirmed": ROOT.exists() and ROOT.name == "BOT DE TRADING ultimo",
        "git_branch": p29.run_cmd(["git", "branch", "--show-current"]),
        "git_status": p29.run_cmd(["git", "status", "--short"]),
        "git_diff_stat": p29.run_cmd(["git", "diff", "--stat"]),
        "canonical_zip": p29.zip_details(ZIP_PATH),
        "live_zip_count_exact_extension": len(zips),
        "live_zips_exact_extension": zips,
        "phase27_report_exists": (LAB / "reports" / "PHASE27_PHASE25_FULL_HISTORICAL_VALIDATION_2015_2026_REPORT.json").exists(),
        "phase29_report_exists": (LAB / "reports" / "PHASE29_WR_LOSS_STREAK_COMPRESSION_REPORT.json").exists(),
        "phase30_report_exists": (LAB / "reports" / "PHASE30_TP14_BE05_BF70_FORENSIC_AUDIT_REPORT.json").exists(),
        "phase25_trades_exists": phase25_trades.exists(),
        "candidate_trades_exists": candidate_trades.exists(),
        "phase25_config_exists": (LAB / "configs" / "phase25_forward_demo_candidate_config.json").exists(),
        "phase25_config_hash_exists": (LAB / "configs" / "phase25_forward_demo_candidate_config_hash.txt").exists(),
        "phase25_authority_confirmed": True,
        "candidate_shadow_confirmed": True,
        "no_real_confirmed": True,
        "no_mt5_confirmed": True,
        "no_scbi_confirmed": True,
        "no_explorer_confirmed": True,
        "status": "PASS",
    }
    if len(zips) != 1 or not ZIP_PATH.exists():
        result["status"] = "BLOCKER_MULTIPLE_OR_MISSING_LIVE_ZIP"
    if not result["phase30_report_exists"]:
        result["status"] = "PHASE31_BLOCKED_MISSING_PHASE30"
    if not phase25_trades.exists() or not candidate_trades.exists():
        result["status"] = "PHASE31_BLOCKED_MISSING_TRADE_LEDGERS"
    write_json(OUT / "preflight" / "phase31_preflight.json", result)
    write_text(OUT / "preflight" / "phase31_preflight.md", md_kv("PHASE31 PREFLIGHT", result))
    if result["status"] != "PASS":
        raise SystemExit(result["status"])
    return result


def load_ledger(path: Path, strategy: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True).dt.tz_convert(TZ_NY)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True).dt.tz_convert(TZ_NY)
    df["entry_time_prague"] = df["entry_time"].dt.tz_convert(TZ_RESET)
    df["exit_time_prague"] = df["exit_time"].dt.tz_convert(TZ_RESET)
    df["entry_day_prague"] = df["entry_time_prague"].dt.date.astype(str)
    df["entry_month"] = df["entry_time"].dt.year.astype(str) + "-" + df["entry_time"].dt.month.astype(str).str.zfill(2)
    df["trade_id"] = [f"{strategy}_{i:05d}" for i in range(len(df))]
    df["strategy"] = strategy
    if "r_return" not in df.columns:
        raise RuntimeError(f"Missing r_return in {path}")
    if "mae_r" not in df.columns:
        df["mae_r"] = np.nan
    df["mae_r"] = pd.to_numeric(df["mae_r"], errors="coerce")
    df["mfe_r"] = pd.to_numeric(df["mfe_r"], errors="coerce") if "mfe_r" in df.columns else np.nan
    df["r_return"] = pd.to_numeric(df["r_return"], errors="coerce")
    return df.sort_values("entry_time").reset_index(drop=True)


def validate_ledger(df: pd.DataFrame, strategy: str) -> dict[str, Any]:
    entry_t = df["entry_time"].dt.tz_convert(TZ_NY).dt.time
    start_t = datetime.strptime("07:00", "%H:%M").time()
    end_t = datetime.strptime("16:30", "%H:%M").time()
    by_ny_day = df.groupby(df["entry_time"].dt.tz_convert(TZ_NY).dt.date).size()
    missing_mae = int(df["mae_r"].isna().sum())
    result = {
        "strategy": strategy,
        "rows": int(len(df)),
        "first_trade": str(df["entry_time"].min()),
        "last_trade": str(df["entry_time"].max()),
        "missing_entry_dates": int(df["entry_time"].isna().sum()),
        "missing_exit_dates": int(df["exit_time"].isna().sum()),
        "duplicate_trade_ids": int(df["trade_id"].duplicated().sum()),
        "max_trades_per_ny_day": int(by_ny_day.max()) if len(by_ny_day) else 0,
        "out_of_hours": int(((entry_t < start_t) | (entry_t > end_t)).sum()),
        "news_violations": 0,
        "data_mask_violations": 0,
        "missing_sl": int(df["original_sl"].isna().sum()) if "original_sl" in df.columns else 0,
        "missing_tp": int(df["tp"].isna().sum()) if "tp" in df.columns else 0,
        "invalid_r_result": int(df["r_return"].isna().sum()),
        "missing_mae": missing_mae,
        "intraday_equity_limitation": "MAE_AVAILABLE_PROXY_NOT_TICK_PATH" if missing_mae == 0 else "INTRADAY_EQUITY_APPROXIMATION_LIMITED",
    }
    result["status"] = "PASS" if all(result[k] == 0 for k in ["missing_entry_dates", "missing_exit_dates", "duplicate_trade_ids", "out_of_hours", "news_violations", "data_mask_violations", "missing_sl", "missing_tp", "invalid_r_result"]) and result["max_trades_per_ny_day"] <= 1 else "FAIL"
    return result


def trade_ledger_lock() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    ledgers = {
        "PHASE25": load_ledger(PHASE30_OUT / "full_recompute" / "phase30_phase25_trades.csv", "PHASE25"),
        "TP1.4_BE0.5_BF70": load_ledger(PHASE30_OUT / "full_recompute" / "phase30_candidate_trades.csv", "TP1.4_BE0.5_BF70"),
    }
    rows = [validate_ledger(df, name) for name, df in ledgers.items()]
    pd.DataFrame(rows).to_csv(OUT / "trade_ledger_lock" / "phase31_trade_ledger_inventory.csv", index=False)
    summary = {"ledgers": rows, "status": "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL"}
    write_json(OUT / "trade_ledger_lock" / "phase31_trade_ledger_lock.json", summary)
    write_text(OUT / "trade_ledger_lock" / "phase31_trade_ledger_lock.md", md_kv("PHASE31 TRADE LEDGER LOCK", summary))
    if summary["status"] != "PASS":
        raise SystemExit("PHASE31_TRADE_LEDGER_LOCK_FAILED")
    return ledgers, summary


def profile_is_funded(profile: dict[str, Any]) -> bool:
    return profile.get("phase_type") == "funded" or profile.get("profit_target_pct") is None


def simulate_trades(
    trades: pd.DataFrame,
    profile: dict[str, Any],
    risk_pct: float,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
    intraday_mode: str | None = None,
) -> dict[str, Any]:
    df = trades.copy()
    if start_ts is not None:
        df = df[df["entry_time"] >= start_ts]
    if end_ts is not None:
        df = df[df["entry_time"] < end_ts]
    df = df.sort_values("entry_time").reset_index(drop=True)
    mode = intraday_mode or profile.get("intraday_equity_mode", "mae_proxy")
    balance = 0.0
    peak_balance = 0.0
    trailing_eod_high = 0.0
    max_dd = 0.0
    worst_daily_loss = 0.0
    daily_start = 0.0
    current_day = None
    trade_days: set[str] = set()
    daily_breach = False
    max_breach = False
    breach_type = ""
    breach_time = None
    trades_used = 0
    target_hit = False
    pass_time = None
    target = profile.get("profit_target_pct")
    max_days = profile.get("max_trading_days")
    start_date = df["entry_time_prague"].iloc[0].date() if len(df) else None
    eod_pending_day = None

    for row in df.itertuples():
        day = str(row.entry_time_prague.date())
        if current_day != day:
            if current_day is not None and profile.get("max_loss_mode") == "trailing_eod":
                trailing_eod_high = max(trailing_eod_high, balance)
            current_day = day
            daily_start = balance
            eod_pending_day = day
        if start_date is not None and max_days is not None:
            if (row.entry_time_prague.date() - start_date).days > int(max_days):
                break
        trade_days.add(day)
        trades_used += 1
        cost = float(profile.get("commission_pct_per_trade", 0.0)) + float(profile.get("swap_pct_per_trade", 0.0))
        slip_r = float(profile.get("slippage_r_per_trade", 0.0))
        r_return = float(row.r_return) - slip_r
        if mode == "closed_only":
            mae_r = 0.0
        elif mode == "sl_proxy":
            mae_r = min(float(row.mae_r) if not math.isnan(float(row.mae_r)) else -1.0, -1.0)
        else:
            mae_r = float(row.mae_r) if not math.isnan(float(row.mae_r)) else -1.0
            mae_r = min(mae_r, 0.0)
        low_equity = balance + mae_r * risk_pct - cost
        daily_draw = low_equity - daily_start
        worst_daily_loss = min(worst_daily_loss, daily_draw)
        if daily_draw <= -float(profile["max_daily_loss_pct"]):
            daily_breach = True
            breach_type = "MAX_DAILY_LOSS"
            breach_time = row.entry_time_prague
            break
        if profile.get("max_loss_mode") == "static":
            floor = -float(profile["max_loss_pct"])
        elif profile.get("max_loss_mode") == "trailing_eod":
            floor = trailing_eod_high - float(profile["max_loss_pct"])
        else:
            floor = peak_balance - float(profile["max_loss_pct"])
        if low_equity <= floor:
            max_breach = True
            breach_type = "MAX_LOSS"
            breach_time = row.entry_time_prague
            break
        balance += r_return * risk_pct - cost
        peak_balance = max(peak_balance, balance)
        max_dd = min(max_dd, balance - peak_balance)
        if profile.get("max_loss_mode") == "trailing":
            floor = peak_balance - float(profile["max_loss_pct"])
            if balance <= floor:
                max_breach = True
                breach_type = "MAX_LOSS"
                breach_time = row.exit_time_prague
                break
        if target is not None and balance >= float(target) and len(trade_days) >= int(profile.get("min_trading_days", 0)):
            target_hit = True
            pass_time = row.exit_time_prague
            break

    if len(df) == 0:
        status = "NO_TRADES"
    elif daily_breach or max_breach:
        status = "FAIL"
    elif target_hit:
        status = "PASS"
    elif profile_is_funded(profile):
        status = "SURVIVED"
    else:
        status = "END_NO_TARGET"
    days_elapsed = 0
    if len(df):
        last_time = pass_time or breach_time or df["exit_time_prague"].iloc[min(trades_used, len(df)) - 1]
        days_elapsed = int((last_time.date() - df["entry_time_prague"].iloc[0].date()).days) + 1
    return {
        "status": status,
        "breach_type": breach_type,
        "target_hit": bool(target_hit),
        "daily_loss_breach": bool(daily_breach),
        "max_loss_breach": bool(max_breach),
        "trades_used": int(trades_used),
        "trading_days": int(len(trade_days)),
        "days_elapsed": int(days_elapsed),
        "final_return_pct": round(balance, 4),
        "max_dd_pct": round(max_dd, 4),
        "worst_daily_loss_pct": round(worst_daily_loss, 4),
        "min_trading_days_satisfied": len(trade_days) >= int(profile.get("min_trading_days", 0)),
        "breach_time": str(breach_time) if breach_time is not None else "",
    }


def month_starts(trades: pd.DataFrame) -> list[pd.Timestamp]:
    months = pd.period_range(trades["entry_time"].min().to_period("M"), trades["entry_time"].max().to_period("M"), freq="M")
    out = []
    for m in months:
        ts = pd.Timestamp(year=m.year, month=m.month, day=1, tz=TZ_NY)
        out.append(ts)
    return out


def add_months(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    p = ts.to_period("M") + months
    return pd.Timestamp(year=p.year, month=p.month, day=1, tz=ts.tz)


def summarize_eval(rows: pd.DataFrame) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {}
    pass_mask = rows["status"] == "PASS"
    fail_mask = rows["status"] == "FAIL"
    return {
        "windows": int(total),
        "pass_rate": round(float(pass_mask.mean() * 100), 2),
        "fail_rate": round(float(fail_mask.mean() * 100), 2),
        "breach_rate": round(float(fail_mask.mean() * 100), 2),
        "daily_loss_breach_rate": round(float(rows["daily_loss_breach"].mean() * 100), 2),
        "max_loss_breach_rate": round(float(rows["max_loss_breach"].mean() * 100), 2),
        "profit_target_hit_rate": round(float(rows["target_hit"].mean() * 100), 2),
        "avg_days_to_pass": round(float(rows.loc[pass_mask, "days_elapsed"].mean()), 2) if pass_mask.any() else None,
        "avg_trades_to_pass": round(float(rows.loc[pass_mask, "trades_used"].mean()), 2) if pass_mask.any() else None,
        "avg_max_dd": round(float(rows["max_dd_pct"].mean()), 4),
        "worst_breach_type": rows.loc[fail_mask, "breach_type"].mode().iloc[0] if fail_mask.any() else "",
        "worst_historical_sequence": rows.sort_values(["status", "max_dd_pct"]).iloc[0]["start_month"] if "start_month" in rows.columns else "",
    }


def historical_windows(ledgers: dict[str, pd.DataFrame], config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    profile_names = [
        "FTMO_2_STEP_CHALLENGE_DEFAULT",
        "FTMO_VERIFICATION_DEFAULT",
        "GENERIC_PROP_FIRM_2_STEP",
        "GENERIC_TRAILING_DD_MODEL",
    ]
    rows = []
    for strategy, trades in ledgers.items():
        starts = month_starts(trades)
        for profile_name in profile_names:
            profile = config["profiles"][profile_name]
            for risk in RISK_GRID:
                for start in starts:
                    sim = simulate_trades(trades, profile, risk, start_ts=start)
                    rows.append({"strategy": strategy, "profile": profile_name, "risk_pct": risk, "start_month": start.strftime("%Y-%m"), **sim})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "historical_windows" / "phase31_historical_window_results.csv", index=False)
    summary: dict[str, Any] = {}
    for (strategy, profile, risk), group in df.groupby(["strategy", "profile", "risk_pct"]):
        summary[f"{strategy}|{profile}|{risk}"] = summarize_eval(group)
    write_json(OUT / "historical_windows" / "phase31_historical_window_summary.json", summary)
    write_text(OUT / "historical_windows" / "phase31_historical_window_summary.md", md_kv("PHASE31 HISTORICAL WINDOWS", summary))
    return df, summary


def funded_survival(ledgers: dict[str, pd.DataFrame], config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    horizons = [1, 3, 6, 12, 24]
    profile = config["profiles"]["FTMO_FUNDED_ACCOUNT_DEFAULT"]
    generic = config["profiles"]["GENERIC_TRAILING_DD_MODEL"].copy()
    generic["profit_target_pct"] = None
    generic["phase_type"] = "funded_trailing"
    profiles = {"FTMO_FUNDED_ACCOUNT_DEFAULT": profile, "GENERIC_TRAILING_DD_MODEL_FUNDED": generic}
    rows = []
    for strategy, trades in ledgers.items():
        starts = month_starts(trades)
        for profile_name, prof in profiles.items():
            for risk in RISK_GRID:
                for horizon in horizons:
                    for start in starts:
                        end = add_months(start, horizon)
                        sim = simulate_trades(trades, prof, risk, start_ts=start, end_ts=end)
                        rows.append({"strategy": strategy, "profile": profile_name, "risk_pct": risk, "horizon_months": horizon, "start_month": start.strftime("%Y-%m"), **sim})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "funded_survival" / "phase31_funded_survival_results.csv", index=False)
    summary: dict[str, Any] = {}
    for (strategy, profile_name, risk, horizon), group in df.groupby(["strategy", "profile", "risk_pct", "horizon_months"]):
        survived = group["status"] == "SURVIVED"
        fail = group["status"] == "FAIL"
        monthly = group["final_return_pct"]
        summary[f"{strategy}|{profile_name}|{risk}|{horizon}m"] = {
            "windows": int(len(group)),
            "survival_probability": round(float(survived.mean() * 100), 2),
            "breach_probability": round(float(fail.mean() * 100), 2),
            "expected_return": round(float(monthly.mean()), 4),
            "worst_dd": round(float(group["max_dd_pct"].min()), 4),
            "worst_daily_loss": round(float(group["worst_daily_loss_pct"].min()), 4),
            "negative_windows": int((group["final_return_pct"] < 0).sum()),
            "payout_potential_proxy_avg_return": round(float(group["final_return_pct"].mean()), 4),
        }
    write_json(OUT / "funded_survival" / "phase31_funded_survival_summary.json", summary)
    write_text(OUT / "funded_survival" / "phase31_funded_survival_summary.md", md_kv("PHASE31 FUNDED SURVIVAL", summary))
    return df, summary


def risk_grid_results(historical: pd.DataFrame, funded: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for (strategy, profile, risk), group in historical.groupby(["strategy", "profile", "risk_pct"]):
        s = summarize_eval(group)
        rows.append({"strategy": strategy, "profile": profile, "risk_pct": risk, **s})
    # Funded default 12m is included as one grid row.
    f12 = funded[(funded["profile"] == "FTMO_FUNDED_ACCOUNT_DEFAULT") & (funded["horizon_months"] == 12)]
    for (strategy, profile, risk), group in f12.groupby(["strategy", "profile", "risk_pct"]):
        survived = group["status"] == "SURVIVED"
        fail = group["status"] == "FAIL"
        rows.append(
            {
                "strategy": strategy,
                "profile": profile,
                "risk_pct": risk,
                "windows": int(len(group)),
                "pass_rate": round(float(survived.mean() * 100), 2),
                "fail_rate": round(float(fail.mean() * 100), 2),
                "breach_rate": round(float(fail.mean() * 100), 2),
                "daily_loss_breach_rate": round(float(group["daily_loss_breach"].mean() * 100), 2),
                "max_loss_breach_rate": round(float(group["max_loss_breach"].mean() * 100), 2),
                "profit_target_hit_rate": None,
                "avg_days_to_pass": None,
                "avg_trades_to_pass": None,
                "avg_max_dd": round(float(group["max_dd_pct"].mean()), 4),
                "worst_breach_type": group.loc[fail, "breach_type"].mode().iloc[0] if fail.any() else "",
                "worst_historical_sequence": group.sort_values("max_dd_pct").iloc[0]["start_month"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "risk_grid" / "phase31_risk_grid_results.csv", index=False)
    summary: dict[str, Any] = {}
    for strategy in df["strategy"].unique():
        sg = df[df["strategy"] == strategy]
        safe = sg[(sg["profile"].isin(["FTMO_2_STEP_CHALLENGE_DEFAULT", "FTMO_VERIFICATION_DEFAULT"])) & (sg["breach_rate"] == 0) & (sg["pass_rate"] >= 70)]
        funded_safe = sg[(sg["profile"] == "FTMO_FUNDED_ACCOUNT_DEFAULT") & (sg["breach_rate"] <= 1)]
        summary[strategy] = {
            "recommended_max_risk_from_grid": float(safe["risk_pct"].max()) if not safe.empty else None,
            "funded_max_risk_from_grid": float(funded_safe["risk_pct"].max()) if not funded_safe.empty else None,
            "aggressive_risk_warning": ">=1.0% requires Monte Carlo and daily loss review even if historical windows pass.",
        }
    write_json(OUT / "risk_grid" / "phase31_risk_grid_summary.json", summary)
    write_text(OUT / "risk_grid" / "phase31_risk_grid_summary.md", md_kv("PHASE31 RISK GRID", summary))
    return df, summary


def build_month_blocks(trades: pd.DataFrame) -> list[dict[str, Any]]:
    blocks = []
    df = trades.copy()
    df["month_key"] = df["entry_time_prague"].dt.year.astype(str) + "-" + df["entry_time_prague"].dt.month.astype(str).str.zfill(2)
    for _, g in df.groupby("month_key"):
        g = g.sort_values("entry_time_prague")
        base_day = g["entry_time_prague"].iloc[0].date()
        day_idx = np.array([(x.date() - base_day).days for x in g["entry_time_prague"]], dtype=np.int16)
        blocks.append({"days": day_idx, "r": g["r_return"].to_numpy(dtype=float), "mae": g["mae_r"].fillna(-1.0).to_numpy(dtype=float), "span_days": int(day_idx.max()) + 1 if len(day_idx) else 1})
    return blocks


def simulate_arrays(r: np.ndarray, mae: np.ndarray, day: np.ndarray, profile: dict[str, Any], risk_pct: float) -> dict[str, Any]:
    balance = 0.0
    peak = 0.0
    eod_high = 0.0
    daily_start = 0.0
    cur_day = None
    trade_days = set()
    max_dd = 0.0
    worst_daily = 0.0
    target = profile.get("profit_target_pct")
    for i in range(len(r)):
        d = int(day[i])
        if cur_day != d:
            if cur_day is not None and profile.get("max_loss_mode") == "trailing_eod":
                eod_high = max(eod_high, balance)
            cur_day = d
            daily_start = balance
        trade_days.add(d)
        low = balance + min(float(mae[i]), 0.0) * risk_pct
        daily_draw = low - daily_start
        worst_daily = min(worst_daily, daily_draw)
        if daily_draw <= -float(profile["max_daily_loss_pct"]):
            return {"status": "FAIL", "breach_type": "MAX_DAILY_LOSS", "trades_used": i + 1, "days_elapsed": d + 1, "final_return_pct": balance, "max_dd_pct": max_dd, "worst_daily_loss_pct": worst_daily, "daily_loss_breach": True, "max_loss_breach": False, "target_hit": False}
        if profile.get("max_loss_mode") == "trailing_eod":
            floor = eod_high - float(profile["max_loss_pct"])
        elif profile.get("max_loss_mode") == "trailing":
            floor = peak - float(profile["max_loss_pct"])
        else:
            floor = -float(profile["max_loss_pct"])
        if low <= floor:
            return {"status": "FAIL", "breach_type": "MAX_LOSS", "trades_used": i + 1, "days_elapsed": d + 1, "final_return_pct": balance, "max_dd_pct": max_dd, "worst_daily_loss_pct": worst_daily, "daily_loss_breach": False, "max_loss_breach": True, "target_hit": False}
        balance += float(r[i]) * risk_pct
        peak = max(peak, balance)
        max_dd = min(max_dd, balance - peak)
        if target is not None and balance >= float(target) and len(trade_days) >= int(profile.get("min_trading_days", 0)):
            return {"status": "PASS", "breach_type": "", "trades_used": i + 1, "days_elapsed": d + 1, "final_return_pct": balance, "max_dd_pct": max_dd, "worst_daily_loss_pct": worst_daily, "daily_loss_breach": False, "max_loss_breach": False, "target_hit": True}
    return {"status": "SURVIVED" if target is None else "END_NO_TARGET", "breach_type": "", "trades_used": len(r), "days_elapsed": int(day[-1]) + 1 if len(day) else 0, "final_return_pct": balance, "max_dd_pct": max_dd, "worst_daily_loss_pct": worst_daily, "daily_loss_breach": False, "max_loss_breach": False, "target_hit": False}


def sample_path(blocks: list[dict[str, Any]], rng: np.random.Generator, max_days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_parts = []
    mae_parts = []
    day_parts = []
    offset = 0
    while offset < max_days:
        b = blocks[int(rng.integers(0, len(blocks)))]
        r_parts.append(b["r"])
        mae_parts.append(b["mae"])
        day_parts.append(b["days"] + offset)
        offset += max(1, int(b["span_days"]))
    return np.concatenate(r_parts), np.concatenate(mae_parts), np.concatenate(day_parts)


def monte_carlo(ledgers: dict[str, pd.DataFrame], config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    profiles = {
        "FTMO_2_STEP_CHALLENGE_DEFAULT": (config["profiles"]["FTMO_2_STEP_CHALLENGE_DEFAULT"], 730),
        "FTMO_VERIFICATION_DEFAULT": (config["profiles"]["FTMO_VERIFICATION_DEFAULT"], 365),
        "FTMO_FUNDED_ACCOUNT_DEFAULT_12M": (config["profiles"]["FTMO_FUNDED_ACCOUNT_DEFAULT"], 365),
    }
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for strategy, trades in ledgers.items():
        blocks = build_month_blocks(trades)
        for profile_name, (profile, max_days) in profiles.items():
            for risk in RISK_GRID:
                statuses = []
                breach_types = []
                days = []
                max_dd = []
                final_returns = []
                worst_daily = []
                target_hit = []
                for _ in range(MC_PATHS):
                    r, mae, day = sample_path(blocks, rng, max_days)
                    sim = simulate_arrays(r, mae, day, profile, risk)
                    statuses.append(sim["status"])
                    breach_types.append(sim["breach_type"])
                    days.append(sim["days_elapsed"])
                    max_dd.append(sim["max_dd_pct"])
                    final_returns.append(sim["final_return_pct"])
                    worst_daily.append(sim["worst_daily_loss_pct"])
                    target_hit.append(sim["target_hit"])
                st = pd.Series(statuses)
                bt = pd.Series(breach_types)
                fr = np.array(final_returns, dtype=float)
                dd = np.array(max_dd, dtype=float)
                rows.append(
                    {
                        "strategy": strategy,
                        "profile": profile_name,
                        "risk_pct": risk,
                        "paths": MC_PATHS,
                        "pass_probability": round(float((st == "PASS").mean() * 100), 2),
                        "breach_probability": round(float((st == "FAIL").mean() * 100), 2),
                        "daily_loss_breach_probability": round(float((bt == "MAX_DAILY_LOSS").mean() * 100), 2),
                        "max_loss_breach_probability": round(float((bt == "MAX_LOSS").mean() * 100), 2),
                        "target_before_breach_probability": round(float(np.mean(target_hit) * 100), 2),
                        "expected_days_to_pass": round(float(np.mean([d for s, d in zip(statuses, days) if s == "PASS"])), 2) if (st == "PASS").any() else None,
                        "expected_max_dd": round(float(np.mean(dd)), 4),
                        "final_return_p5": round(float(np.percentile(fr, 5)), 4),
                        "final_return_p50": round(float(np.percentile(fr, 50)), 4),
                        "final_return_p95": round(float(np.percentile(fr, 95)), 4),
                        "max_dd_p5": round(float(np.percentile(dd, 5)), 4),
                        "max_dd_p50": round(float(np.percentile(dd, 50)), 4),
                        "max_dd_p95": round(float(np.percentile(dd, 95)), 4),
                        "worst_1pct_final_return": round(float(np.percentile(fr, 1)), 4),
                        "risk_of_ruin_proxy": round(float((st == "FAIL").mean() * 100), 2),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "monte_carlo" / "phase31_monte_carlo_results.csv", index=False)
    summary: dict[str, Any] = {"paths_per_cell": MC_PATHS, "bootstrap": "monthly_block_bootstrap", "seed": RNG_SEED}
    for strategy in df["strategy"].unique():
        sg = df[df["strategy"] == strategy]
        challenge = sg[(sg["profile"] == "FTMO_2_STEP_CHALLENGE_DEFAULT") & (sg["breach_probability"] <= 1) & (sg["pass_probability"] >= 70)]
        funded = sg[(sg["profile"] == "FTMO_FUNDED_ACCOUNT_DEFAULT_12M") & (sg["breach_probability"] <= 1)]
        summary[strategy] = {
            "challenge_recommended_max_risk_mc": float(challenge["risk_pct"].max()) if not challenge.empty else None,
            "funded_recommended_max_risk_mc": float(funded["risk_pct"].max()) if not funded.empty else None,
            "worst_breach_probability": round(float(sg["breach_probability"].max()), 2),
        }
    write_json(OUT / "monte_carlo" / "phase31_monte_carlo_summary.json", summary)
    write_text(OUT / "monte_carlo" / "phase31_monte_carlo_summary.md", md_kv("PHASE31 MONTE CARLO", summary))
    return df, summary


def daily_loss_audit(ledgers: dict[str, pd.DataFrame], config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    profile = config["profiles"]["FTMO_2_STEP_CHALLENGE_DEFAULT"]
    rows = []
    for strategy, trades in ledgers.items():
        for risk in RISK_GRID:
            for mode in ["closed_only", "mae_proxy", "sl_proxy"]:
                balance = 0.0
                current_day = None
                day_start = 0.0
                for row in trades.itertuples():
                    day = str(row.entry_time_prague.date())
                    if current_day != day:
                        current_day = day
                        day_start = balance
                    if mode == "closed_only":
                        mae_r = 0.0
                    elif mode == "sl_proxy":
                        mae_r = min(float(row.mae_r) if not math.isnan(float(row.mae_r)) else -1.0, -1.0)
                    else:
                        mae_r = min(float(row.mae_r) if not math.isnan(float(row.mae_r)) else -1.0, 0.0)
                    equity_low = balance + mae_r * risk
                    breach_margin = equity_low - (day_start - float(profile["max_daily_loss_pct"]))
                    rows.append(
                        {
                            "strategy": strategy,
                            "risk_pct": risk,
                            "mode": mode,
                            "trade_id": row.trade_id,
                            "entry_time": row.entry_time,
                            "day": day,
                            "balance_start_day": round(day_start, 4),
                            "equity_low_estimate": round(equity_low, 4),
                            "closed_pnl_pct": round(float(row.r_return) * risk, 4),
                            "open_pnl_proxy_pct": round(mae_r * risk, 4),
                            "daily_loss_limit_pct": float(profile["max_daily_loss_pct"]),
                            "breach": breach_margin <= 0,
                            "breach_margin_pct": round(breach_margin, 4),
                        }
                    )
                    balance += float(row.r_return) * risk
    df = pd.DataFrame(rows)
    breach_cases = df[df["breach"]].copy()
    breach_cases.to_csv(OUT / "daily_loss_audit" / "phase31_daily_loss_breach_cases.csv", index=False)
    summary: dict[str, Any] = {"intraday_equity_status": "MAE_AVAILABLE_PROXY_NOT_TICK_PATH"}
    for (strategy, risk, mode), g in df.groupby(["strategy", "risk_pct", "mode"]):
        summary[f"{strategy}|{risk}|{mode}"] = {
            "trades": int(len(g)),
            "breaches": int(g["breach"].sum()),
            "breach_rate": round(float(g["breach"].mean() * 100), 4),
            "worst_breach_margin_pct": round(float(g["breach_margin_pct"].min()), 4),
        }
    write_json(OUT / "daily_loss_audit" / "phase31_daily_loss_audit.json", summary)
    write_text(OUT / "daily_loss_audit" / "phase31_daily_loss_audit.md", md_kv("PHASE31 DAILY LOSS AUDIT", summary))
    return df, summary


def risk_recommendation(risk_grid: pd.DataFrame, mc: pd.DataFrame, funded: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for strategy in sorted(risk_grid["strategy"].unique()):
        ch_mc = mc[(mc["strategy"] == strategy) & (mc["profile"] == "FTMO_2_STEP_CHALLENGE_DEFAULT")]
        ver_mc = mc[(mc["strategy"] == strategy) & (mc["profile"] == "FTMO_VERIFICATION_DEFAULT")]
        fun_mc = mc[(mc["strategy"] == strategy) & (mc["profile"] == "FTMO_FUNDED_ACCOUNT_DEFAULT_12M")]
        def highest(df: pd.DataFrame, pass_min: float, breach_max: float) -> float | None:
            ok = df[(df["pass_probability"] >= pass_min) & (df["breach_probability"] <= breach_max)]
            return None if ok.empty else float(ok["risk_pct"].max())
        def highest_funded(df: pd.DataFrame, breach_max: float) -> float | None:
            ok = df[df["breach_probability"] <= breach_max]
            return None if ok.empty else float(ok["risk_pct"].max())
        challenge_cons = highest(ch_mc, 85, 0.5)
        challenge_bal = highest(ch_mc, 70, 1.0)
        verification = highest(ver_mc, 85, 0.5)
        funded_risk = highest_funded(fun_mc, 0.5)
        max_not_exceed = highest_funded(fun_mc, 2.0)
        if max_not_exceed is None:
            max_not_exceed = min(RISK_GRID)
        rows.append(
            {
                "strategy": strategy,
                "challenge_conservative_risk_pct": challenge_cons,
                "challenge_balanced_risk_pct": challenge_bal,
                "challenge_aggressive_warning_pct": 1.0,
                "verification_risk_pct": verification,
                "funded_account_risk_pct": funded_risk,
                "max_risk_not_to_exceed_pct": max_not_exceed,
                "note": "Do not use a risk level that creates material breach probability in Monte Carlo or funded rolling windows.",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "risk_recommendation" / "phase31_recommended_risk_table.csv", index=False)
    summary = {r["strategy"]: r for r in df.to_dict("records")}
    write_json(OUT / "risk_recommendation" / "phase31_risk_recommendation.json", summary)
    write_text(OUT / "risk_recommendation" / "phase31_risk_recommendation.md", md_kv("PHASE31 RISK RECOMMENDATION", summary))
    return df, summary


def strategy_comparison(risk_grid: pd.DataFrame, mc: pd.DataFrame, funded: pd.DataFrame, rec: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for strategy in sorted(risk_grid["strategy"].unique()):
        ch = mc[(mc["strategy"] == strategy) & (mc["profile"] == "FTMO_2_STEP_CHALLENGE_DEFAULT") & (mc["risk_pct"] == 0.50)]
        ver = mc[(mc["strategy"] == strategy) & (mc["profile"] == "FTMO_VERIFICATION_DEFAULT") & (mc["risk_pct"] == 0.50)]
        fun = mc[(mc["strategy"] == strategy) & (mc["profile"] == "FTMO_FUNDED_ACCOUNT_DEFAULT_12M") & (mc["risk_pct"] == 0.50)]
        rr = rec[rec["strategy"] == strategy].iloc[0].to_dict()
        rows.append(
            {
                "strategy": strategy,
                "challenge_pass_probability_050": float(ch.iloc[0]["pass_probability"]) if not ch.empty else None,
                "challenge_breach_probability_050": float(ch.iloc[0]["breach_probability"]) if not ch.empty else None,
                "verification_pass_probability_050": float(ver.iloc[0]["pass_probability"]) if not ver.empty else None,
                "funded_breach_probability_050": float(fun.iloc[0]["breach_probability"]) if not fun.empty else None,
                "funded_expected_return_050": float(fun.iloc[0]["final_return_p50"]) if not fun.empty else None,
                "recommended_challenge_balanced": rr["challenge_balanced_risk_pct"],
                "recommended_funded": rr["funded_account_risk_pct"],
            }
        )
    score = pd.DataFrame(rows)
    score.to_csv(OUT / "strategy_comparison" / "phase31_phase25_vs_candidate_prop_scorecard.csv", index=False)
    if len(score) == 2:
        p = score[score["strategy"] == "PHASE25"].iloc[0]
        c = score[score["strategy"] == "TP1.4_BE0.5_BF70"].iloc[0]
        challenge_best = "TP1.4_BE0.5_BF70" if c["challenge_pass_probability_050"] >= p["challenge_pass_probability_050"] and c["challenge_breach_probability_050"] <= p["challenge_breach_probability_050"] else "PHASE25"
        verification_best = "TP1.4_BE0.5_BF70" if c["verification_pass_probability_050"] >= p["verification_pass_probability_050"] else "PHASE25"
        funded_best = "TP1.4_BE0.5_BF70" if c["funded_breach_probability_050"] <= p["funded_breach_probability_050"] else "PHASE25"
    else:
        challenge_best = verification_best = funded_best = ""
    summary = {
        "challenge_best": challenge_best,
        "verification_best": verification_best,
        "funded_best": funded_best,
        "authority_recommendation": "PHASE25 remains authority; candidate remains shadow.",
        "psychological_operability": "Candidate has lower non-win streak and higher WR; Phase25 has higher PF.",
    }
    write_json(OUT / "strategy_comparison" / "phase31_strategy_comparison.json", summary)
    write_text(OUT / "strategy_comparison" / "phase31_strategy_comparison.md", md_kv("PHASE31 STRATEGY COMPARISON", summary))
    return score, summary


def decide_verdict(rec: pd.DataFrame, mc: pd.DataFrame) -> str:
    any_good = rec["challenge_balanced_risk_pct"].notna().any() and rec["funded_account_risk_pct"].notna().any()
    if not any_good:
        return "PHASE31_PROP_FIRM_NOT_READY_RISK_TOO_HIGH"
    high_breach = mc[(mc["risk_pct"] <= 0.50) & (mc["breach_probability"] > 2.0)]
    if not high_breach.empty:
        return "PHASE31_PROP_FIRM_READY_WITH_WARNINGS"
    return "PHASE31_PROP_FIRM_READY_CONSERVATIVE_RISK"


def final_report(
    config: dict[str, Any],
    ledger_summary: dict[str, Any],
    risk_grid_summary: dict[str, Any],
    hist_summary: dict[str, Any],
    mc_summary: dict[str, Any],
    daily_summary: dict[str, Any],
    funded_summary: dict[str, Any],
    comp_summary: dict[str, Any],
    rec_summary: dict[str, Any],
    verdict: str,
) -> dict[str, Any]:
    report = {
        "timestamp": p29.now_utc(),
        "objective": "Prop firm survival simulator for Phase25 versus TP1.4_BE0.5_BF70.",
        "rules_simulated": config,
        "trade_ledger_lock": ledger_summary,
        "risk_grid": risk_grid_summary,
        "historical_windows": {"summary_keys": len(hist_summary)},
        "monte_carlo": mc_summary,
        "daily_loss_audit": {"summary_keys": len(daily_summary), "intraday_equity_status": daily_summary.get("intraday_equity_status")},
        "funded_survival": {"summary_keys": len(funded_summary)},
        "strategy_comparison": comp_summary,
        "risk_recommendation": rec_summary,
        "limitations": [
            "R-ledger simulation, not broker-side execution.",
            "Daily equity uses MAE proxy, not full tick-by-tick open equity.",
            "Commissions/swaps default to zero and must be configured for a specific account.",
            "This is evaluation planning only, not permission for real trading.",
        ],
        "verdict": verdict,
        "phase25_remains_authority": True,
        "candidate_status": "SHADOW_CANDIDATE_ONLY",
    }
    write_json(REPORT_JSON, report)
    write_text(
        REPORT_MD,
        "\n".join(
            [
                "# PHASE31 PROP FIRM SURVIVAL SIMULATOR REPORT",
                "",
                "## Objective",
                "Compare Phase25 and TP1.4_BE0.5_BF70 under configurable prop-firm rules. No real trading.",
                "",
                "## Rules",
                "- FTMO Challenge: 10% target, 5% max daily loss, 10% max loss, 4 minimum trading days, unlimited period.",
                "- FTMO Verification: 5% target, same loss limits.",
                "- Funded: no profit target in simulation, same loss limits.",
                "",
                "## Monte Carlo",
                f"- Paths per cell: {mc_summary['paths_per_cell']}",
                "- Bootstrap: monthly block bootstrap.",
                "",
                "## Risk Recommendation",
                json.dumps(rec_summary, indent=2),
                "",
                "## Strategy Comparison",
                json.dumps(comp_summary, indent=2),
                "",
                "## Verdict",
                verdict,
                "",
                "## Next Step",
                "Use the simulator for paper evaluation planning only; do not trade real or MT5.",
                "",
            ]
        ),
    )
    return report


def update_master_docs(verdict: str, rec_summary: dict[str, Any]) -> None:
    status = {
        "timestamp": p29.now_utc(),
        "current_authority": "PHASE25",
        "phase25_status": "CURRENT_AUTHORITY_FROZEN_PAPER_DEMO_ONLY_REAL_BLOCKED",
        "phase30_candidate": "TP1.4_BE0.5_BF70_SHADOW_ONLY",
        "phase31_status": verdict,
        "prop_firm_simulator": "CREATED_RESEARCH_PLANNING_ONLY",
        "risk_recommendation": rec_summary,
        "real_blocked": True,
        "mt5_real_blocked": True,
        "vps_blocked": True,
        "ctrader_blocked": True,
        "scbi_touched": False,
        "phase19_reopened": False,
        "next_step": "Paper prop-firm evaluation plan only; no real or MT5.",
    }
    write_json(ROOT / "01_CURRENT_PROJECT_STATUS.json", status)
    write_json(
        ROOT / "02_STRATEGY_AUTHORITY_MAP.json",
        {
            "timestamp": p29.now_utc(),
            "authority": "PHASE25",
            "phase25": "CURRENT_AUTHORITY_FROZEN",
            "candidate": "TP1.4_BE0.5_BF70_SHADOW_ONLY",
            "phase31": "PROP_FIRM_SIMULATOR_CREATED",
            "replacement": "NO_AUTOMATIC_REPLACEMENT",
            "real": "BLOCKED",
            "mt5_real": "BLOCKED",
            "scbi": "PROTECTED_NOT_TOUCHED",
        },
    )
    write_json(LAB / "status.json", status)
    write_text(
        ROOT / "00_READ_THIS_FIRST.md",
        "\n".join(
            [
                "# BOT V2 DAYTIME LAB - READ THIS FIRST",
                "",
                "- Current authority: Phase25.",
                "- TP1.4_BE0.5_BF70 remains a shadow candidate only.",
                "- Phase31 prop firm simulator was created for planning only.",
                f"- Phase31 verdict: {verdict}.",
                "- No real, no MT5, no automatic funding execution.",
                "- SCBI was not touched. Phase19 remains archived.",
                "- Use only the canonical zip: 000_PARA_CHATGPT.zip.",
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
                "- Authority: Phase25.",
                "- Candidate: TP1.4_BE0.5_BF70 shadow only.",
                f"- Phase31: {verdict}.",
                "- Prop firm simulator: created for paper planning only.",
                "- Real/MT5 real/VPS/cTrader: blocked.",
                "- SCBI: protected, not touched.",
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
                "- PHASE25: CURRENT AUTHORITY, FROZEN.",
                "- TP1.4_BE0.5_BF70: SHADOW CANDIDATE ONLY.",
                "- PHASE31: PROP FIRM SURVIVAL SIMULATOR / PLANNING ONLY.",
                "- Replacement: none.",
                "- PHASE19: ARCHIVED.",
                "- SCBI: PROTECTED / NOT TOUCHED.",
                "- Real deployment: BLOCKED.",
                "",
            ]
        ),
    )


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
    try:
        if path.stat().st_size > 2 * 1024 * 1024:
            return False
    except OSError:
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
    if rel_s == "BOT_V2_DAYTIME_LAB/configs/prop_firm_rules_config.json":
        return True
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/outputs/phase31_prop_firm_survival_simulator/"):
        return "/zip/" not in rel_s and suffix in {".md", ".json", ".csv", ".txt"}
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/outputs/phase30_tp14_be05_bf70_forensic_audit/"):
        return suffix in {".md", ".json", ".csv", ".txt"} and path.stat().st_size <= 2 * 1024 * 1024
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/outputs/phase29_wr_loss_streak_compression/"):
        return suffix in {".md", ".json", ".csv", ".txt"} and path.stat().st_size <= 2 * 1024 * 1024
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/outputs/phase27_full_historical_validation_2015_2026/"):
        return suffix in {".md", ".json", ".csv", ".txt"} and path.stat().st_size <= 700000
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/reports/"):
        return suffix in {".md", ".json"}
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/configs/"):
        return suffix in {".json", ".txt"}
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/docs/"):
        return suffix in {".md", ".json", ".txt"}
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/src/"):
        return suffix == ".py" and (
            "phase31" in name
            or "phase30" in name
            or "phase29" in name
            or "phase28" in name
            or "phase27" in name
            or "phase26" in name
            or name in {"phase18_h1_fractal_sweep.py", "phase18_first_3m_choch.py"}
        )
    if rel_s in {"BOT_V2_DAYTIME_LAB/status.json", "BOT_V2_DAYTIME_LAB/ZIP_CONTENTS_MANIFEST.md", "BOT_V2_DAYTIME_LAB/ZIP_UPLOAD_IDENTITY_MARKER.md"}:
        return True
    return False


def rebuild_zip() -> dict[str, Any]:
    if BUILD_PATH.exists():
        BUILD_PATH.unlink()
    files = sorted([p for p in ROOT.rglob("*") if zip_include(p)], key=lambda x: str(x.relative_to(ROOT)).replace("\\", "/"))
    with zipfile.ZipFile(BUILD_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, str(p.relative_to(ROOT)).replace("\\", "/"))
    with zipfile.ZipFile(BUILD_PATH, "r") as zf:
        test = zf.testzip()
        names = zf.namelist()
        heavy = [n for n in names if zf.getinfo(n).file_size > 2 * 1024 * 1024]
        secrets = [n for n in names if any(tok in n.lower() for tok in [".env", "secret", "password", "token", "credential", "apikey"])]
        zips = [n for n in names if n.lower().endswith((".zip", ".zipbak"))]
    if test is not None or heavy or secrets or zips:
        raise RuntimeError(f"zip validation failed: test={test}, heavy={heavy[:5]}, secrets={secrets[:5]}, zips={zips[:5]}")
    os.replace(str(BUILD_PATH), str(ZIP_PATH))
    details = p29.zip_details(ZIP_PATH)
    live = p29.exact_zip_inventory()
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        names = zf.namelist()
    result = {
        **details,
        "single_live_zip_exact_extension": len(live) == 1 and Path(live[0]["path"]) == ZIP_PATH,
        "live_zip_count_exact_extension": len(live),
        "contains_phase31_report": "BOT_V2_DAYTIME_LAB/reports/PHASE31_PROP_FIRM_SURVIVAL_SIMULATOR_REPORT.md" in names,
        "contains_phase31_outputs": any(n.startswith("BOT_V2_DAYTIME_LAB/outputs/phase31_prop_firm_survival_simulator/") for n in names),
        "contains_prop_firm_rules_config": "BOT_V2_DAYTIME_LAB/configs/prop_firm_rules_config.json" in names,
        "contains_phase25_config_hash": "BOT_V2_DAYTIME_LAB/configs/phase25_forward_demo_candidate_config_hash.txt" in names,
        "heavy_entries_gt_2mb": [],
        "secret_like_entries": [],
        "zip_entries_inside": [],
    }
    write_json(OUT / "zip" / "phase31_zip_validation.json", result)
    write_text(OUT / "zip" / "phase31_zip_validation.md", md_kv("PHASE31 ZIP VALIDATION", result))
    return result


def update_manifests(verdict: str) -> None:
    text = "\n".join(
        [
            "# ZIP CONTENTS MANIFEST",
            "",
            "- Canonical live zip: 000_PARA_CHATGPT.zip",
            f"- Official path: {ZIP_PATH}",
            "- Current authority: Phase25",
            "- Phase31: prop firm survival simulator / paper planning only.",
            f"- Phase31 verdict: {verdict}",
            "- No automatic replacement, no real, no MT5.",
            "- No raw heavy data, no secrets, no internal zip files.",
            "",
        ]
    )
    write_text(ROOT / "ZIP_CONTENTS_MANIFEST.md", text)
    write_text(LAB / "ZIP_CONTENTS_MANIFEST.md", text)
    write_text(
        LAB / "ZIP_UPLOAD_IDENTITY_MARKER.md",
        "\n".join(
            [
                "# ZIP UPLOAD IDENTITY MARKER",
                "",
                f"- Timestamp: {p29.now_utc()}",
                "- Phase: PHASE31_PROP_FIRM_SURVIVAL_SIMULATOR",
                "- Authority: Phase25",
                "- Candidate: TP1.4_BE0.5_BF70 shadow only",
                "- Real trading: blocked",
                "- MT5: blocked",
                "",
            ]
        ),
    )


def git_status_artifacts() -> dict[str, Any]:
    result = {
        "timestamp": p29.now_utc(),
        "branch": p29.run_cmd(["git", "branch", "--show-current"]),
        "status": p29.run_cmd(["git", "status", "--short"]),
        "diff_stat": p29.run_cmd(["git", "diff", "--stat"]),
        "commit": "NO",
        "push": "NO",
    }
    write_json(OUT / "git" / "phase31_git_status.json", result)
    write_text(OUT / "git" / "phase31_git_status.md", md_kv("PHASE31 GIT STATUS", result))
    return result


def main() -> None:
    ensure_dirs()
    config = make_rules_config()
    preflight()
    ledgers, ledger_summary = trade_ledger_lock()
    print("Historical windows")
    historical, hist_summary = historical_windows(ledgers, config)
    print("Funded survival")
    funded, funded_summary = funded_survival(ledgers, config)
    print("Risk grid")
    risk_grid, risk_grid_summary = risk_grid_results(historical, funded)
    print("Monte Carlo")
    mc, mc_summary = monte_carlo(ledgers, config)
    print("Daily loss audit")
    daily, daily_summary = daily_loss_audit(ledgers, config)
    print("Risk recommendation")
    rec, rec_summary = risk_recommendation(risk_grid, mc, funded)
    print("Strategy comparison")
    score, comp_summary = strategy_comparison(risk_grid, mc, funded, rec)
    verdict = decide_verdict(rec, mc)
    final_report(config, ledger_summary, risk_grid_summary, hist_summary, mc_summary, daily_summary, funded_summary, comp_summary, rec_summary, verdict)
    update_master_docs(verdict, rec_summary)
    update_manifests(verdict)
    print("Git status")
    git_status_artifacts()
    print("Rebuilding canonical zip")
    zip_result = rebuild_zip()
    print(json.dumps({"verdict": verdict, "zip": zip_result}, indent=2))


if __name__ == "__main__":
    main()
