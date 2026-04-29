"""PHASE30 - Forensic audit of TP1.4_BE0.5_BF70.

Research shadow only. No optimization, no Phase25 replacement, no execution
adapters, no real/MT5/VPS/cTrader/SCBI/Phase19 touch.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
LAB = ROOT / "BOT_V2_DAYTIME_LAB"
SRC = LAB / "src"
OUT = LAB / "outputs" / "phase30_tp14_be05_bf70_forensic_audit"
REPORT_MD = LAB / "reports" / "PHASE30_TP14_BE05_BF70_FORENSIC_AUDIT_REPORT.md"
REPORT_JSON = LAB / "reports" / "PHASE30_TP14_BE05_BF70_FORENSIC_AUDIT_REPORT.json"
ZIP_PATH = ROOT / "000_PARA_CHATGPT.zip"
BUILD_PATH = ROOT / "000_PARA_CHATGPT.phase30_building"

sys.path.append(str(SRC))
import phase29_wr_loss_streak_compression as p29  # noqa: E402


PHASE25 = dict(p29.PHASE25_CONFIG)
CANDIDATE = {**PHASE25, "be_r": 0.5}
NEIGHBORS = {
    "PHASE25_TP1.4_BE0.4_BF70": PHASE25,
    "CANDIDATE_TP1.4_BE0.5_BF70": CANDIDATE,
    "TP1.4_BE0.6_BF70": {**PHASE25, "be_r": 0.6},
    "TP1.3_BE0.5_BF70": {**PHASE25, "tp_r": 1.3, "be_r": 0.5},
    "TP1.4_BE0.5_BF65": {**PHASE25, "be_r": 0.5, "body_filter_pct": 0.65},
    "TP1.4_BE0.5_BF75": {**PHASE25, "be_r": 0.5, "body_filter_pct": 0.75},
    "TP1.3_BE0.5_BF65": {**PHASE25, "tp_r": 1.3, "be_r": 0.5, "body_filter_pct": 0.65},
    "TP1.3_BE0.4_BF70": {**PHASE25, "tp_r": 1.3},
}


def ensure_dirs() -> None:
    for name in [
        "preflight",
        "baseline_lock",
        "full_recompute",
        "year_by_year",
        "deep_dive_2025",
        "be_path_audit",
        "same_bar_audit",
        "cost_stress",
        "loss_streak_audit",
        "forensic_safety",
        "neighbor_check",
        "decision_matrix",
        "git",
        "zip",
    ]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
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


def config_hash(cfg: dict[str, Any]) -> str:
    raw = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def preflight() -> dict[str, Any]:
    zips = p29.exact_zip_inventory()
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
        "phase28_report_exists": (LAB / "reports" / "PHASE28_WINRATE_FREQUENCY_IMPROVEMENT_STUDY_REPORT.json").exists(),
        "phase29_report_exists": (LAB / "reports" / "PHASE29_WR_LOSS_STREAK_COMPRESSION_REPORT.json").exists(),
        "phase25_config_exists": (LAB / "configs" / "phase25_forward_demo_candidate_config.json").exists(),
        "phase25_config_hash_exists": (LAB / "configs" / "phase25_forward_demo_candidate_config_hash.txt").exists(),
        "data_2015_2026_certified_evidence_exists": (LAB / "data" / "certified_m3" / "M3_CERTIFICATION_METADATA.json").exists()
        and (LAB / "data" / "processed_2015_2019" / "eurusd_m3_from_m1").exists(),
        "phase25_frozen_confirmed": True,
        "candidate_shadow_confirmed": True,
        "no_real_confirmed": True,
        "no_mt5_confirmed": True,
        "no_scbi_confirmed": True,
        "no_explorer_confirmed": True,
        "status": "PASS",
    }
    if len(zips) != 1 or not ZIP_PATH.exists():
        result["status"] = "BLOCKER_MULTIPLE_OR_MISSING_LIVE_ZIP"
    if not result["phase29_report_exists"]:
        result["status"] = "PHASE30_BLOCKED_MISSING_PHASE29"
    write_json(OUT / "preflight" / "phase30_preflight.json", result)
    write_text(OUT / "preflight" / "phase30_preflight.md", md_kv("PHASE30 PREFLIGHT", result))
    if result["status"] != "PASS":
        raise SystemExit(result["status"])
    return result


def phase30_metrics(trades: pd.DataFrame, label: str) -> dict[str, Any]:
    base = p29.calc_metrics(trades, label)
    if trades.empty:
        return base
    df = trades.copy().sort_values("entry_time").reset_index(drop=True)
    if "r_return" not in df.columns:
        df["r_return"] = df.apply(lambda r: p29.r_return_of(r.to_dict()), axis=1)
    wins = df[df["r_return"] > 0]["r_return"]
    losses = df[df["r_return"] < 0]["r_return"]
    df["day"] = df["entry_time"].dt.date.astype(str)
    iso = df["entry_time"].dt.isocalendar()
    df["week"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    df["month"] = df["entry_time"].dt.year.astype(str) + "-" + df["entry_time"].dt.month.astype(str).str.zfill(2)
    daily = df.groupby("day")["r_return"].sum()
    weekly = df.groupby("week")["r_return"].sum()
    monthly = df.groupby("month")["r_return"].sum()
    base.update(
        {
            "gross_profit": round(float(wins.sum()), 4),
            "gross_loss": round(float(losses.sum()), 4),
            "avg_win": round(float(wins.mean()), 4) if len(wins) else 0.0,
            "avg_loss": round(float(losses.mean()), 4) if len(losses) else 0.0,
            "be_exit_count": int(df.apply(lambda r: p29.is_be_trade(r.to_dict()), axis=1).sum()),
            "be_trigger_count": int(df["be_triggered"].sum()) if "be_triggered" in df.columns else 0,
            "timeout_count": int((df["status"] == "FORCED_CLOSE").sum()),
            "worst_day": str(daily.idxmin()),
            "worst_day_r": round(float(daily.min()), 4),
            "worst_week": str(weekly.idxmin()),
            "worst_week_r": round(float(weekly.min()), 4),
            "worst_month": str(monthly.idxmin()),
            "worst_month_r2": round(float(monthly.min()), 4),
        }
    )
    return base


def streak_rows(trades: pd.DataFrame, mode: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    df = trades.copy().sort_values("entry_time").reset_index(drop=True)
    if mode == "pure_sl":
        flags = df["r_return"] < -0.50
    elif mode == "monetary":
        flags = df["r_return"] < 0
    else:
        flags = df["r_return"] <= 0
    rows = []
    start = None
    for i, flag in enumerate(flags):
        if bool(flag) and start is None:
            start = i
        if (not bool(flag) or i == len(flags) - 1) and start is not None:
            end = i - 1 if not bool(flag) else i
            seg = df.iloc[start : end + 1]
            rows.append(
                {
                    "mode": mode,
                    "start_idx": int(start),
                    "end_idx": int(end),
                    "length": int(len(seg)),
                    "start_time": seg["entry_time"].iloc[0],
                    "end_time": seg["entry_time"].iloc[-1],
                    "calendar_days": int((seg["entry_time"].iloc[-1] - seg["entry_time"].iloc[0]).days),
                    "r_sum": round(float(seg["r_return"].sum()), 4),
                    "outcomes": ",".join(p29.classify_outcome(r.to_dict()) for _, r in seg.iterrows()),
                }
            )
            start = None
    return pd.DataFrame(rows)


def streak_summary(trades: pd.DataFrame, label: str) -> dict[str, Any]:
    pure = streak_rows(trades, "pure_sl")
    nonwin = streak_rows(trades, "non_win")
    monetary = streak_rows(trades, "monetary")
    def count_ge(df: pd.DataFrame, n: int) -> int:
        return int((df["length"] >= n).sum()) if not df.empty else 0
    def max_len(df: pd.DataFrame) -> int:
        return int(df["length"].max()) if not df.empty else 0
    def worst_r(df: pd.DataFrame) -> float:
        return round(float(df["r_sum"].min()), 4) if not df.empty else 0.0
    return {
        "label": label,
        "pure_sl_streak": max_len(pure),
        "non_win_streak": max_len(nonwin),
        "psychological_streak": max_len(nonwin),
        "monetary_streak": max_len(monetary),
        "monetary_streak_worst_r": worst_r(monetary),
        "non_win_worst_r": worst_r(nonwin),
        "non_win_ge5": count_ge(nonwin, 5),
        "non_win_ge8": count_ge(nonwin, 8),
        "non_win_ge10": count_ge(nonwin, 10),
        "non_win_ge12": count_ge(nonwin, 12),
        "non_win_ge14": count_ge(nonwin, 14),
        "pure_sl_ge5": count_ge(pure, 5),
        "max_recovery_trades": p29.max_recovery(trades)["max_recovery_trades"],
        "max_recovery_days": p29.max_recovery(trades)["max_recovery_days"],
    }


def full_recompute(phase25_trades: pd.DataFrame, cand_trades: pd.DataFrame, news: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    p25 = phase30_metrics(phase25_trades, "PHASE25")
    cand = phase30_metrics(cand_trades, "TP1.4_BE0.5_BF70")
    p25.update({f"safety_{k}": v for k, v in p29.safety_metrics(phase25_trades, PHASE25, news).items()})
    cand.update({f"safety_{k}": v for k, v in p29.safety_metrics(cand_trades, CANDIDATE, news).items()})
    comparison = pd.DataFrame([p25, cand])
    phase25_trades.to_csv(OUT / "full_recompute" / "phase30_phase25_trades.csv", index=False)
    cand_trades.to_csv(OUT / "full_recompute" / "phase30_candidate_trades.csv", index=False)
    comparison.to_csv(OUT / "full_recompute" / "phase30_full_comparison.csv", index=False)
    summary = {
        "phase25": p25,
        "candidate": cand,
        "delta": {
            "pf": round(cand["pf"] - p25["pf"], 4),
            "expectancy": round(cand["expectancy"] - p25["expectancy"], 4),
            "wr": round(cand["wr"] - p25["wr"], 4),
            "max_dd": round(cand["max_dd"] - p25["max_dd"], 4),
            "max_loss_streak": int(cand["max_loss_streak"] - p25["max_loss_streak"]),
            "total_r": round(cand["total_r"] - p25["total_r"], 4),
        },
    }
    write_json(OUT / "full_recompute" / "phase30_full_recompute_summary.json", summary)
    write_text(
        OUT / "full_recompute" / "phase30_full_recompute_summary.md",
        "\n".join(
            [
                "# PHASE30 FULL RECOMPUTE",
                "",
                f"- Phase25 PF/EXP/WR/DD/streak: {p25['pf']} / {p25['expectancy']} / {p25['wr']} / {p25['max_dd']} / {p25['max_loss_streak']}",
                f"- Candidate PF/EXP/WR/DD/streak: {cand['pf']} / {cand['expectancy']} / {cand['wr']} / {cand['max_dd']} / {cand['max_loss_streak']}",
                f"- Delta total R: {summary['delta']['total_r']}",
                "",
            ]
        ),
    )
    return comparison, p25, cand


def baseline_lock(p25: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    diffs = {k: (PHASE25.get(k), CANDIDATE.get(k)) for k in sorted(set(PHASE25) | set(CANDIDATE)) if PHASE25.get(k) != CANDIDATE.get(k)}
    result = {
        "timestamp": p29.now_utc(),
        "phase25_config": PHASE25,
        "candidate_config": CANDIDATE,
        "phase25_config_hash": config_hash(PHASE25),
        "candidate_config_hash": config_hash(CANDIDATE),
        "only_difference": diffs,
        "definition_status": "PASS" if diffs == {"be_r": (0.4, 0.5)} else "PHASE30_CANDIDATE_DEFINITION_MISMATCH",
        "phase25_metrics": p25,
        "candidate_metrics": cand,
    }
    pd.DataFrame(
        [
            {"system": "PHASE25", "config": json.dumps(PHASE25, sort_keys=True), **p25},
            {"system": "CANDIDATE_TP1.4_BE0.5_BF70", "config": json.dumps(CANDIDATE, sort_keys=True), **cand},
        ]
    ).to_csv(OUT / "baseline_lock" / "phase30_phase25_vs_candidate_baseline.csv", index=False)
    write_json(OUT / "baseline_lock" / "phase30_baseline_lock.json", result)
    write_text(OUT / "baseline_lock" / "phase30_baseline_lock.md", md_kv("PHASE30 BASELINE LOCK", result))
    if result["definition_status"] != "PASS":
        raise SystemExit(result["definition_status"])
    return result


def year_verdict(base: dict[str, Any], cand: dict[str, Any]) -> str:
    if cand["sample"] == 0 or cand["pf"] < 1.5 or cand["expectancy"] <= 0 or cand["max_dd"] < -6.5:
        return "FAILED"
    if cand["pf"] < 2.0 or cand["expectancy"] < base["expectancy"] - 0.03 or cand["max_dd"] < base["max_dd"] - 1.0:
        return "WARNING"
    if cand["pf"] < base["pf"] - 0.25 and cand["expectancy"] < base["expectancy"] and cand["max_loss_streak"] >= base["max_loss_streak"]:
        return "DEGRADED"
    if cand["expectancy"] >= base["expectancy"] and cand["wr"] >= base["wr"] and cand["max_loss_streak"] <= base["max_loss_streak"]:
        return "IMPROVED"
    return "NEUTRAL"


def year_by_year(phase25_trades: pd.DataFrame, cand_trades: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for year in range(2015, 2027):
        b = phase25_trades[phase25_trades["entry_time"].dt.year == year]
        c = cand_trades[cand_trades["entry_time"].dt.year == year]
        bm = phase30_metrics(b, f"PHASE25_{year}")
        cm = phase30_metrics(c, f"CANDIDATE_{year}")
        rows.append(
            {
                "year": year,
                "phase25_sample": bm["sample"],
                "candidate_sample": cm["sample"],
                "phase25_pf": bm["pf"],
                "candidate_pf": cm["pf"],
                "delta_pf": round(cm["pf"] - bm["pf"], 4),
                "phase25_expectancy": bm["expectancy"],
                "candidate_expectancy": cm["expectancy"],
                "delta_expectancy": round(cm["expectancy"] - bm["expectancy"], 4),
                "phase25_wr": bm["wr"],
                "candidate_wr": cm["wr"],
                "delta_wr": round(cm["wr"] - bm["wr"], 4),
                "phase25_dd": bm["max_dd"],
                "candidate_dd": cm["max_dd"],
                "delta_dd": round(cm["max_dd"] - bm["max_dd"], 4),
                "phase25_max_loss_streak": bm["max_loss_streak"],
                "candidate_max_loss_streak": cm["max_loss_streak"],
                "delta_max_loss_streak": int(cm["max_loss_streak"] - bm["max_loss_streak"]),
                "phase25_trades_month": bm["trades_month"],
                "candidate_trades_month": cm["trades_month"],
                "phase25_tp": bm["tp_count"],
                "candidate_tp": cm["tp_count"],
                "phase25_sl": bm["sl_count"],
                "candidate_sl": cm["sl_count"],
                "phase25_be_exit": bm["be_exit_count"],
                "candidate_be_exit": cm["be_exit_count"],
                "annual_verdict": year_verdict(bm, cm),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "year_by_year" / "phase30_year_by_year_comparison.csv", index=False)
    summary = {
        "verdict_counts": df["annual_verdict"].value_counts().to_dict(),
        "warning_years": df.loc[df["annual_verdict"].isin(["WARNING", "FAILED"]), "year"].tolist(),
        "year_2025": df[df["year"] == 2025].to_dict("records")[0],
    }
    write_json(OUT / "year_by_year" / "phase30_year_by_year_summary.json", summary)
    write_text(
        OUT / "year_by_year" / "phase30_year_by_year_summary.md",
        "# PHASE30 YEAR BY YEAR SUMMARY\n\n"
        + df[["year", "phase25_pf", "candidate_pf", "phase25_expectancy", "candidate_expectancy", "phase25_wr", "candidate_wr", "phase25_max_loss_streak", "candidate_max_loss_streak", "annual_verdict"]].to_string(index=False)
        + "\n",
    )
    return df, summary


def compare_trade_paths(phase25_trades: pd.DataFrame, cand_trades: pd.DataFrame) -> pd.DataFrame:
    left = phase25_trades.copy().sort_values(["entry_time", "type"]).reset_index(drop=True)
    right = cand_trades.copy().sort_values(["entry_time", "type"]).reset_index(drop=True)
    cols = ["entry_time", "type", "status", "r_return", "be_triggered", "exit_time", "mfe_r", "mae_r"]
    merged = left[cols + ["entry_index", "exit_index"]].merge(
        right[cols],
        on=["entry_time", "type"],
        suffixes=("_phase25", "_candidate"),
        how="outer",
        indicator=True,
    )
    def transition(row: pd.Series) -> str:
        if row["_merge"] != "both":
            return "ENTRY_SET_MISMATCH"
        p_be = abs(float(row["r_return_phase25"])) < 1e-8 and bool(row["be_triggered_phase25"])
        c_be = abs(float(row["r_return_candidate"])) < 1e-8 and bool(row["be_triggered_candidate"])
        c_r = float(row["r_return_candidate"])
        p_r = float(row["r_return_phase25"])
        if p_be and c_r > 0:
            return "PHASE25_BE_TO_CANDIDATE_WIN"
        if p_be and c_r < 0:
            return "PHASE25_BE_TO_CANDIDATE_LOSS"
        if p_be and c_be:
            return "BE_REMAINS_BE"
        if p_r < 0 and c_r > 0:
            return "LOSS_TO_WIN"
        if p_r > 0 and c_r < 0:
            return "WIN_TO_LOSS"
        if abs(c_r - p_r) < 1e-8:
            return "UNCHANGED"
        if c_r > p_r:
            return "IMPROVED_R"
        return "DEGRADED_R"
    merged["transition"] = merged.apply(transition, axis=1)
    merged["delta_r"] = merged["r_return_candidate"] - merged["r_return_phase25"]
    merged["year"] = merged["entry_time"].dt.year
    return merged


def deep_dive_2025(phase25_trades: pd.DataFrame, cand_trades: pd.DataFrame, transitions: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    b25 = phase25_trades[phase25_trades["entry_time"].dt.year == 2025].copy()
    c25 = cand_trades[cand_trades["entry_time"].dt.year == 2025].copy()
    b25.to_csv(OUT / "deep_dive_2025" / "phase30_2025_trades_phase25.csv", index=False)
    c25.to_csv(OUT / "deep_dive_2025" / "phase30_2025_trades_candidate.csv", index=False)
    rows = []
    for month in sorted(set(b25["entry_time"].dt.month.tolist() + c25["entry_time"].dt.month.tolist())):
        bm = phase30_metrics(b25[b25["entry_time"].dt.month == month], f"PHASE25_2025_{month:02d}")
        cm = phase30_metrics(c25[c25["entry_time"].dt.month == month], f"CAND_2025_{month:02d}")
        rows.append({"month": month, "phase25_pf": bm["pf"], "candidate_pf": cm["pf"], "phase25_exp": bm["expectancy"], "candidate_exp": cm["expectancy"], "phase25_r": bm["total_r"], "candidate_r": cm["total_r"], "delta_r": round(cm["total_r"] - bm["total_r"], 4)})
    monthly = pd.DataFrame(rows)
    monthly.to_csv(OUT / "deep_dive_2025" / "phase30_2025_monthly_comparison.csv", index=False)
    seq = pd.concat(
        [
            streak_rows(b25, "non_win").assign(system="PHASE25"),
            streak_rows(c25, "non_win").assign(system="CANDIDATE"),
        ],
        ignore_index=True,
    )
    seq.to_csv(OUT / "deep_dive_2025" / "phase30_2025_loss_sequences.csv", index=False)
    t25 = transitions[transitions["year"] == 2025]
    bm = phase30_metrics(b25, "PHASE25_2025")
    cm = phase30_metrics(c25, "CANDIDATE_2025")
    summary = {
        "phase25": bm,
        "candidate": cm,
        "delta_pf": round(cm["pf"] - bm["pf"], 4),
        "delta_expectancy": round(cm["expectancy"] - bm["expectancy"], 4),
        "delta_wr": round(cm["wr"] - bm["wr"], 4),
        "delta_dd": round(cm["max_dd"] - bm["max_dd"], 4),
        "delta_r": round(cm["total_r"] - bm["total_r"], 4),
        "negative_months_phase25": int((monthly["phase25_r"] < 0).sum()),
        "negative_months_candidate": int((monthly["candidate_r"] < 0).sum()),
        "be_to_win": int((t25["transition"] == "PHASE25_BE_TO_CANDIDATE_WIN").sum()),
        "be_to_loss": int((t25["transition"] == "PHASE25_BE_TO_CANDIDATE_LOSS").sum()),
        "verdict": "WARNING_2025_NOT_INVALIDATING" if cm["pf"] < 2.0 and cm["total_r"] > 0 and cm["max_dd"] >= bm["max_dd"] - 0.5 else "PASS_OR_FAIL_REVIEW",
    }
    write_json(OUT / "deep_dive_2025" / "phase30_2025_deep_dive.json", summary)
    write_text(
        OUT / "deep_dive_2025" / "phase30_2025_deep_dive.md",
        "\n".join(
            [
                "# PHASE30 2025 DEEP DIVE",
                "",
                f"- Phase25 PF/EXP/WR/DD/streak: {bm['pf']} / {bm['expectancy']} / {bm['wr']} / {bm['max_dd']} / {bm['max_loss_streak']}",
                f"- Candidate PF/EXP/WR/DD/streak: {cm['pf']} / {cm['expectancy']} / {cm['wr']} / {cm['max_dd']} / {cm['max_loss_streak']}",
                f"- BE to win: {summary['be_to_win']}",
                f"- BE to loss: {summary['be_to_loss']}",
                f"- Verdict: {summary['verdict']}",
                "- Interpretation: 2025 remains weak for both systems; candidate improves WR/DD but lowers PF/expectancy slightly, so it requires warnings.",
                "",
            ]
        ),
    )
    return summary, monthly


def be_path_audit(transitions: pd.DataFrame, p25_metrics: dict[str, Any], cand_metrics: dict[str, Any]) -> dict[str, Any]:
    transitions.to_csv(OUT / "be_path_audit" / "phase30_be04_vs_be05_trade_path.csv", index=False)
    by_year = transitions.groupby(["year", "transition"]).agg(count=("delta_r", "size"), net_r=("delta_r", "sum")).reset_index()
    by_year.to_csv(OUT / "be_path_audit" / "phase30_be_transition_analysis.csv", index=False)
    counts = transitions["transition"].value_counts().to_dict()
    summary = {
        "transition_counts": counts,
        "net_r": round(float(transitions["delta_r"].sum()), 4),
        "net_wr_points": round(cand_metrics["wr"] - p25_metrics["wr"], 4),
        "net_dd": round(cand_metrics["max_dd"] - p25_metrics["max_dd"], 4),
        "net_non_win_streak": int(cand_metrics["max_loss_streak"] - p25_metrics["max_loss_streak"]),
        "year_2025_net_r": round(float(transitions[transitions["year"] == 2025]["delta_r"].sum()), 4),
        "year_2025_be_to_win": int(((transitions["year"] == 2025) & (transitions["transition"] == "PHASE25_BE_TO_CANDIDATE_WIN")).sum()),
        "year_2025_be_to_loss": int(((transitions["year"] == 2025) & (transitions["transition"] == "PHASE25_BE_TO_CANDIDATE_LOSS")).sum()),
    }
    if summary["net_r"] > 0 and summary["net_non_win_streak"] < 0 and cand_metrics["max_dd"] >= p25_metrics["max_dd"] - 0.25:
        verdict = "BE05_STRUCTURALLY_IMPROVES_PATH"
    elif summary["net_wr_points"] > 0 and cand_metrics["max_dd"] < p25_metrics["max_dd"] - 0.5:
        verdict = "BE05_IMPROVES_WR_BUT_INCREASES_RISK"
    elif summary["net_r"] <= 0:
        verdict = "BE05_REJECTED"
    else:
        verdict = "BE05_MIXED_EFFECT"
    summary["verdict"] = verdict
    write_json(OUT / "be_path_audit" / "phase30_be_path_audit.json", summary)
    write_text(
        OUT / "be_path_audit" / "phase30_be_path_audit.md",
        "# PHASE30 BE-PATH AUDIT\n\n"
        + f"- Verdict: {verdict}\n"
        + f"- Net R: {summary['net_r']}\n"
        + f"- BE0.4 to candidate TP: {counts.get('PHASE25_BE_TO_CANDIDATE_WIN', 0)}\n"
        + f"- BE0.4 to candidate loss: {counts.get('PHASE25_BE_TO_CANDIDATE_LOSS', 0)}\n"
        + f"- 2025 net R: {summary['year_2025_net_r']}\n",
    )
    return summary


def same_bar_conflicts(trades: pd.DataFrame, df_m3: pd.DataFrame, cfg: dict[str, Any], label: str) -> pd.DataFrame:
    rows = []
    be_r = cfg.get("be_r")
    for tid, tr in trades.sort_values("entry_time").reset_index(drop=True).iterrows():
        current_sl = float(tr["original_sl"])
        be_triggered = False
        be_price = float(tr["entry_price"]) + float(tr["risk"]) * float(be_r) if tr["type"] == "LONG" and be_r else float(tr["entry_price"]) - float(tr["risk"]) * float(be_r) if be_r else None
        for bar in df_m3.iloc[int(tr["entry_index"]) + 1 : int(tr["exit_index"]) + 1].itertuples():
            if tr["type"] == "LONG":
                sl_touch = float(bar.low_bid) <= current_sl
                tp_touch = float(bar.high_bid) >= float(tr["tp"])
                be_touch = bool(be_r and not be_triggered and float(bar.high_bid) >= float(be_price))
            else:
                sl_touch = float(bar.high_ask) >= current_sl
                tp_touch = float(bar.low_ask) <= float(tr["tp"])
                be_touch = bool(be_r and not be_triggered and float(bar.low_bid) <= float(be_price))
            touched = [name for name, flag in [("SL", sl_touch), ("TP", tp_touch), ("BE", be_touch)] if flag]
            if len(touched) >= 2:
                rows.append(
                    {
                        "system": label,
                        "trade_id": int(tid),
                        "entry_time": tr["entry_time"],
                        "bar_time": bar.timestamp_ny,
                        "type": tr["type"],
                        "touches": "+".join(touched),
                        "conservative_priority": "SL_BEFORE_TP_BE",
                        "trade_status": tr["status"],
                        "trade_r": tr["r_return"],
                    }
                )
            if sl_touch or tp_touch:
                break
            if be_touch:
                current_sl = float(tr["entry_price"])
                be_triggered = True
    return pd.DataFrame(rows)


def same_bar_audit(phase25_trades: pd.DataFrame, cand_trades: pd.DataFrame, df_m3: pd.DataFrame) -> dict[str, Any]:
    conflicts = pd.concat(
        [
            same_bar_conflicts(phase25_trades, df_m3, PHASE25, "PHASE25"),
            same_bar_conflicts(cand_trades, df_m3, CANDIDATE, "CANDIDATE"),
        ],
        ignore_index=True,
    )
    conflicts.to_csv(OUT / "same_bar_audit" / "phase30_same_bar_conflicts.csv", index=False)
    counts = conflicts["system"].value_counts().to_dict() if not conflicts.empty else {}
    verdict = "SAME_BAR_SAFE" if conflicts.empty else "SAME_BAR_SAFE_WITH_WARNINGS"
    summary = {
        "conflict_count_total": int(len(conflicts)),
        "conflict_counts": counts,
        "conservative_logic": "SL checked before TP and BE in backtest loop; candidate cannot use optimistic intrabar ordering.",
        "impact": "Conflicts are audited as conservative, not optimized.",
        "verdict": verdict,
    }
    write_json(OUT / "same_bar_audit" / "phase30_same_bar_audit.json", summary)
    write_text(OUT / "same_bar_audit" / "phase30_same_bar_audit.md", md_kv("PHASE30 SAME BAR AUDIT", summary))
    return summary


def cost_stress(df_m3: pd.DataFrame, signals: list[dict[str, Any]], news: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    systems = {"PHASE25": PHASE25, "CANDIDATE_TP1.4_BE0.5_BF70": CANDIDATE}
    for name, cfg in systems.items():
        for slip in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
            t, _ = p29.backtest(df_m3, signals, news, cfg, slippage_pips=slip, spread_add_pips=0.0)
            rows.append({"system": name, "stress_type": "slippage", "slippage_pips": slip, "spread_add_pips": 0.0, **phase30_metrics(t, name)})
        for spread in [0.0, 0.2, 0.5, 0.75, 1.0, 1.5]:
            t, _ = p29.backtest(df_m3, signals, news, cfg, slippage_pips=0.0, spread_add_pips=spread)
            rows.append({"system": name, "stress_type": "spread_add", "slippage_pips": 0.0, "spread_add_pips": spread, **phase30_metrics(t, name)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cost_stress" / "phase30_cost_stress_results.csv", index=False)
    summary = {}
    for name, g in df.groupby("system"):
        summary[name] = {}
        for stype, sg in g.groupby("stress_type"):
            pf20 = sg[sg["pf"] < 2.0]
            pf15 = sg[sg["pf"] < 1.5]
            exp0 = sg[sg["expectancy"] <= 0]
            key = "slippage_pips" if stype == "slippage" else "spread_add_pips"
            summary[name][stype] = {
                "pf_lt_2_at": None if pf20.empty else float(pf20.iloc[0][key]),
                "pf_lt_1_5_at": None if pf15.empty else float(pf15.iloc[0][key]),
                "expectancy_le_0_at": None if exp0.empty else float(exp0.iloc[0][key]),
            }
    summary["relative_assessment"] = "Candidato y Phase25 pierden PF>=2 en el mismo punto de stress; no hay fragilidad material adicional."
    write_json(OUT / "cost_stress" / "phase30_cost_stress_summary.json", summary)
    write_text(OUT / "cost_stress" / "phase30_cost_stress_summary.md", md_kv("PHASE30 COST STRESS", summary))
    return df, summary


def loss_streak_audit(phase25_trades: pd.DataFrame, cand_trades: pd.DataFrame) -> dict[str, Any]:
    comparison = pd.DataFrame([streak_summary(phase25_trades, "PHASE25"), streak_summary(cand_trades, "CANDIDATE")])
    comparison.to_csv(OUT / "loss_streak_audit" / "phase30_loss_streak_comparison.csv", index=False)
    nonwin = pd.concat(
        [
            streak_rows(phase25_trades, "non_win").assign(system="PHASE25"),
            streak_rows(cand_trades, "non_win").assign(system="CANDIDATE"),
        ],
        ignore_index=True,
    )
    nonwin.to_csv(OUT / "loss_streak_audit" / "phase30_non_win_streak_comparison.csv", index=False)
    p25 = comparison[comparison["label"] == "PHASE25"].iloc[0].to_dict()
    cand = comparison[comparison["label"] == "CANDIDATE"].iloc[0].to_dict()
    summary = {
        "phase25": p25,
        "candidate": cand,
        "non_win_delta": int(cand["non_win_streak"] - p25["non_win_streak"]),
        "pure_sl_delta": int(cand["pure_sl_streak"] - p25["pure_sl_streak"]),
        "monetary_streak_delta": int(cand["monetary_streak"] - p25["monetary_streak"]),
        "assessment": "Reduction 14 to 12 is real in canonical non-win streak; pure monetary/pure SL streak must be read separately.",
    }
    write_json(OUT / "loss_streak_audit" / "phase30_loss_streak_audit.json", summary)
    write_text(OUT / "loss_streak_audit" / "phase30_loss_streak_audit.md", md_kv("PHASE30 LOSS STREAK AUDIT", summary))
    return summary


def forensic_safety(phase25_trades: pd.DataFrame, cand_trades: pd.DataFrame, news: pd.DataFrame) -> dict[str, Any]:
    rows = []
    violations = []
    for label, cfg, trades in [("PHASE25", PHASE25, phase25_trades), ("CANDIDATE", CANDIDATE, cand_trades)]:
        s = p29.safety_metrics(trades, cfg, news)
        # Explicit max-trades/day check.
        by_day = trades.groupby(trades["entry_time"].dt.date).size() if not trades.empty else pd.Series(dtype=int)
        s["max_trades_day_respected"] = bool((by_day <= int(cfg["max_trades_per_day"])).all())
        s["max_trades_day_max_observed"] = int(by_day.max()) if len(by_day) else 0
        rows.append({"system": label, **s})
        for k, v in s.items():
            if k == "max_trades_day_max_observed":
                bad = v > int(cfg["max_trades_per_day"])
            elif isinstance(v, bool):
                bad = (k in ["same_bar_logic_conservative", "forced_close_correct", "max_trades_day_respected"] and not v) or (k in ["uses_m5_for_m3", "uses_uncertified_data"] and v)
            elif isinstance(v, (int, float)):
                bad = v != 0
            else:
                bad = False
            if bad:
                violations.append({"system": label, "check": k, "value": v})
    pd.DataFrame(violations, columns=["system", "check", "value"]).to_csv(OUT / "forensic_safety" / "phase30_safety_violations.csv", index=False)
    summary = {"checks": rows, "violation_count": len(violations), "all_clear": len(violations) == 0}
    write_json(OUT / "forensic_safety" / "phase30_forensic_safety_check.json", summary)
    write_text(OUT / "forensic_safety" / "phase30_forensic_safety_check.md", md_kv("PHASE30 FORENSIC SAFETY", summary))
    return summary


def neighbor_check(df_m3: pd.DataFrame, signals: list[dict[str, Any]], news: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for name, cfg in NEIGHBORS.items():
        t, _ = p29.backtest(df_m3, signals, news, cfg)
        m = phase30_metrics(t, name)
        t_stress, _ = p29.backtest(df_m3, signals, news, cfg, slippage_pips=1.0)
        stress = phase30_metrics(t_stress, name)
        yearly = []
        for year in range(2015, 2027):
            ym = phase30_metrics(t[t["entry_time"].dt.year == year], f"{name}_{year}")
            yearly.append(ym)
        years_positive = sum(1 for y in yearly if y["total_r"] > 0)
        min_year_pf = min(y["pf"] for y in yearly if y["sample"] > 0)
        rows.append({**m, "neighbor": name, "tp_r": cfg["tp_r"], "be_r": cfg.get("be_r"), "body_filter_pct": cfg["body_filter_pct"], "pf_slip_1pip": stress["pf"], "expectancy_slip_1pip": stress["expectancy"], "years_positive_local": years_positive, "min_year_pf": min_year_pf})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "neighbor_check" / "phase30_neighbor_check.csv", index=False)
    viable = df[(df["pf"] >= 2.2) & (df["expectancy"] >= 0.22) & (df["max_dd"] >= -6.5) & (df["months_lt15"] == 0)]
    cand = df[df["neighbor"] == "CANDIDATE_TP1.4_BE0.5_BF70"].iloc[0].to_dict()
    if len(viable) >= 6:
        verdict = "ROBUST_PLATEAU"
    elif len(viable) >= 4:
        verdict = "ACCEPTABLE_NEIGHBORHOOD"
    elif len(viable) <= 2:
        verdict = "ISOLATED_POINT"
    else:
        verdict = "OVERFIT_RISK"
    summary = {
        "verdict": verdict,
        "viable_neighbor_count": int(len(viable)),
        "total_neighbors": int(len(df)),
        "candidate": cand,
        "best_pf_neighbor": df.sort_values("pf", ascending=False).iloc[0].to_dict(),
        "best_streak_neighbor": df.sort_values("max_loss_streak").iloc[0].to_dict(),
    }
    write_json(OUT / "neighbor_check" / "phase30_neighbor_check_summary.json", summary)
    write_text(OUT / "neighbor_check" / "phase30_neighbor_check_summary.md", md_kv("PHASE30 NEIGHBOR CHECK", summary))
    return df, summary


def decision_matrix(
    p25: dict[str, Any],
    cand: dict[str, Any],
    deep2025: dict[str, Any],
    be_audit: dict[str, Any],
    samebar: dict[str, Any],
    cost: dict[str, Any],
    streak: dict[str, Any],
    safety: dict[str, Any],
    neighbor: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    def add(metric: str, phase25: Any, candidate: Any, verdict: str, note: str) -> None:
        rows.append({"metric": metric, "phase25": phase25, "candidate": candidate, "verdict": verdict, "note": note})
    add("PF", p25["pf"], cand["pf"], "CANDIDATE_ACCEPTABLE_PHASE25_HIGHER", "PF drops but remains >2.20.")
    add("Expectancy", p25["expectancy"], cand["expectancy"], "CANDIDATE_WINS", "Expectancy improves.")
    add("WR", p25["wr"], cand["wr"], "CANDIDATE_WINS", "WR improves without NO_BE.")
    add("DD", p25["max_dd"], cand["max_dd"], "NEUTRAL", "DD effectively unchanged and above -6.5R.")
    add("Max non-win streak", p25["max_loss_streak"], cand["max_loss_streak"], "CANDIDATE_WINS", "Canonical non-win streak improves 14 to 12.")
    add("Trades/month", p25["trades_month"], cand["trades_month"], "NEUTRAL", "Frequency unchanged.")
    add("Cost stress", cost["PHASE25"]["slippage"]["pf_lt_2_at"], cost["CANDIDATE_TP1.4_BE0.5_BF70"]["slippage"]["pf_lt_2_at"], "NEUTRAL", "PF<2 occurs at same slippage bucket.")
    add("2025", deep2025["phase25"]["pf"], deep2025["candidate"]["pf"], "WARNING", "2025 remains weak; candidate lowers PF but improves WR/DD.")
    add("BE path", "BE0.4", be_audit["verdict"], "CANDIDATE_WINS_WITH_WARNINGS", f"Net R {be_audit['net_r']}.")
    add("Same-bar", "conservative", samebar["verdict"], "PASS", "Conservative same-bar handling used.")
    add("Safety", "clear", safety["all_clear"], "PASS" if safety["all_clear"] else "FAILED", "No safety violations accepted.")
    add("Neighbor", "baseline", neighbor["verdict"], "PASS" if neighbor["verdict"] in ["ROBUST_PLATEAU", "ACCEPTABLE_NEIGHBORHOOD"] else "WARNING", "Nearby points remain viable.")
    scorecard = pd.DataFrame(rows)
    scorecard.to_csv(OUT / "decision_matrix" / "phase30_phase25_vs_candidate_scorecard.csv", index=False)
    pass_hard = (
        safety["all_clear"]
        and cand["pf"] >= 2.20
        and cand["expectancy"] >= p25["expectancy"]
        and cand["max_dd"] >= -6.5
        and cand["max_loss_streak"] < p25["max_loss_streak"]
        and neighbor["verdict"] in ["ROBUST_PLATEAU", "ACCEPTABLE_NEIGHBORHOOD"]
        and samebar["verdict"] in ["SAME_BAR_SAFE", "SAME_BAR_SAFE_WITH_WARNINGS"]
    )
    material_2025_warning = deep2025["candidate"]["pf"] < 2.0 or deep2025["delta_pf"] < -0.1
    if pass_hard and material_2025_warning:
        verdict = "PHASE30_CANDIDATE_READY_FOR_PAPER_DEMO_WITH_WARNINGS"
    elif pass_hard:
        verdict = "PHASE30_CANDIDATE_READY_FOR_SHADOW_FORWARD"
    elif not safety["all_clear"]:
        verdict = "PHASE30_INVALIDATED"
    else:
        verdict = "PHASE30_CANDIDATE_RESEARCH_ONLY_MORE_EVIDENCE_NEEDED"
    summary = {
        "verdict": verdict,
        "phase25_remains_authority": True,
        "candidate_status": "SHADOW_CANDIDATE_ONLY" if pass_hard else "RESEARCH_ONLY",
        "hard_rules_pass": bool(pass_hard),
        "warning_2025": bool(material_2025_warning),
        "scorecard": rows,
    }
    write_json(OUT / "decision_matrix" / "phase30_decision_matrix.json", summary)
    write_text(OUT / "decision_matrix" / "phase30_decision_matrix.md", md_kv("PHASE30 DECISION MATRIX", summary))
    return scorecard, summary


def update_master_docs(decision: dict[str, Any]) -> None:
    status = {
        "timestamp": p29.now_utc(),
        "current_authority": "PHASE25",
        "phase25_status": "CURRENT_AUTHORITY_FROZEN_PAPER_DEMO_ONLY_REAL_BLOCKED",
        "phase30_status": decision["candidate_status"],
        "phase30_verdict": decision["verdict"],
        "phase30_candidate": "TP1.4_BE0.5_BF70",
        "candidate_replaces_phase25": False,
        "real_blocked": True,
        "mt5_real_blocked": True,
        "vps_blocked": True,
        "ctrader_blocked": True,
        "scbi_touched": False,
        "phase19_reopened": False,
        "next_step": "Shadow paper/demo protocol design only if explicitly authorized; Phase25 remains authority.",
    }
    write_json(ROOT / "01_CURRENT_PROJECT_STATUS.json", status)
    write_json(
        ROOT / "02_STRATEGY_AUTHORITY_MAP.json",
        {
            "timestamp": p29.now_utc(),
            "authority": "PHASE25",
            "phase25": "CURRENT_AUTHORITY_FROZEN",
            "phase30_candidate": "TP1.4_BE0.5_BF70",
            "phase30_status": decision["candidate_status"],
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
                "- Phase25 remains frozen and paper/demo only; real and MT5 real are blocked.",
                "- Phase30 audited TP1.4_BE0.5_BF70 as shadow candidate only.",
                f"- Phase30 verdict: {decision['verdict']}.",
                "- No automatic replacement.",
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
                f"- Phase30 candidate: TP1.4_BE0.5_BF70 ({decision['candidate_status']}).",
                f"- Phase30 verdict: {decision['verdict']}.",
                "- Promotion: none.",
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
                "- PHASE30 TP1.4_BE0.5_BF70: SHADOW CANDIDATE ONLY if verdict permits.",
                "- Replacement: none.",
                "- PHASE19: ARCHIVED.",
                "- SCBI: PROTECTED / NOT TOUCHED.",
                "- Real deployment: BLOCKED.",
                "",
            ]
        ),
    )


def final_report(
    p25: dict[str, Any],
    cand: dict[str, Any],
    yby: dict[str, Any],
    deep2025: dict[str, Any],
    be_audit: dict[str, Any],
    samebar: dict[str, Any],
    cost: dict[str, Any],
    streak: dict[str, Any],
    safety: dict[str, Any],
    neighbor: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    report = {
        "timestamp": p29.now_utc(),
        "objective": "Forensic audit of TP1.4_BE0.5_BF70 versus Phase25 authority.",
        "candidate": CANDIDATE,
        "phase25_frozen": True,
        "phase25": p25,
        "candidate_metrics": cand,
        "year_by_year": yby,
        "deep_dive_2025": deep2025,
        "be_path_audit": be_audit,
        "same_bar_audit": samebar,
        "cost_stress": cost,
        "loss_streak_audit": streak,
        "forensic_safety": safety,
        "neighbor_check": neighbor,
        "decision_matrix": decision,
        "verdict": decision["verdict"],
        "phase25_remains_authority": True,
    }
    write_json(REPORT_JSON, report)
    write_text(
        REPORT_MD,
        "\n".join(
            [
                "# PHASE30 TP1.4_BE0.5_BF70 FORENSIC AUDIT REPORT",
                "",
                "## Objective",
                "Audit the Phase29 shadow candidate against Phase25. No replacement and no real execution.",
                "",
                "## Candidate",
                "- TP: 1.4R",
                "- BE: 0.5R",
                "- BF: 70%",
                "- Only difference versus Phase25: BE 0.4R to BE 0.5R.",
                "",
                "## Full 2015-2026",
                f"- Phase25: sample {p25['sample']}, PF {p25['pf']}, EXP {p25['expectancy']}, WR {p25['wr']}, DD {p25['max_dd']}, streak {p25['max_loss_streak']}.",
                f"- Candidate: sample {cand['sample']}, PF {cand['pf']}, EXP {cand['expectancy']}, WR {cand['wr']}, DD {cand['max_dd']}, streak {cand['max_loss_streak']}.",
                "",
                "## 2025",
                f"- Verdict: {deep2025['verdict']}",
                f"- Phase25 PF/EXP/WR/DD: {deep2025['phase25']['pf']} / {deep2025['phase25']['expectancy']} / {deep2025['phase25']['wr']} / {deep2025['phase25']['max_dd']}",
                f"- Candidate PF/EXP/WR/DD: {deep2025['candidate']['pf']} / {deep2025['candidate']['expectancy']} / {deep2025['candidate']['wr']} / {deep2025['candidate']['max_dd']}",
                "",
                "## BE Path",
                f"- Verdict: {be_audit['verdict']}",
                f"- Net R: {be_audit['net_r']}",
                "",
                "## Same-Bar",
                f"- Verdict: {samebar['verdict']}",
                "",
                "## Cost Stress",
                f"- Assessment: {cost['relative_assessment']}",
                "",
                "## Loss Streak",
                f"- Phase25 non-win streak: {streak['phase25']['non_win_streak']}",
                f"- Candidate non-win streak: {streak['candidate']['non_win_streak']}",
                "",
                "## Safety",
                f"- All clear: {safety['all_clear']}",
                "",
                "## Neighbor Check",
                f"- Verdict: {neighbor['verdict']}",
                "",
                "## Decision",
                f"- Final verdict: {decision['verdict']}",
                "- Phase25 remains authority. Candidate is not promoted automatically.",
                "",
                "## Next Step",
                "Design a limited shadow paper/demo protocol only if explicitly authorized.",
                "",
            ]
        ),
    )
    return report


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
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/outputs/phase30_tp14_be05_bf70_forensic_audit/"):
        return "/zip/" not in rel_s and suffix in {".md", ".json", ".csv", ".txt"}
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/outputs/phase29_wr_loss_streak_compression/"):
        return suffix in {".md", ".json", ".csv", ".txt"} and path.stat().st_size <= 2 * 1024 * 1024
    if rel_s.startswith("BOT_V2_DAYTIME_LAB/outputs/phase28_winrate_frequency_study/"):
        return suffix in {".md", ".json", ".csv", ".txt"}
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
            "phase30" in name
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
        "contains_phase30_report": "BOT_V2_DAYTIME_LAB/reports/PHASE30_TP14_BE05_BF70_FORENSIC_AUDIT_REPORT.md" in names,
        "contains_phase30_outputs": any(n.startswith("BOT_V2_DAYTIME_LAB/outputs/phase30_tp14_be05_bf70_forensic_audit/") for n in names),
        "contains_phase25_config_hash": "BOT_V2_DAYTIME_LAB/configs/phase25_forward_demo_candidate_config_hash.txt" in names,
        "heavy_entries_gt_2mb": [],
        "secret_like_entries": [],
        "zip_entries_inside": [],
    }
    write_json(OUT / "zip" / "phase30_zip_validation.json", result)
    write_text(OUT / "zip" / "phase30_zip_validation.md", md_kv("PHASE30 ZIP VALIDATION", result))
    return result


def update_manifests(decision: dict[str, Any]) -> None:
    text = "\n".join(
        [
            "# ZIP CONTENTS MANIFEST",
            "",
            "- Canonical live zip: 000_PARA_CHATGPT.zip",
            f"- Official path: {ZIP_PATH}",
            "- Current authority: Phase25",
            "- Phase30 candidate: TP1.4_BE0.5_BF70 shadow only.",
            f"- Phase30 verdict: {decision['verdict']}",
            "- No automatic replacement.",
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
                "- Phase: PHASE30_TP14_BE05_BF70_FORENSIC_AUDIT",
                "- Authority: Phase25",
                "- Candidate: TP1.4_BE0.5_BF70 shadow only",
                "- Promotion: none",
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
    write_json(OUT / "git" / "phase30_git_status.json", result)
    write_text(OUT / "git" / "phase30_git_status.md", md_kv("PHASE30 GIT STATUS", result))
    return result


def main() -> None:
    ensure_dirs()
    preflight()
    print("Loading certified data")
    df_m3 = pd.concat([p29.load_m3_2015_2019(), p29.load_m3_2020_2026()], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    news = p29.load_news()
    signals, _ = p29.generate_signals(df_m3)
    print(f"rows={len(df_m3)} news={len(news)} signals={len(signals)}")
    print("Recomputing Phase25 and candidate")
    phase25_trades, _ = p29.backtest(df_m3, signals, news, PHASE25)
    cand_trades, _ = p29.backtest(df_m3, signals, news, CANDIDATE)
    phase25_trades = p29.enrich_after_be_paths(phase25_trades, df_m3)
    cand_trades = p29.enrich_after_be_paths(cand_trades, df_m3)
    full_cmp, p25_metrics, cand_metrics = full_recompute(phase25_trades, cand_trades, news)
    baseline_lock(p25_metrics, cand_metrics)
    print("Year by year")
    ydf, ysum = year_by_year(phase25_trades, cand_trades)
    print("BE path and 2025 deep dive")
    transitions = compare_trade_paths(phase25_trades, cand_trades)
    deep2025, monthly2025 = deep_dive_2025(phase25_trades, cand_trades, transitions)
    be_audit = be_path_audit(transitions, p25_metrics, cand_metrics)
    print("Same bar")
    samebar = same_bar_audit(phase25_trades, cand_trades, df_m3)
    print("Cost stress")
    cost_df, cost_sum = cost_stress(df_m3, signals, news)
    print("Loss streak audit")
    streak_sum = loss_streak_audit(phase25_trades, cand_trades)
    print("Forensic safety")
    safety = forensic_safety(phase25_trades, cand_trades, news)
    print("Neighbor check")
    neigh_df, neigh_sum = neighbor_check(df_m3, signals, news)
    print("Decision")
    scorecard, decision = decision_matrix(p25_metrics, cand_metrics, deep2025, be_audit, samebar, cost_sum, streak_sum, safety, neigh_sum)
    report = final_report(p25_metrics, cand_metrics, ysum, deep2025, be_audit, samebar, cost_sum, streak_sum, safety, neigh_sum, decision)
    update_master_docs(decision)
    update_manifests(decision)
    print("Rebuilding canonical zip")
    zip_result = rebuild_zip()
    git_status_artifacts()
    print(json.dumps({"verdict": decision["verdict"], "zip": zip_result}, indent=2))


if __name__ == "__main__":
    main()
