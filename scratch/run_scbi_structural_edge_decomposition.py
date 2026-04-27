"""
SCBI_M5_GLOBAL Structural Edge Decomposition

Reads the canonical trades_baseline.csv from the 2020-2025 durability run
and performs ablation + timing decomposition WITHOUT re-running the backtest.
No parameters are changed. No optimization. Pure segmentation of existing evidence.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
TRADES_FILE = ROOT / "results" / "SCBI_2020_2025_DURABILITY" / "trades_baseline.csv"
OUTPUT_FILE = ROOT / "results" / "SCBI_2020_2025_DURABILITY" / "structural_edge_decomposition.json"

STRESS_DELTA_PIPS = 0.9
PIP_SIZE = 0.0001


def compute_metrics(df: pd.DataFrame, pnl_col: str = "pnl_r") -> dict:
    if df.empty:
        return {"N": 0, "wins": 0, "losses": 0, "pf": 0.0, "expectancy": 0.0,
                "max_drawdown": 0.0, "win_rate": 0.0, "total_r": 0.0}
    pnls = df[pnl_col].values
    wins = int((pnls > 0).sum())
    losses = int(len(pnls) - wins)
    gross_profit = float(pnls[pnls > 0].sum())
    gross_loss = float(abs(pnls[pnls <= 0].sum()))
    pf = gross_profit / gross_loss if gross_loss > 0 else 999.0
    total_r = float(pnls.sum())
    equity = 0.0
    peak = 0.0
    dd = 0.0
    for v in pnls:
        equity += v
        peak = max(peak, equity)
        dd = min(dd, equity - peak)
    return {
        "N": int(len(pnls)),
        "wins": wins,
        "losses": losses,
        "pf": round(pf, 3),
        "expectancy": round(total_r / len(pnls), 4),
        "max_drawdown": round(dd, 2),
        "win_rate": round(wins / len(pnls), 3),
        "total_r": round(total_r, 2),
    }


def main() -> None:
    print("=" * 70)
    print("SCBI_M5_GLOBAL STRUCTURAL EDGE DECOMPOSITION")
    print("Source: trades_baseline.csv (2020-2025 durability, N=1610)")
    print("=" * 70)

    df = pd.read_csv(TRADES_FILE)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True).dt.tz_convert("US/Eastern")
    df["entry_hour"] = df["entry_time"].dt.hour
    df["entry_year"] = df["entry_time"].dt.year
    df["entry_month"] = df["entry_time"].dt.month
    df["regime"] = df["entry_year"].map(
        lambda y: "2020-2021" if y <= 2021 else ("2022-2023" if y <= 2023 else "2024-2025")
    )
    # stress pnl
    df["pnl_r_stress"] = df["pnl_r"] - (STRESS_DELTA_PIPS / df["risk_pips"])

    results: dict = {}

    # ===== 1. GLOBAL BASELINE =====
    results["global_baseline"] = compute_metrics(df)

    # ===== 2. BY LIQUIDITY SOURCE =====
    by_source = {}
    for source in sorted(df["level"].unique()):
        by_source[source] = compute_metrics(df[df["level"] == source])
    results["by_liquidity_source"] = by_source

    # ===== 3. BY DIRECTION =====
    by_dir = {}
    for d in ["long", "short"]:
        by_dir[d] = compute_metrics(df[df["direction"] == d])
    results["by_direction"] = by_dir

    # ===== 4. BY ENTRY HOUR (TIMING BLOCKS) =====
    # Group into reasonable intraday blocks (NY time)
    def hour_block(h: int) -> str:
        if h < 4:
            return "00-04_overnight"
        elif h < 8:
            return "04-08_london_am"
        elif h < 12:
            return "08-12_ny_am"
        elif h < 16:
            return "12-16_ny_pm"
        else:
            return "16+_late"

    df["hour_block"] = df["entry_hour"].map(hour_block)
    by_block = {}
    for block in sorted(df["hour_block"].unique()):
        by_block[block] = compute_metrics(df[df["hour_block"] == block])
    results["by_hour_block"] = by_block

    # ===== 5. BY REGIME =====
    by_regime = {}
    for regime in sorted(df["regime"].unique()):
        by_regime[regime] = compute_metrics(df[df["regime"] == regime])
    results["by_regime"] = by_regime

    # ===== 6. BY EXIT REASON =====
    by_exit = {}
    for reason in sorted(df["exit_reason"].unique()):
        by_exit[reason] = compute_metrics(df[df["exit_reason"] == reason])
    results["by_exit_reason"] = by_exit

    # ===== 7. ABLATIONS =====
    ablations = {}

    # 7a. Without London (remove london_h + london_l)
    mask_no_london = ~df["level"].isin(["london_h", "london_l"])
    ablations["without_london"] = compute_metrics(df[mask_no_london])

    # 7b. Without Asia (remove asia_h + asia_l)
    mask_no_asia = ~df["level"].isin(["asia_h", "asia_l"])
    ablations["without_asia"] = compute_metrics(df[mask_no_asia])

    # 7c. Without PDH/PDL
    mask_no_pdhl = ~df["level"].isin(["pdh", "pdl"])
    ablations["without_pdh_pdl"] = compute_metrics(df[mask_no_pdhl])

    # 7d. Only London
    mask_london_only = df["level"].isin(["london_h", "london_l"])
    ablations["london_only"] = compute_metrics(df[mask_london_only])

    # 7e. Without Longs
    ablations["shorts_only"] = compute_metrics(df[df["direction"] == "short"])

    # 7f. Without Shorts
    ablations["longs_only"] = compute_metrics(df[df["direction"] == "long"])

    # 7g. Without overnight block (00-04)
    mask_no_overnight = df["hour_block"] != "00-04_overnight"
    ablations["without_overnight"] = compute_metrics(df[mask_no_overnight])

    # 7h. Without NY PM block (12-16)
    mask_no_ny_pm = df["hour_block"] != "12-16_ny_pm"
    ablations["without_ny_pm"] = compute_metrics(df[mask_no_ny_pm])

    results["ablations"] = ablations

    # ===== 8. CROSS: SOURCE x REGIME =====
    cross_source_regime = {}
    for source in sorted(df["level"].unique()):
        for regime in sorted(df["regime"].unique()):
            mask = (df["level"] == source) & (df["regime"] == regime)
            sub = df[mask]
            if len(sub) >= 5:
                key = f"{source}_{regime}"
                cross_source_regime[key] = compute_metrics(sub)
    results["cross_source_regime"] = cross_source_regime

    # ===== 9. CROSS: SOURCE x DIRECTION =====
    cross_source_dir = {}
    for source in sorted(df["level"].unique()):
        for d in ["long", "short"]:
            mask = (df["level"] == source) & (df["direction"] == d)
            sub = df[mask]
            if len(sub) >= 5:
                key = f"{source}_{d}"
                cross_source_dir[key] = compute_metrics(sub)
    results["cross_source_direction"] = cross_source_dir

    # ===== 10. STRESS ABLATIONS =====
    stress_ablations = {}
    stress_ablations["global_stress"] = compute_metrics(df, pnl_col="pnl_r_stress")
    stress_ablations["without_pdh_pdl_stress"] = compute_metrics(
        df[mask_no_pdhl], pnl_col="pnl_r_stress"
    )
    stress_ablations["london_only_stress"] = compute_metrics(
        df[mask_london_only], pnl_col="pnl_r_stress"
    )
    results["stress_ablations"] = stress_ablations

    # ===== 11. CONTRIBUTION ANALYSIS =====
    total_r = results["global_baseline"]["total_r"]
    contribution = {}
    for source, metrics in by_source.items():
        contribution[source] = {
            "total_r": metrics["total_r"],
            "share_pct": round(100.0 * metrics["total_r"] / total_r, 1) if total_r else 0.0,
            "N": metrics["N"],
            "pf": metrics["pf"],
        }
    results["contribution_by_source"] = contribution

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OUTPUT] {OUTPUT_FILE}")
    print(f"[GLOBAL] N={results['global_baseline']['N']} PF={results['global_baseline']['pf']} Exp={results['global_baseline']['expectancy']}")

    # Print contribution summary
    print("\n--- CONTRIBUTION BY SOURCE ---")
    for source, c in sorted(contribution.items(), key=lambda x: -x[1]["total_r"]):
        print(f"  {source:12s}: {c['total_r']:+8.2f}R ({c['share_pct']:5.1f}%) N={c['N']:4d} PF={c['pf']:.3f}")

    # Print ablation summary
    print("\n--- ABLATION SUMMARY ---")
    for name, m in ablations.items():
        delta_pf = m["pf"] - results["global_baseline"]["pf"]
        delta_r = m["total_r"] - results["global_baseline"]["total_r"]
        print(f"  {name:30s}: N={m['N']:5d} PF={m['pf']:.3f} (d{delta_pf:+.3f}) TotalR={m['total_r']:+.2f}R (d{delta_r:+.2f}R)")

    # Print hour blocks
    print("\n--- HOUR BLOCKS ---")
    for block, m in sorted(by_block.items()):
        print(f"  {block:25s}: N={m['N']:5d} PF={m['pf']:.3f} Exp={m['expectancy']:+.4f}R")

    print("\nDone.")


if __name__ == "__main__":
    main()
