import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data_intake_2015_2019"

def generate_comparison(hist_summary_file, current_summary_file):
    with open(hist_summary_file, "r") as f:
        hist = json.load(f)
    with open(current_summary_file, "r") as f:
        current = json.load(f)
        
    report = {
        "comparison_metadata": {
            "strategy": "am_silver_bullet_ny_v2",
            "historical_period": "2015-2019",
            "modern_period": "2020-2025"
        },
        "metrics": {
            "sample_size": {"hist": hist["total_trades"], "modern": current["total_trades"]},
            "profit_factor": {"hist": hist["profit_factor"], "modern": current["profit_factor"]},
            "expectancy_r": {"hist": hist["expectancy_r"], "modern": current["expectancy_r"]},
            "win_rate": {"hist": hist["win_rate"], "modern": current["win_rate"]},
            "max_drawdown_pct": {"hist": hist["max_drawdown_pct"], "modern": current["max_drawdown_pct"]}
        },
        "verdict": ""
    }
    
    # Simple logic for verdict
    if hist["profit_factor"] > 1.1 and current["profit_factor"] < 1.0:
        report["verdict"] = "HISTORICAL_ROBUSTNESS_MIXED_BUT_USABLE (Lost edge in modern era)"
    elif hist["profit_factor"] > 1.1 and current["profit_factor"] > 1.1:
        report["verdict"] = "HISTORICAL_ROBUSTNESS_CONFIRMED"
    else:
        report["verdict"] = "HISTORICAL_ROBUSTNESS_WEAK"
        
    with open(OUTPUT_DIR / "comparison_2015_2019_vs_2020_2026.json", "w") as f:
        json.dump(report, f, indent=2)
        
    with open(OUTPUT_DIR / "comparison_2015_2019_vs_2020_2026.md", "w", encoding="utf-8") as f:
        f.write("# Comparison Report: 2015-2019 vs 2020-2025\n\n")
        f.write(f"## Verdict: {report['verdict']}\n\n")
        f.write("| Metric | 2015-2019 | 2020-2025 |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| Total Trades | {hist['total_trades']} | {current['total_trades']} |\n")
        f.write(f"| Profit Factor | {hist['profit_factor']:.2f} | {current['profit_factor']:.2f} |\n")
        f.write(f"| Expectancy R | {hist['expectancy_r']:.2f} | {current['expectancy_r']:.2f} |\n")
        f.write(f"| Win Rate | {hist['win_rate']:.1f}% | {current['win_rate']:.1f}% |\n")
        f.write(f"| Max Drawdown | {hist['max_drawdown_pct']:.1f}% | {current['max_drawdown_pct']:.1f}% |\n")

if __name__ == "__main__":
    pass
