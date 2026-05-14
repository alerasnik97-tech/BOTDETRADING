import pandas as pd
from pathlib import Path

# Paths
REPORTS = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\reports")
RUN_DIR = REPORTS / "v49_7b_controlled_restart_memory_safe"
GATE_DIR = REPORTS / "v49_7b_gate_validation_coverage_candidate_sanity"

def analyze_sanity():
    # 1. Ranking Analysis
    rank_file = RUN_DIR / "R1_V49_7B_CONTROLLED_CANDIDATE_RANKING.csv"
    rdf = pd.read_csv(rank_file)
    
    total_configs = len(rdf)
    n_train_pos = len(rdf[rdf["N_train"] > 0])
    n_val_pos = len(rdf[rdf["N_val"] > 0])
    n_val_zero = len(rdf[rdf["N_val"] == 0])
    
    pass_train = len(rdf[rdf["PF_train"] >= 1.0])
    pass_val = len(rdf[rdf["PF_val"] >= 1.15])
    pass_both = len(rdf[(rdf["PF_train"] >= 1.0) & (rdf["PF_val"] >= 1.15)])
    
    sanity_summary = {
        "metric": [
            "total_configs", "configs_with_N_train_gt_0", "configs_with_N_val_gt_0", 
            "configs_with_N_val_eq_0", "configs_passing_PF_train_1.0", 
            "configs_passing_PF_val_1.15", "configs_passing_both"
        ],
        "value": [
            total_configs, n_train_pos, n_val_pos, n_val_zero, pass_train, pass_val, pass_both
        ]
    }
    pd.DataFrame(sanity_summary).to_csv(GATE_DIR / "R1_V49_7B_GATE_RANKING_SANITY_AUDIT.csv", index=False)
    
    # 2. Trades Coverage
    trades_file = RUN_DIR / "R1_V49_7B_CONTROLLED_TRADES.csv"
    tdf = pd.read_csv(trades_file)
    tdf["entry_time"] = pd.to_datetime(tdf["entry_time"])
    tdf["month"] = tdf["entry_time"].dt.to_period("M")
    
    monthly_counts = tdf.groupby(["month", "phase"]).size().reset_index(name="trade_count")
    monthly_counts.to_csv(GATE_DIR / "R1_V49_7B_GATE_VAL_COVERAGE_AUDIT.csv", index=False)
    
    # 3. Top 20 Preview
    rdf.sort_values("PF_val", ascending=False).head(20).to_csv(GATE_DIR / "R1_V49_7B_GATE_TOP20_PREVIEW.csv", index=False)
    
    print(f"Sanity Audit Complete. PASS BOTH: {pass_both}")

if __name__ == "__main__":
    analyze_sanity()
