import pandas as pd
import numpy as np
from pathlib import Path
import hashlib

# Paths
REPORTS = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\reports")
R2_DIR = REPORTS / "v49_7b_r2_fix_validation_coverage"
R2B_DIR = REPORTS / "v49_7b_r2b_candidate_failure_analysis"

def analyze():
    # 1. Load Inputs
    rdf = pd.read_csv(R2_DIR / "R1_V49_7B_R2_CANDIDATE_RANKING.csv")
    tdf = pd.read_csv(R2_DIR / "R1_V49_7B_R2_TRADES.csv")
    cdf = pd.read_csv(R2_DIR / "R1_V49_7B_R2_CONFIGS.csv")
    
    tdf["entry_time"] = pd.to_datetime(tdf["entry_time"])
    tdf["month"] = tdf["entry_time"].dt.to_period("M")
    
    # 2. Divergence Analysis
    div = rdf.copy()
    div["PF_gap"] = div["PF_val"] - div["PF_train"]
    div["PF_ratio"] = div["PF_val"] / div["PF_train"].replace(0, 0.01)
    div["train_pass"] = (div["PF_train"] >= 1.0) & (div["N_train"] >= 30)
    div["val_pass"] = (div["PF_val"] >= 1.15) & (div["N_val"] >= 20)
    div["both_pass"] = div["train_pass"] & div["val_pass"]
    
    def get_fail_type(row):
        if row["both_pass"]: return "BOTH_PASS"
        if row["train_pass"]: return "TRAIN_PASS_VAL_FAIL"
        if row["val_pass"]: return "TRAIN_FAIL_VAL_PASS"
        if row["N_train"] < 30 or row["N_val"] < 20: return "LOW_N"
        return "BOTH_FAIL"
    
    div["failure_type"] = div.apply(get_fail_type, axis=1)
    div.to_csv(R2B_DIR / "R1_V49_7B_R2B_TRAIN_VAL_DIVERGENCE.csv", index=False)
    
    # 3. Parameter Effects
    full = pd.merge(rdf, cdf, on="config_id")
    params = ["session_window", "level_type", "wick_to_body_min", "entry_type", "sl_model", "target_model", "max_trades_per_day"]
    
    pe_list = []
    for p in params:
        grouped = full.groupby(p).agg({
            "PF_train": ["count", "median", "mean"],
            "PF_val": ["median", "mean"],
            "Total_R": ["mean"]
        })
        grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]
        grouped = grouped.reset_index().rename(columns={p: "parameter_value"})
        grouped["parameter_name"] = p
        pe_list.append(grouped)
    
    pd.concat(pe_list).to_csv(R2B_DIR / "R1_V49_7B_R2B_PARAMETER_EFFECTS.csv", index=False)
    
    # 4. Monthly Decomposition
    monthly = tdf.groupby(["config_id", "month", "phase"]).agg({
        "pnl_net_r": ["sum", "count", lambda x: (x > 0).sum(), lambda x: (x < 0).sum()]
    }).reset_index()
    monthly.columns = ["config_id", "month", "phase", "net_R", "N", "wins", "losses"]
    
    def calc_pf(row):
        w = row["net_R"] if row["net_R"] > 0 else 0 # Simplified monthly PF
        l = abs(row["net_R"]) if row["net_R"] < 0 else 0
        return w/l if l > 0 else 0
    
    monthly.to_csv(R2B_DIR / "R1_V49_7B_R2B_MONTHLY_DECOMPOSITION.csv", index=False)
    
    # 5. Temporal Concentration (Top 50 PF_val)
    top50_val = rdf.sort_values("PF_val", ascending=False).head(50)
    conc_list = []
    for cid in top50_val["config_id"]:
        ct = tdf[tdf["config_id"] == cid]
        val_t = ct[ct["phase"] == "VAL"]
        train_t = ct[ct["phase"] == "TRAIN"]
        
        m_val = val_t.groupby("month")["pnl_net_r"].sum()
        best_m_val = m_val.max()
        share = best_m_val / m_val.sum() if m_val.sum() > 0 else 0
        
        conc_list.append({
            "config_id": cid,
            "total_R_VAL": m_val.sum(),
            "best_month_VAL": m_val.idxmax() if not m_val.empty else "N/A",
            "best_month_R_VAL": best_m_val,
            "best_month_share_VAL": share,
            "concentration_flag": "HIGH" if share > 0.6 else "OK"
        })
    pd.DataFrame(conc_list).to_csv(R2B_DIR / "R1_V49_7B_R2B_TEMPORAL_CONCENTRATION.csv", index=False)
    
    # 6. Duplicate Tradeset
    def get_hash(cid):
        ct = tdf[tdf["config_id"] == cid].sort_values("entry_time")
        # Hash critical columns
        data = ct[["phase", "entry_time", "entry_price", "pnl_net_r"]].to_string()
        return hashlib.sha256(data.encode()).hexdigest()
    
    rdf["trade_set_hash"] = rdf["config_id"].apply(get_hash)
    hashes = rdf.groupby("trade_set_hash").agg({
        "config_id": [list, "count"]
    }).reset_index()
    hashes.columns = ["trade_set_hash", "config_ids", "config_count"]
    hashes.to_csv(R2B_DIR / "R1_V49_7B_R2B_DUPLICATE_TRADESET_AUDIT.csv", index=False)
    
    # 7. Relaxed Gate
    gates = [
        ("A", 1.00, 1.15), ("B", 0.95, 1.15), ("C", 0.90, 1.10),
        ("D", 0.90, 1.05), ("E", 0.85, 1.15), ("F", 1.00, 1.00)
    ]
    gate_results = []
    for g_id, t_th, v_th in gates:
        passing = rdf[(rdf["PF_train"] >= t_th) & (rdf["PF_val"] >= v_th)]
        gate_results.append({
            "gate_id": g_id, "train_threshold": t_th, "val_threshold": v_th,
            "configs_passing": len(passing), "best_total_R": passing["Total_R"].max() if not passing.empty else 0
        })
    pd.DataFrame(gate_results).to_csv(R2B_DIR / "R1_V49_7B_R2B_RELAXED_GATE_ANALYSIS.csv", index=False)
    
    print("Analysis Complete.")

if __name__ == "__main__":
    analyze()
