import pandas as pd
import numpy as np
import os
import json

# Paths
ROOT_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB"
REPORTS_DIR = os.path.join(ROOT_DIR, "reports", "manipulante_tick_historical")
SPREAD_COSTS_PATH = os.path.join(REPORTS_DIR, "PHASE51_GROUP_B_SPREAD_COSTS.csv")
AGG_METRICS_PATH = os.path.join(REPORTS_DIR, "PHASE50Z_ADVERSE_GROUP_B_AGGREGATE_METRICS.json")

def recalculate_costs():
    df_costs = pd.read_csv(SPREAD_COSTS_PATH)
    with open(AGG_METRICS_PATH, "r") as f:
        agg_base = json.load(f)
        
    sample = len(df_costs)
    avg_risk_pips = df_costs['risk_pips'].mean()
    
    # Scenarios
    # BASE is in agg_base['total_r_agregado']
    base_r = agg_base['total_r_agregado']
    
    # A) OBSERVED_SPREAD_ONLY
    cost_observed_total = df_costs['spread_r'].sum()
    r_observed = base_r - cost_observed_total
    
    # E) BROKER_CONSERVATIVE
    # slippage 0.4 pips total, commission 0.03R
    slip_cons = (0.4 / df_costs['risk_pips']).sum()
    comm_cons = sample * 0.03
    r_cons = base_r - cost_observed_total - slip_cons - comm_cons
    
    # F) BROKER_ADVERSE
    # slippage 1.0 pips total, commission 0.05R
    slip_adv = (1.0 / df_costs['risk_pips']).sum()
    comm_adv = sample * 0.05
    r_adv = base_r - cost_observed_total - slip_adv - comm_adv
    
    # Extra fixed costs
    r_01 = base_r - (sample * 0.1)
    r_02 = base_r - (sample * 0.2)
    
    recalc = {
        "sample": int(sample),
        "BASE": round(base_r, 2),
        "OBSERVED_SPREAD_ONLY": round(r_observed, 2),
        "BROKER_CONSERVATIVE": round(r_cons, 2),
        "BROKER_ADVERSE": round(r_adv, 2),
        "EXTRA_COST_0.1R": round(r_01, 2),
        "EXTRA_COST_0.2R": round(r_02, 2)
    }
    
    # Stats for JSON
    summary = {
        "scenarios": recalc,
        "verdict": "PHASE51_COSTS_SUPPORT_EDGE_WITH_WARNINGS" if r_cons > 0 else "PHASE51_EDGE_TOO_COST_SENSITIVE",
        "institutional_read": {
            "0.1R_is_realistic": True,
            "0.2R_is_extreme": True,
            "avg_cost_conservative_R": round((base_r - r_cons) / sample, 4),
            "avg_cost_adverse_R": round((base_r - r_adv) / sample, 4)
        }
    }
    
    with open(os.path.join(REPORTS_DIR, "PHASE51_GROUP_B_COST_RECALCULATION.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    # Save CSV breakdown
    df_recalc = pd.DataFrame([recalc])
    df_recalc.to_csv(os.path.join(REPORTS_DIR, "PHASE51_GROUP_B_COST_RECALCULATION.csv"), index=False)
    
    print(f"Recalculation finished. Conservative R: {r_cons:.2f}")

if __name__ == "__main__":
    recalculate_costs()
