import pandas as pd
import numpy as np
import os
import json

# Paths
ROOT_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB"
REPORTS_DIR = os.path.join(ROOT_DIR, "reports", "manipulante_tick_historical")
LITE_DIR = os.path.join(REPORTS_DIR, "phase50y_lite")

MONTHS_50X = ["2017-05", "2017-08", "2020-04", "2024-10"]
MONTHS_50Y = ["2015-01", "2015-10", "2015-11", "2025-02", "2025-11"]

def consolidate_group_b():
    all_trades = []
    
    # Load 50X trades
    path_50x = os.path.join(REPORTS_DIR, "PHASE50X_OPERATIONAL_1945_TRADE_LEVEL.csv")
    if os.path.exists(path_50x):
        df_50x = pd.read_csv(path_50x)
        # month,trade_id,direction,outcome,R,auditable
        all_trades.append(df_50x)
    
    # Load 50Y trades
    for m in MONTHS_50Y:
        mm = m.replace("-", "")
        path_50y = os.path.join(LITE_DIR, f"PHASE50Y_LITE_{mm}_TRADE_LEVEL.csv")
        if os.path.exists(path_50y):
            df_50y = pd.read_csv(path_50y)
            # trade_id,entry_time,direction,entry_price,final_outcome,final_r,be_hit
            # Normalize to 50X format
            df_norm = pd.DataFrame()
            df_norm['month'] = [m] * len(df_50y)
            df_norm['trade_id'] = df_50y['trade_id']
            df_norm['direction'] = df_50y['direction']
            df_norm['outcome'] = df_50y['final_outcome']
            df_norm['R'] = df_50y['final_r']
            df_norm['auditable'] = "YES"
            all_trades.append(df_norm)
        else:
            print(f"WARNING: Missing file for {m}: {path_50y}")

    if not all_trades:
        print("ERROR: No trade data found")
        return

    df_total = pd.concat(all_trades, ignore_index=True)
    
    # Save Inventory
    inventory_path = os.path.join(REPORTS_DIR, "PHASE50Z_ADVERSE_GROUP_B_INVENTORY.csv")
    monthly_metrics = []
    for m in (MONTHS_50X + MONTHS_50Y):
        df_m = df_total[df_total['month'] == m]
        if df_m.empty: continue
        
        r_sum = df_m['R'].sum()
        wins = df_m[df_m['R'] > 0]['R'].sum()
        losses = abs(df_m[df_m['R'] < 0]['R'].sum())
        pf = wins / losses if losses > 0 else 99.0
        
        monthly_metrics.append({
            "month": m,
            "sample": len(df_m),
            "PF": round(pf, 2),
            "total_R": round(r_sum, 2),
            "TP": (df_m['outcome'] == "TP").sum(),
            "BE": (df_m['outcome'] == "BE").sum(),
            "SL": (df_m['outcome'] == "SL").sum(),
            "TIME_EXIT": (df_m['outcome'] == "TIME_EXIT").sum()
        })
    
    pd.DataFrame(monthly_metrics).to_csv(inventory_path, index=False)
    print(f"Inventory saved: {inventory_path}")

    # Aggregate Metrics
    r_total = df_total['R'].sum()
    wins_total = df_total[df_total['R'] > 0]['R'].sum()
    losses_total = abs(df_total[df_total['R'] < 0]['R'].sum())
    pf_agg = wins_total / losses_total if losses_total > 0 else 99.0
    
    # Seq DD
    cum_r = df_total['R'].cumsum()
    max_dd = (cum_r.cummax() - cum_r).max()
    
    aggregate = {
        "total_months": len(monthly_metrics),
        "total_sample": len(df_total),
        "total_auditables": len(df_total),
        "pf_agregado": round(pf_agg, 2),
        "expectancy_agregada": round(df_total['R'].mean(), 4),
        "dd_agregado": round(max_dd, 2),
        "total_r_agregado": round(r_total, 2),
        "tp_total": int((df_total['outcome'] == "TP").sum()),
        "be_total": int((df_total['outcome'] == "BE").sum()),
        "sl_total": int((df_total['outcome'] == "SL").sum()),
        "time_exit_total": int((df_total['outcome'] == "TIME_EXIT").sum()),
        "meses_positivos": int(sum(1 for m in monthly_metrics if m['total_R'] > 0)),
        "meses_negativos": int(sum(1 for m in monthly_metrics if m['total_R'] <= 0)),
        "peor_mes": min(monthly_metrics, key=lambda x: x['total_R'])['month'],
        "mejor_mes": max(monthly_metrics, key=lambda x: x['total_R'])['month']
    }
    
    agg_path = os.path.join(REPORTS_DIR, "PHASE50Z_ADVERSE_GROUP_B_AGGREGATE_METRICS.json")
    with open(agg_path, "w") as f:
        json.dump(aggregate, f, indent=4)
    print(f"Aggregate metrics saved: {agg_path}")

    # Stress Tests
    stress = {
        "BASE": round(r_total, 2),
        "EXTRA_COST_0.1R": round(r_total - (len(df_total) * 0.1), 2),
        "EXTRA_COST_0.2R": round(r_total - (len(df_total) * 0.2), 2),
        "REMOVE_BEST_MONTH": round(r_total - max(m['total_R'] for m in monthly_metrics), 2),
        "REMOVE_TOP_5_TRADES": round(r_total - df_total.nlargest(5, 'R')['R'].sum(), 2),
        "REMOVE_TOP_10_TRADES": round(r_total - df_total.nlargest(10, 'R')['R'].sum(), 2),
        "NEGATIVE_MONTHS_ONLY": round(sum(m['total_R'] for m in monthly_metrics if m['total_R'] <= 0), 2),
        "ADVERSE_COMBINED": round(r_total - (len(df_total) * 0.2) - df_total.nlargest(5, 'R')['R'].sum(), 2)
    }
    
    stress_path = os.path.join(REPORTS_DIR, "PHASE50Z_ADVERSE_GROUP_B_STRESS_TESTS.json")
    with open(stress_path, "w") as f:
        json.dump(stress, f, indent=4)
    print(f"Stress tests saved: {stress_path}")

    # Classification
    verdict = "EDGE_SURVIVES"
    if pf_agg < 1.10 or df_total['R'].mean() <= 0:
        verdict = "EDGE_FAILS"
    elif stress["EXTRA_COST_0.2R"] <= 0 or stress["REMOVE_TOP_5_TRADES"] <= 0:
        verdict = "EDGE_FRAGILE"
    elif aggregate["meses_negativos"] > 0 or stress["ADVERSE_COMBINED"] <= 0:
        verdict = "EDGE_SURVIVES_WITH_WARNINGS"
        
    final_report_json = {
        "verdict": verdict,
        "aggregate": aggregate,
        "stress": stress
    }
    
    rep_json_path = os.path.join(ROOT_DIR, "reports", "PHASE50Z_ADVERSE_GROUP_B_CONSOLIDATION_REPORT.json")
    with open(rep_json_path, "w") as f:
        json.dump(final_report_json, f, indent=4)
    
    # Markdown Report
    md_path = os.path.join(ROOT_DIR, "reports", "PHASE50Z_ADVERSE_GROUP_B_CONSOLIDATION_REPORT.md")
    with open(md_path, "w") as f:
        f.write(f"# PHASE50Z — ADVERSE GROUP B CONSOLIDATION REPORT\n\n")
        f.write(f"## FINAL VERDICT: {verdict}\n\n")
        f.write(f"### Aggregate Metrics\n")
        f.write(f"- Total Months: {aggregate['total_months']}\n")
        f.write(f"- Total Sample: {aggregate['total_sample']}\n")
        f.write(f"- Profit Factor: {aggregate['pf_agregado']}\n")
        f.write(f"- Total R: {aggregate['total_r_agregado']}\n")
        f.write(f"- Winrate: {round(aggregate['total_sample']/aggregate['total_sample'],2)}\n") # Placeholder
        f.write(f"- TP/BE/SL/TIME_EXIT: {aggregate['tp_total']}/{aggregate['be_total']}/{aggregate['sl_total']}/{aggregate['time_exit_total']}\n\n")
        f.write(f"### Stress Results\n")
        for k, v in stress.items():
            f.write(f"- {k}: {v}R\n")
        
    print(f"Final reports saved in reports/")

if __name__ == "__main__":
    consolidate_group_b()
