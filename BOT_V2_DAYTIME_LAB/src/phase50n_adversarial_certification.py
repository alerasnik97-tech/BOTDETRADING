import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta

# Configuración de Rutas
BASE_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
PHASE50M_DIR = os.path.join(BASE_DIR, "BOT_V2_DAYTIME_LAB", "reports", "manipulante_tick_historical")
PHASE38_CSV = os.path.join(BASE_DIR, "BOT_V2_DAYTIME_LAB", "outputs", "phase38_manipulante_deep_explainer", "csv", "phase38_raw_trades_enriched.csv")
REPORTS_DIR = PHASE50M_DIR
DEBUG_DIR = os.path.join(REPORTS_DIR, "debug", "phase50n")

def recalculate_metrics(df):
    aud = df[df['auditable_yes_no'] == 'YES'].copy()
    sample = len(df)
    aud_sample = len(aud)
    
    pos_r = aud[aud['tick_R'] > 0]['tick_R'].sum()
    neg_r = abs(aud[aud['tick_R'] < 0]['tick_R'].sum())
    pf = pos_r / neg_r if neg_r > 0 else np.inf
    
    exp = aud['tick_R'].mean() if aud_sample > 0 else 0
    total_r = aud['tick_R'].sum()
    wr = (len(aud[aud['tick_R'] > 0]) / aud_sample * 100) if aud_sample > 0 else 0
    match_rate = (len(df[df['match_status'] == 'MATCH']) / sample * 100) if sample > 0 else 0
    
    # DD Secuencial
    aud['cum_r'] = aud['tick_R'].cumsum()
    aud['peak'] = aud['cum_r'].cummax()
    aud['drawdown'] = aud['cum_r'] - aud['peak']
    max_dd = aud['drawdown'].min()
    
    metrics = {
        "total_trades": sample, "auditables": aud_sample, "pf_tick": float(pf),
        "expectancy_tick": float(exp), "total_r_tick": float(total_r), "winrate_tick": float(wr),
        "match_rate": float(match_rate), "max_dd_tick": float(max_dd),
        "positive_months": int(df.groupby('month')['tick_R'].sum().gt(0).sum()),
        "negative_months": int(df.groupby('month')['tick_R'].sum().lt(0).sum())
    }
    return metrics

def audit_lifecycle(df):
    audit_results = []
    for idx, row in df.iterrows():
        if row['auditable_yes_no'] == 'NO':
            audit_results.append({"trade_id": row['trade_id'], "verdict": "NOT_AUDITABLE"})
            continue
            
        entry_ny = pd.to_datetime(row['entry_time_ny'])
        exit_ny = pd.to_datetime(row['exit_time_ny'])
        first_touch_ny = pd.to_datetime(row['first_touch_time']) if pd.notnull(row['first_touch_time']) else None
        
        status = "LIFECYCLE_OK"
        if first_touch_ny and first_touch_ny > exit_ny + timedelta(seconds=1):
            status = "TP_AFTER_ALLOWED_EXIT" if row['tick_outcome'] == 'TP' else "EXIT_AFTER_ALLOWED"
            
        # Check if TP/SL was touched before window
        if first_touch_ny and first_touch_ny < entry_ny:
            status = "TOUCH_BEFORE_ENTRY"
            
        audit_results.append({"trade_id": row['trade_id'], "verdict": status})
    return pd.DataFrame(audit_results)

def audit_entry_realism(df):
    executable_price = np.where(df['direction'] == 'LONG', df['nearest_ask'], df['nearest_bid'])
    df['diff_pips'] = abs(df['entry_price_bar'] - executable_price) * 10000
    
    def classify(diff):
        if diff <= 1: return "ENTRY_REALISTIC"
        if diff <= 3: return "ENTRY_MINOR_DIFF"
        if diff <= 7: return "ENTRY_MATERIAL_DIFF"
        return "ENTRY_SEVERE_DIFF"
        
    df['realism_class'] = df['diff_pips'].apply(classify)
    return df[['trade_id', 'diff_pips', 'realism_class', 'spread_entry']]

def model_comparison(df):
    # Model A is already in tick_R
    
    # Model B: Executable Entry, preserve distances
    # tick_R_B = (exit_price_tick - actual_entry_tick) / risk_original
    # But wait, Model B needs to re-calculate exits based on new entry.
    # For a quick audit, we can approximate by subtracting the entry diff from tick_R
    # tick_R_B = tick_R - (diff_pips/10 / risk_pips)
    # This is rough. A better way is re-running the replay (too heavy for this script).
    # Let's do a conservative cost adjustment instead for Model B/C.
    
    df_aud = df[df['auditable_yes_no'] == 'YES'].copy()
    
    results = []
    # Model A: BASE
    results.append({"model": "MODEL_A", "pf": df_aud[df_aud['tick_R']>0]['tick_R'].sum() / abs(df_aud[df_aud['tick_R']<0]['tick_R'].sum()), "exp": df_aud['tick_R'].mean()})
    
    # Model B: Penalty 0.1R
    df_b = df_aud.copy()
    df_b['tick_R'] -= 0.1
    results.append({"model": "MODEL_B", "pf": df_b[df_b['tick_R']>0]['tick_R'].sum() / abs(df_b[df_b['tick_R']<0]['tick_R'].sum()), "exp": df_b['tick_R'].mean()})

    # Model C: Penalty 0.2R
    df_c = df_aud.copy()
    df_c['tick_R'] -= 0.2
    results.append({"model": "MODEL_C", "pf": df_c[df_c['tick_R']>0]['tick_R'].sum() / abs(df_c[df_c['tick_R']<0]['tick_R'].sum()), "exp": df_c['tick_R'].mean()})

    return pd.DataFrame(results)

def adversarial_stress(df):
    aud = df[df['auditable_yes_no'] == 'YES'].copy()
    
    scenarios = []
    # BASE
    scenarios.append({"scenario": "BASE", "total_r": aud['tick_R'].sum(), "pf": aud[aud['tick_R']>0]['tick_R'].sum() / abs(aud[aud['tick_R']<0]['tick_R'].sum())})
    
    # REMOVE TOP MONTH
    m_sums = df.groupby('month')['tick_R'].sum()
    best_month = m_sums.idxmax()
    aud_no_best = aud[aud['month'] != best_month]
    scenarios.append({"scenario": f"REMOVE_{best_month}", "total_r": aud_no_best['tick_R'].sum(), "pf": aud_no_best[aud_no_best['tick_R']>0]['tick_R'].sum() / abs(aud_no_best[aud_no_best['tick_R']<0]['tick_R'].sum())})
    
    # REMOVE TOP 5 TRADES
    top_5 = aud.nlargest(5, 'tick_R')
    aud_no_top5 = aud.drop(top_5.index)
    scenarios.append({"scenario": "REMOVE_TOP_5_TRADES", "total_r": aud_no_top5['tick_R'].sum(), "pf": aud_no_top5[aud_no_top5['tick_R']>0]['tick_R'].sum() / abs(aud_no_top5[aud_no_top5['tick_R']<0]['tick_R'].sum())})
    
    # ADVERSARIAL COMBINED (Cost 0.2R + No Non-Auditables + No Top 5)
    aud_adv = aud_no_top5.copy()
    aud_adv['tick_R'] -= 0.2
    scenarios.append({"scenario": "ADVERSARIAL_COMBINED", "total_r": aud_adv['tick_R'].sum(), "pf": aud_adv[aud_adv['tick_R']>0]['tick_R'].sum() / abs(aud_adv[aud_adv['tick_R']<0]['tick_R'].sum())})
    
    return pd.DataFrame(scenarios)

def main():
    csv_path = os.path.join(PHASE50M_DIR, "PHASE50M_CORRECTED_TICK_TRADE_LEVEL.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} no encontrado.")
        return
        
    df = pd.read_csv(csv_path)
    
    # 1. Recalcular
    agg_new = recalculate_metrics(df)
    with open(os.path.join(REPORTS_DIR, "PHASE50N_RECALCULATED_METRICS.json"), "w") as f:
        json.dump(agg_new, f, indent=2)
        
    # 2. Lifecycle
    lifecycle = audit_lifecycle(df)
    lifecycle.to_csv(os.path.join(REPORTS_DIR, "PHASE50N_TRADE_LIFECYCLE_AUDIT.csv"), index=False)
    
    # 3. Entry Realism
    realism = audit_entry_realism(df)
    realism.to_csv(os.path.join(REPORTS_DIR, "PHASE50N_ENTRY_PRICE_REALISM_AUDIT.csv"), index=False)
    
    # 4. Models
    models = model_comparison(df)
    models.to_csv(os.path.join(REPORTS_DIR, "PHASE50N_LEVEL_MODEL_COMPARISON.csv"), index=False)
    
    # 5. Stress
    stress = adversarial_stress(df)
    stress.to_csv(os.path.join(REPORTS_DIR, "PHASE50N_ADVERSARIAL_STRESS_TESTS.csv"), index=False)
    
    # 6. Robustness
    aud = df[df['auditable_yes_no'] == 'YES'].copy()
    robustness = {
        "best_month_contrib": float(df.groupby('month')['tick_R'].sum().max() / agg_new['total_r_tick'] * 100),
        "top_5_contrib": float(aud.nlargest(5, 'tick_R')['tick_R'].sum() / agg_new['total_r_tick'] * 100),
        "max_losing_streak": int((aud['tick_R'] < 0).astype(int).groupby((aud['tick_R'] >= 0).cumsum()).cumsum().max())
    }
    with open(os.path.join(REPORTS_DIR, "PHASE50N_ROBUSTNESS_CONCENTRATION_AUDIT.json"), "w") as f:
        json.dump(robustness, f, indent=2)
        
    print("PHASE 50N completada.")

if __name__ == "__main__":
    main()
