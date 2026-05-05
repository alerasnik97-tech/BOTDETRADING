import pandas as pd
import numpy as np
import json
import os

csv_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase38_manipulante_deep_explainer\csv\phase38_raw_trades_enriched.csv"
output_dir = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)
df['entry_time_dt'] = pd.to_datetime(df['entry_time'], utc=True)
df['exit_time_dt'] = pd.to_datetime(df['exit_time'], utc=True)
df['duration_min'] = (df['exit_time_dt'] - df['entry_time_dt']).dt.total_seconds() / 60

# Identify TIME_EXIT
# Based on the user's prompt, TIME_EXIT might be labeled in the 'outcome' column or similar.
# Let's see unique values of 'outcome'
outcomes = df['outcome'].unique().tolist()
# We'll treat as TIME_EXIT anything that isn't TP, SL, or BE if it's labeled so.
# But often 'BE' is also a type of closing.
# Looking at the previous results, the user thinks 14 out of 19 trades in 2017-08 were TIME_EXIT.
# In my previous run, I labeled them TIME_EXIT because they didn't hit TP/SL/BE in the tick replay.
# But here we audit the CSV.

total_trades = len(df)

# Let's check distribution of 'outcome'
outcome_counts = df['outcome'].value_counts()

# We'll define TIME_EXIT for this audit as trades where 'outcome' is NOT 'TP', 'SL', or 'BE'?
# Actually, let's see what 'outcome' values exist.
print(f"Unique outcomes: {outcomes}")

# For the global audit, we'll follow the user's models.
def calc_global_metrics(df_m, model_name):
    total_R = df_m['R'].sum()
    wins = df_m[df_m['R'] > 0]['R'].sum()
    losses = abs(df_m[df_m['R'] < 0]['R'].sum())
    pf = wins / losses if losses > 0 else (wins if wins > 0 else 0.0)
    expectancy = total_R / len(df_m)
    winrate = len(df_m[df_m['R'] > 0]) / len(df_m) * 100
    
    return {
        'PF': float(pf),
        'expectancy': float(expectancy),
        'winrate': float(winrate),
        'total_R': float(total_R)
    }

# We assume for this audit that 'TIME_EXIT' trades are those that are NOT TP, SL, or BE in the CSV.
# Wait, if 'outcome' is 'BE', is it a TIME_EXIT? No, BE is a specific rule.
# Let's look for 'FORCED_CLOSE' or similar in 'outcome'
is_time_exit = ~df['outcome'].isin(['TP', 'SL', 'BE'])

# Distribution
global_dist = {
    'total_trades': total_trades,
    'time_exit_count': int(is_time_exit.sum()),
    'time_exit_pct': float(is_time_exit.mean() * 100),
    'avg_duration': float(df['duration_min'].mean()),
    'median_duration': float(df['duration_min'].median()),
    'min_duration': float(df['duration_min'].min()),
    'max_duration': float(df['duration_min'].max()),
}

# Distribution by year
dist_year = df.groupby('year')['outcome'].value_counts(normalize=True).unstack().fillna(0)
dist_year['time_exit_pct'] = (1 - dist_year.get('TP', 0) - dist_year.get('SL', 0) - dist_year.get('BE', 0)) * 100

# Stress Test
audit_df = df.copy()
audit_df['is_te'] = is_time_exit

# Model: Actual
audit_df['R'] = audit_df['r_result']
metrics_actual = calc_global_metrics(audit_df, "Actual")

# Model: No TIME_EXIT (Excluir)
metrics_no_te = calc_global_metrics(audit_df[~audit_df['is_te']], "Exclude TE")

# Model: TIME_EXIT = 0R
df_c = audit_df.copy()
df_c.loc[df_c['is_te'], 'R'] = 0.0
metrics_c = calc_global_metrics(df_c, "TE = 0R")

# Model: TIME_EXIT = -0.2R
df_d = audit_df.copy()
df_d.loc[df_d['is_te'], 'R'] = -0.2
metrics_d = calc_global_metrics(df_d, "TE = -0.2R")

stress_json = {
    'Actual': metrics_actual,
    'Exclude_TE': metrics_no_te,
    'TE_0R': metrics_c,
    'TE_minus_0_2R': metrics_d
}

with open(os.path.join(output_dir, 'PHASE50T_GLOBAL_TIME_EXIT_STRESS.json'), 'w') as f:
    json.dump(stress_json, f, indent=4)

# Save distribution CSV
df_dist = df.groupby(['year', 'outcome']).size().unstack(fill_value=0)
df_dist.to_csv(os.path.join(output_dir, 'PHASE50T_GLOBAL_TIME_EXIT_DISTRIBUTION.csv'))

# TAREA 3: Recent results review
# We check if files exist
recent_files = [
    r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\PHASE50M_CORRECTED_TICK_TRADE_LEVEL.csv",
    r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\PHASE50P_VALIDATED_REPLAY_TRADE_LEVEL.csv",
    r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\PHASE50P_FULL_BATCH_CERTIFIED_LAT_1S.csv",
    r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase50s_results\PHASE50S_201708_GEMINI_TRADE_LEVEL.csv"
]

recent_audit = []
for f_path in recent_files:
    if os.path.exists(f_path):
        rdf = pd.read_csv(f_path)
        # Identify outcome column (might vary)
        outcome_col = None
        for c in ['tick_outcome', 'outcome', 'exit_reason']:
            if c in rdf.columns:
                outcome_col = c
                break
        
        if outcome_col:
            is_te = rdf[outcome_col].isin(['TIME_EXIT', 'FORCED_CLOSE', 'forced_session_close', 'time_exit'])
            total = len(rdf)
            te_count = is_te.sum()
            # Calculate R
            r_col = 'tick_R' if 'tick_R' in rdf.columns else 'pnl_r'
            if r_col in rdf.columns:
                te_r = rdf[is_te][r_col].sum()
                total_r = rdf[r_col].sum()
                
                # Model stress
                pf_orig = rdf[rdf[r_col] > 0][r_col].sum() / abs(rdf[rdf[r_col] < 0][r_col].sum()) if any(rdf[r_col] < 0) else 0.0
                
                # PF 0R
                df_0 = rdf.copy()
                df_0.loc[is_te, r_col] = 0.0
                pf_0 = df_0[df_0[r_col] > 0][r_col].sum() / abs(df_0[df_0[r_col] < 0][r_col].sum()) if any(df_0[r_col] < 0) else 0.0
                
                recent_audit.append({
                    'file': os.path.basename(f_path),
                    'te_pct': te_count / total * 100,
                    'te_r_contribution': te_r,
                    'total_r': total_r,
                    'pf_original': pf_orig,
                    'pf_te_0r': pf_0
                })

pd.DataFrame(recent_audit).to_csv(os.path.join(output_dir, 'PHASE50T_RECENT_RESULTS_TIME_EXIT_DEPENDENCY.csv'), index=False)

# TAREA 1 Output files
source_audit = {
    "rule_exists_in_code": True,
    "engine_implementation": "time_exit triggered by max_hold_bars",
    "strategy_implementation": "Not found in MANIPULANTE_STRATEGY_CARD (except weekend close)",
    "definition": "Forced close when holding exceeds max_hold_bars or session ends",
    "is_session_close": True,
    "is_max_holding": True,
    "is_historical_artifact": "Possibly, if the CSV exit_time was generated by an engine with max_hold_bars",
}

with open(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\PHASE50T_TIME_EXIT_RULE_SOURCE_AUDIT.json", 'w') as f:
    json.dump(source_audit, f, indent=4)

# Create MD report
with open(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\PHASE50T_TIME_EXIT_RULE_SOURCE_AUDIT.md", 'w') as f:
    f.write("# PHASE50T TIME_EXIT RULE SOURCE AUDIT\n\n")
    f.write("## 1. Engine Findings\n")
    f.write("- The engine code (`research_lab/engine.py`) explicitly implements `time_exit` logic.\n")
    f.write("- It is triggered when `held_bars >= max_hold_bars`.\n")
    f.write("- Another trigger is `forced_session_close` based on `force_close_minute`.\n\n")
    f.write("## 2. Strategy Findings\n")
    f.write("- `MANIPULANTE_STRATEGY_CARD.md` does NOT list `max_hold_bars` as a core parameter.\n")
    f.write("- It ONLY mentions `Global Weekend Hard Close` at Friday 16:55 NY.\n")
    f.write("- The `manipulante_config.json` also excludes `max_hold_bars`.\n\n")
    f.write("## 3. Preliminary Conclusion\n")
    f.write("- `TIME_EXIT` appears to be a generic engine safety rule that was active during the generation of the `phase38` dataset, even if not explicitly defined as a MANIPULANTE core rule.\n")
    f.write("- Its high prevalence in August 2017 suggests the engine was 'killing' trades at a session end or bar limit.\n")

print("Audit script finished.")
