import pandas as pd
import numpy as np
import json
import os

# Paths
trade_level_201708 = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase50s_results\PHASE50S_201708_GEMINI_TRADE_LEVEL.csv"
parquet_201708 = r"C:\Users\alera\Desktop\Bot\BOT_MARKET_DATA\tick\EURUSD\monthly\EURUSD_ticks_2017_08.parquet"
output_dir_201708 = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase50s_results"
output_dir_global = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical"

# Load 2017-08 trade level
df_tl = pd.read_csv(trade_level_201708)
df_tl['exit_time_utc'] = pd.to_datetime(df_tl['exit_time_utc'])

# Load parquet (only for the trades we need)
df_ticks = pd.read_parquet(parquet_201708)
df_ticks['timestamp_utc'] = pd.to_datetime(df_ticks['timestamp_utc'], utc=True)
df_ticks = df_ticks.sort_values('timestamp_utc')

# Task 1: Audit execution
audit_201708 = []
for _, row in df_tl.iterrows():
    if row['auditable_yes_no'] == 'NO': continue
    if row['tick_outcome'] != 'TIME_EXIT': continue # User asked for TIME_EXIT forensic
    
    # Get last tick for this trade
    mask = (df_ticks['timestamp_utc'] <= row['exit_time_utc']) & (df_ticks['timestamp_utc'] >= pd.to_datetime(row['entry_time_utc']))
    trade_ticks = df_ticks[mask]
    if trade_ticks.empty: continue
    
    last_tick = trade_ticks.iloc[-1]
    bid_exit = last_tick['bid']
    ask_exit = last_tick['ask']
    spread_exit = ask_exit - bid_exit
    
    direction = row['direction']
    exec_price = bid_exit if direction == 'LONG' else ask_exit
    
    # R calculations
    risk = row['risk_pips']
    # tick_R (already calculated in previous script)
    r_before = row['tick_R']
    
    audit_201708.append({
        'trade_id': int(row['trade_id']),
        'direction': direction,
        'entry_time': row['entry_time_utc'],
        'exit_time': row['exit_time_utc'],
        'bid_exit': bid_exit,
        'ask_exit': ask_exit,
        'spread_exit_pips': spread_exit * 10000,
        'exec_price': exec_price,
        'risk_pips_val': risk,
        'r_before_costs': r_before
    })

df_audit_201708 = pd.DataFrame(audit_201708)
df_audit_201708.to_csv(os.path.join(output_dir_201708, 'PHASE50U_201708_TIME_EXIT_EXECUTION_AUDIT.csv'), index=False)

# Task 2: Cost Stress 2017-08
def calc_stress_metrics(df_m, spread_mult=1.0, slippage_pips=0.0, comm_r=0.0, no_audit_r=0.0):
    # We only have auditables here
    results = []
    for _, r in df_m.iterrows():
        # Adjust entry and exit for spread/slippage
        # Original tick_R used real Bid/Ask at entry and exit.
        # R = (Exit - Entry) / Risk (LONG)
        # R = (Entry - Exit) / Risk (SHORT)
        
        # We'll apply the increments to the R
        # 1. Spread shock on exit (already included in tick_R, but we add more)
        # extra_spread = (spread_mult - 1.0) * spread_exit
        # 2. Slippage (fixed pips at entry AND exit)
        # total_slip = 2 * slippage_pips
        
        # Simpler way: Penalize R
        # R_new = R_old - (extra_spread_pips / risk_pips) - (total_slip_pips / risk_pips) - comm_r
        
        # But spread_exit varies.
        extra_spread_r = ( (r['spread_exit_pips']/10000) * (spread_mult - 1.0) ) / r['risk_pips_val']
        slip_r = (slippage_pips / 10000 * 2) / r['risk_pips_val']
        
        r_new = r['r_before_costs'] - extra_spread_r - slip_r - comm_r
        results.append(r_new)
    
    # Add no_audit_r for missing trades if needed
    # (Actually sample 15 auditables)
    
    r_arr = np.array(results)
    total_r = r_arr.sum()
    wins = r_arr[r_arr > 0].sum()
    losses = abs(r_arr[r_arr < 0].sum())
    pf = wins / losses if losses > 0 else (wins if wins > 0 else 0.0)
    expectancy = total_r / len(r_arr)
    winrate = len(r_arr[r_arr > 0]) / len(r_arr) * 100
    cum_r = r_arr.cumsum()
    dd = (pd.Series(cum_r).cummax() - cum_r).max()
    
    return {
        'PF': float(pf),
        'expectancy': float(expectancy),
        'DD': float(dd),
        'winrate': float(winrate),
        'total_R': float(total_r),
        'pass': "Pass" if pf >= 1.1 else "Fail"
    }

stress_201708 = {
    'BASE_EXECUTABLE': calc_stress_metrics(df_audit_201708),
    'SPREAD_SHOCK_1_5X': calc_stress_metrics(df_audit_201708, spread_mult=1.5),
    'SPREAD_SHOCK_2X': calc_stress_metrics(df_audit_201708, spread_mult=2.0),
    'SLIPPAGE_0_1_PIP': calc_stress_metrics(df_audit_201708, slippage_pips=0.1),
    'SLIPPAGE_0_2_PIP': calc_stress_metrics(df_audit_201708, slippage_pips=0.2),
    'SLIPPAGE_0_5_PIP': calc_stress_metrics(df_audit_201708, slippage_pips=0.5),
    'SLIPPAGE_1_0_PIP': calc_stress_metrics(df_audit_201708, slippage_pips=1.0),
    'COMMISSION_0_05R': calc_stress_metrics(df_audit_201708, comm_r=0.05),
    'COMMISSION_0_10R': calc_stress_metrics(df_audit_201708, comm_r=0.10),
    'COMMISSION_0_20R': calc_stress_metrics(df_audit_201708, comm_r=0.20),
    'ADVERSE_COMBINED': calc_stress_metrics(df_audit_201708, spread_mult=2.0, slippage_pips=0.5, comm_r=0.10),
    'ULTRA_ADVERSE': calc_stress_metrics(df_audit_201708, spread_mult=2.0, slippage_pips=1.0, comm_r=0.20)
}

with open(os.path.join(output_dir_201708, 'PHASE50U_201708_TIME_EXIT_COST_STRESS.json'), 'w') as f:
    json.dump(stress_201708, f, indent=4)

# Task 3: Global Stress
recent_files = [
    r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\PHASE50M_CORRECTED_TICK_TRADE_LEVEL.csv",
    r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\PHASE50P_VALIDATED_REPLAY_TRADE_LEVEL.csv",
    r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\PHASE50P_FULL_BATCH_CERTIFIED_LAT_1S.csv",
    trade_level_201708
]

global_audit = []
for f_path in recent_files:
    if os.path.exists(f_path):
        rdf = pd.read_csv(f_path)
        r_col = 'tick_R' if 'tick_R' in rdf.columns else 'pnl_r'
        reason_col = 'tick_outcome' if 'tick_outcome' in rdf.columns else ('exit_reason' if 'exit_reason' in rdf.columns else 'outcome')
        
        if r_col in rdf.columns and reason_col in rdf.columns:
            is_te = rdf[reason_col].isin(['TIME_EXIT', 'FORCED_CLOSE', 'forced_session_close', 'time_exit'])
            
            def get_pf(vals):
                w = vals[vals > 0].sum()
                l = abs(vals[vals < 0].sum())
                return w / l if l > 0 else (w if w > 0 else 0.0)

            pf_orig = get_pf(rdf[r_col])
            
            # Stress TE trades
            def pf_te_stress(rdf, pen):
                vals = rdf[r_col].copy()
                vals.loc[is_te] -= pen
                return get_pf(vals)

            global_audit.append({
                'file': os.path.basename(f_path),
                'pf_original': pf_orig,
                'pf_te_minus_0_05R': pf_te_stress(rdf, 0.05),
                'pf_te_minus_0_10R': pf_te_stress(rdf, 0.10),
                'pf_te_minus_0_20R': pf_te_stress(rdf, 0.20),
                'pf_te_minus_0_50R': pf_te_stress(rdf, 0.50),
                'pf_no_te': get_pf(rdf[~is_te][r_col]) if any(~is_te) else 0.0
            })

pd.DataFrame(global_audit).to_csv(os.path.join(output_dir_global, 'PHASE50U_GLOBAL_TIME_EXIT_COST_STRESS.csv'), index=False)

print("Cost stress analysis finished.")
