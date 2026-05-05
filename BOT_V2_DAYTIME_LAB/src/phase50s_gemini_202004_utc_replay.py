import pandas as pd
import numpy as np
import json
import os

csv_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase38_manipulante_deep_explainer\csv\phase38_raw_trades_enriched.csv"
parquet_path = r"C:\Users\alera\Desktop\Bot\BOT_MARKET_DATA\tick\EURUSD\monthly\EURUSD_ticks_2020_04.parquet"
output_dir = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase50s_results"

os.makedirs(output_dir, exist_ok=True)

df_trades = pd.read_csv(csv_path)
df_trades = df_trades[df_trades['year_month'].astype(str) == '2020-04'].copy()

# Initialize data quality metrics
dq = {
    'total_trades': len(df_trades),
    'parquet_missing': False,
    'gaps_found': 0,
    'first_tick': None,
    'last_tick': None,
    'rows': 0
}

if not os.path.exists(parquet_path):
    dq['parquet_missing'] = True
    df_ticks = pd.DataFrame()
else:
    df_ticks = pd.read_parquet(parquet_path)
    df_ticks['timestamp_utc'] = pd.to_datetime(df_ticks['timestamp_utc'], utc=True)
    df_ticks = df_ticks.sort_values('timestamp_utc')
    dq['rows'] = len(df_ticks)
    dq['first_tick'] = str(df_ticks['timestamp_utc'].min())
    dq['last_tick'] = str(df_ticks['timestamp_utc'].max())

df_trades['entry_time_utc'] = pd.to_datetime(df_trades['entry_time'], utc=True)
df_trades['exit_time_utc'] = pd.to_datetime(df_trades['exit_time'], utc=True)

results = []

for idx, row in df_trades.iterrows():
    trade_id = idx
    entry_time_original = row['entry_time']
    exit_time_original = row['exit_time']
    entry_time_utc = row['entry_time_utc']
    exit_time_utc = row['exit_time_utc']
    direction = row['type']
    entry_price_raw = row['entry_price']
    risk_pips = row['risk']
    tp = row['tp']
    
    if direction == 'LONG':
        sl = entry_price_raw - risk_pips
    else:
        sl = entry_price_raw + risk_pips
    
    if risk_pips == 0:
        continue
        
    be_trigger_price = entry_price_raw + 0.4 * risk_pips if direction == 'LONG' else entry_price_raw - 0.4 * risk_pips
    be_stop_price = entry_price_raw
    
    if dq['parquet_missing']:
        results.append({
            'trade_id': trade_id,
            'entry_time_original': entry_time_original,
            'entry_time_utc': entry_time_utc,
            'exit_time_original': exit_time_original,
            'exit_time_utc': exit_time_utc,
            'direction': direction,
            'entry_price_raw': entry_price_raw,
            'executable_entry': None,
            'executable_side': 'Ask' if direction == 'LONG' else 'Bid',
            'spread_entry': None,
            'sl': sl,
            'tp': tp,
            'risk_pips': risk_pips,
            'be_trigger_price': be_trigger_price,
            'be_stop_price': be_stop_price,
            'ticks_loaded': 0,
            'tick_outcome': 'NO_AUDITABLE_DATA_MISSING',
            'tick_R': 0.0,
            'auditable_yes_no': 'NO',
            'non_auditable_reason': 'PARQUET_NOT_FOUND',
            'notes': 'Parquet file is missing'
        })
        continue

    mask = (df_ticks['timestamp_utc'] >= entry_time_utc) & (df_ticks['timestamp_utc'] <= exit_time_utc)
    trade_ticks = df_ticks[mask]
    
    if trade_ticks.empty:
        results.append({
            'trade_id': trade_id,
            'entry_time_original': entry_time_original,
            'entry_time_utc': entry_time_utc,
            'exit_time_original': exit_time_original,
            'exit_time_utc': exit_time_utc,
            'direction': direction,
            'entry_price_raw': entry_price_raw,
            'executable_entry': None,
            'executable_side': 'Ask' if direction == 'LONG' else 'Bid',
            'spread_entry': None,
            'sl': sl,
            'tp': tp,
            'risk_pips': risk_pips,
            'be_trigger_price': be_trigger_price,
            'be_stop_price': be_stop_price,
            'ticks_loaded': 0,
            'tick_outcome': 'NO_AUDITABLE_DATA_MISSING',
            'tick_R': 0.0,
            'auditable_yes_no': 'NO',
            'non_auditable_reason': 'NO_TICKS_IN_WINDOW',
            'notes': 'Parquet lacks data for this period'
        })
        continue
        
    first_tick = trade_ticks.iloc[0]
    executable_entry = first_tick['ask'] if direction == 'LONG' else first_tick['bid']
    spread_entry = first_tick['ask'] - first_tick['bid']
    
    be_triggered = False
    tick_outcome = 'TIME_EXIT'
    tick_R = 0.0
    first_sl_time_utc = None
    first_be_trigger_time_utc = None
    first_be_stop_time_utc = None
    first_tp_time_utc = None
    
    for t_idx, t_row in trade_ticks.iterrows():
        eval_price = t_row['bid'] if direction == 'LONG' else t_row['ask']
        current_time = t_row['timestamp_utc']
        
        hit_sl = False
        hit_tp = False
        hit_be_trigger = False
        hit_be_stop = False
        
        if direction == 'LONG':
            if eval_price <= sl: hit_sl = True
            if eval_price >= tp: hit_tp = True
            if eval_price >= be_trigger_price: hit_be_trigger = True
            if eval_price <= be_stop_price: hit_be_stop = True
        else: # SHORT
            if eval_price >= sl: hit_sl = True
            if eval_price <= tp: hit_tp = True
            if eval_price <= be_trigger_price: hit_be_trigger = True
            if eval_price >= be_stop_price: hit_be_stop = True
            
        if not be_triggered:
            hits = sum([hit_sl, hit_tp, hit_be_trigger])
            if hits > 1:
                tick_outcome = 'AMBIGUOUS_SAME_TIMESTAMP'
                tick_R = 0.0
                if hit_sl: first_sl_time_utc = current_time
                if hit_tp: first_tp_time_utc = current_time
                if hit_be_trigger: first_be_trigger_time_utc = current_time
                break
            elif hit_sl:
                tick_outcome = 'SL'
                tick_R = -1.0
                first_sl_time_utc = current_time
                break
            elif hit_tp:
                tick_outcome = 'TP'
                tick_R = abs(tp - executable_entry) / risk_pips if risk_pips > 0 else 0
                first_tp_time_utc = current_time
                break
            elif hit_be_trigger:
                be_triggered = True
                first_be_trigger_time_utc = current_time
        else:
            hits = sum([hit_be_stop, hit_tp])
            if hits > 1:
                tick_outcome = 'AMBIGUOUS_SAME_TIMESTAMP'
                tick_R = 0.0
                if hit_be_stop: first_be_stop_time_utc = current_time
                if hit_tp: first_tp_time_utc = current_time
                break
            elif hit_be_stop:
                tick_outcome = 'BE'
                tick_R = 0.0
                first_be_stop_time_utc = current_time
                break
            elif hit_tp:
                tick_outcome = 'TP'
                tick_R = abs(tp - executable_entry) / risk_pips if risk_pips > 0 else 0
                first_tp_time_utc = current_time
                break

    if tick_outcome == 'TIME_EXIT':
        last_tick = trade_ticks.iloc[-1]
        last_eval_price = last_tick['bid'] if direction == 'LONG' else last_tick['ask']
        if direction == 'LONG':
            tick_R = (last_eval_price - executable_entry) / risk_pips
        else:
            tick_R = (executable_entry - last_eval_price) / risk_pips
            
    results.append({
        'trade_id': trade_id,
        'entry_time_original': entry_time_original,
        'entry_time_utc': entry_time_utc,
        'exit_time_original': exit_time_original,
        'exit_time_utc': exit_time_utc,
        'direction': direction,
        'entry_price_raw': entry_price_raw,
        'executable_entry': executable_entry,
        'executable_side': 'Ask' if direction == 'LONG' else 'Bid',
        'spread_entry': spread_entry,
        'sl': sl,
        'tp': tp,
        'risk_pips': risk_pips,
        'be_trigger_price': be_trigger_price,
        'be_stop_price': be_stop_price,
        'ticks_loaded': len(trade_ticks),
        'tick_outcome': tick_outcome,
        'tick_R': tick_R,
        'auditable_yes_no': 'YES',
        'non_auditable_reason': None,
        'notes': None
    })

df_results = pd.DataFrame(results)
csv_out_path = os.path.join(output_dir, 'PHASE50S_202004_GEMINI_TRADE_LEVEL.csv')
df_results.to_csv(csv_out_path, index=False)

sample_total = len(df_results)
auditables_df = df_results[df_results['auditable_yes_no'] == 'YES']
no_auditables_df = df_results[df_results['auditable_yes_no'] == 'NO']

auditables = len(auditables_df)
no_auditables = len(no_auditables_df)

if auditables > 0:
    total_R_real = auditables_df['tick_R'].sum()
    win_count = len(auditables_df[auditables_df['tick_R'] > 0])
    loss_count = len(auditables_df[auditables_df['tick_R'] < 0])
    gross_profit = auditables_df[auditables_df['tick_R'] > 0]['tick_R'].sum()
    gross_loss = abs(auditables_df[auditables_df['tick_R'] < 0]['tick_R'].sum())
    
    pf_real = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
    winrate_real = win_count / auditables * 100
    expectancy_real = total_R_real / auditables
    
    cum_R = auditables_df['tick_R'].cumsum()
    running_max = cum_R.cummax()
    drawdowns = running_max - cum_R
    dd_real = drawdowns.max()
    
    is_loss = auditables_df['tick_R'] < 0
    streak = is_loss.groupby((~is_loss).cumsum()).sum()
    max_losing_streak = streak.max()
else:
    pf_real = 0.0
    expectancy_real = 0.0
    dd_real = 0.0
    winrate_real = 0.0
    total_R_real = 0.0
    max_losing_streak = 0

tp_count = len(auditables_df[auditables_df['tick_outcome'] == 'TP'])
be_count = len(auditables_df[auditables_df['tick_outcome'] == 'BE'])
sl_count = len(auditables_df[auditables_df['tick_outcome'] == 'SL'])
time_exit_count = len(auditables_df[auditables_df['tick_outcome'] == 'TIME_EXIT'])
ambiguous_count = len(auditables_df[auditables_df['tick_outcome'] == 'AMBIGUOUS_SAME_TIMESTAMP'])

metrics = {
    'sample_total': sample_total,
    'auditables': auditables,
    'no_auditables': no_auditables,
    'PF_real': float(pf_real),
    'expectancy_real': float(expectancy_real),
    'DD_secuencial_real': float(dd_real),
    'winrate_real': float(winrate_real),
    'total_R_real': float(total_R_real),
    'TP_count': tp_count,
    'BE_count': be_count,
    'SL_count': sl_count,
    'TIME_EXIT_count': time_exit_count,
    'AMBIGUOUS_count': ambiguous_count,
    'max_losing_streak': int(max_losing_streak)
}

json_out_path = os.path.join(output_dir, 'PHASE50S_202004_GEMINI_METRICS.json')
with open(json_out_path, 'w') as f:
    json.dump(metrics, f, indent=4)

with open(os.path.join(output_dir, 'PHASE50S_202004_DATA_QUALITY.json'), 'w') as f:
    json.dump(dq, f, indent=4)

# TIME_EXIT stress if applies
if auditables > 0 and (time_exit_count / auditables > 0.3):
    stress_models = {
        'model_A_actual': metrics,
        'model_B_zero': {**metrics, 'total_R_real': float(total_R_real - auditables_df[auditables_df['tick_outcome'] == 'TIME_EXIT']['tick_R'].sum())},
        'model_C_penalty': {**metrics, 'total_R_real': float(total_R_real - auditables_df[auditables_df['tick_outcome'] == 'TIME_EXIT']['tick_R'].sum() - 0.2 * time_exit_count)}
    }
    with open(os.path.join(output_dir, 'PHASE50S_202004_TIME_EXIT_STRESS.json'), 'w') as f:
        json.dump(stress_models, f, indent=4)

print(f"Replay 2020-04 finished. Auditables: {auditables}")
