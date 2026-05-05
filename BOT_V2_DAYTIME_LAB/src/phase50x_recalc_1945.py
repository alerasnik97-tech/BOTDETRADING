import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, time
import zoneinfo

NY = zoneinfo.ZoneInfo("America/New_York")
UTC = zoneinfo.ZoneInfo("UTC")

raw_trades_path = r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase38_manipulante_deep_explainer\csv\phase38_raw_trades_enriched.csv'
df_raw = pd.read_csv(raw_trades_path)
df_raw['trade_id'] = df_raw.index

months_to_audit = ['2017-05', '2017-08', '2020-04', '2024-10']

# Standardized column names and paths
paths = {
    '2017-05': r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase50s_results\PHASE50S_201705_GEMINI_TRADE_LEVEL.csv',
    '2017-08': r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase50s_results\PHASE50S_201708_GEMINI_TRADE_LEVEL.csv',
    '2020-04': r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase50s_results\PHASE50S_202004_GEMINI_TRADE_LEVEL.csv',
    '2024-10': r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\PHASE50S_202410_TICK_AUDIT.csv'
}

output_trade_level = r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\PHASE50X_OPERATIONAL_1945_TRADE_LEVEL.csv'
output_metrics = r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\PHASE50X_OPERATIONAL_1945_MONTHLY_METRICS.csv'
output_aggregate = r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\PHASE50X_OPERATIONAL_1945_AGGREGATE_METRICS.json'

all_results = []
monthly_metrics = []

for month in months_to_audit:
    p = paths.get(month)
    if not os.path.exists(p):
        continue
        
    df_trades = pd.read_csv(p)
    if 'trade_id' not in df_trades.columns:
        df_trades['trade_id'] = df_trades.index
        
    # Get info from raw trades to fill gaps
    df_trades = pd.merge(df_trades, df_raw[['trade_id', 'type', 'entry_time', 'entry_price', 'risk', 'tp', 'sl']], on='trade_id', how='left', suffixes=('', '_raw'))
    
    parquet_file = f'EURUSD_ticks_{month.replace("-", "_")}.parquet'
    parquet_path = os.path.join(r'C:\Users\alera\Desktop\Bot\BOT_MARKET_DATA\tick\EURUSD\monthly', parquet_file)
    
    if not os.path.exists(parquet_path):
        continue
        
    df_ticks = pd.read_parquet(parquet_path)
    df_ticks['timestamp_utc'] = pd.to_datetime(df_ticks['timestamp_utc'], utc=True)
    df_ticks = df_ticks.sort_values('timestamp_utc')
    
    month_results = []
    
    for _, row in df_trades.iterrows():
        trade_id = row['trade_id']
        direction = row['type']
        entry_time_utc = pd.to_datetime(row['entry_time'], utc=True)
        entry_price_raw = row['entry_price']
        risk_pips = row['risk']
        tp_price = row['tp']
        sl_price_orig = row['sl']
        
        if direction == 'LONG':
            sl_price = entry_price_raw - risk_pips
            be_trigger = entry_price_raw + 0.4 * risk_pips
            be_stop = entry_price_raw
        else:
            sl_price = entry_price_raw + risk_pips
            be_trigger = entry_price_raw - 0.4 * risk_pips
            be_stop = entry_price_raw
            
        entry_ny = entry_time_utc.astimezone(NY)
        limit_ny = datetime.combine(entry_ny.date(), time(19, 45)).replace(tzinfo=NY)
        limit_utc = limit_ny.astimezone(UTC)
        
        if entry_time_utc >= limit_utc:
            continue

        mask = (df_ticks['timestamp_utc'] >= entry_time_utc) & (df_ticks['timestamp_utc'] <= limit_utc + pd.Timedelta(minutes=5))
        trade_ticks = df_ticks[mask]
        
        if trade_ticks.empty:
            continue
            
        first_tick = trade_ticks.iloc[0]
        exec_entry = first_tick['ask'] if direction == 'LONG' else first_tick['bid']
        
        outcome = 'TIME_EXIT'
        final_R = 0.0
        be_triggered = False
        
        for _, t_row in trade_ticks.iterrows():
            current_time_utc = t_row['timestamp_utc']
            if current_time_utc > limit_utc:
                exit_price = t_row['bid'] if direction == 'LONG' else t_row['ask']
                if direction == 'LONG':
                    final_R = (exit_price - exec_entry) / risk_pips
                else:
                    final_R = (exec_entry - exit_price) / risk_pips
                outcome = 'TIME_EXIT'
                break
                
            eval_price = t_row['bid'] if direction == 'LONG' else t_row['ask']
            
            if not be_triggered:
                if direction == 'LONG':
                    if eval_price <= sl_price:
                        outcome = 'SL'; final_R = -1.0; break
                    if eval_price >= tp_price:
                        outcome = 'TP'; final_R = abs(tp_price - exec_entry) / risk_pips; break
                    if eval_price >= be_trigger:
                        be_triggered = True
                else:
                    if eval_price >= sl_price:
                        outcome = 'SL'; final_R = -1.0; break
                    if eval_price <= tp_price:
                        outcome = 'TP'; final_R = abs(tp_price - exec_entry) / risk_pips; break
                    if eval_price <= be_trigger:
                        be_triggered = True
            else:
                if direction == 'LONG':
                    if eval_price <= be_stop:
                        outcome = 'BE'; final_R = 0.0; break
                    if eval_price >= tp_price:
                        outcome = 'TP'; final_R = abs(tp_price - exec_entry) / risk_pips; break
                else:
                    if eval_price >= be_stop:
                        outcome = 'BE'; final_R = 0.0; break
                    if eval_price <= tp_price:
                        outcome = 'TP'; final_R = abs(tp_price - exec_entry) / risk_pips; break
        
        month_results.append({
            'month': month, 'trade_id': trade_id, 'direction': direction,
            'outcome': outcome, 'R': final_R, 'auditable': 'YES'
        })

    df_m = pd.DataFrame(month_results)
    if not df_m.empty:
        total_R = df_m['R'].sum()
        wins = df_m[df_m['R'] > 0]['R'].sum()
        losses = abs(df_m[df_m['R'] < 0]['R'].sum())
        pf = wins / losses if losses > 0 else wins
        expectancy = df_m['R'].mean()
        winrate = len(df_m[df_m['R'] > 0]) / len(df_m) * 100
        counts = df_m['outcome'].value_counts().to_dict()
        monthly_metrics.append({
            'month': month, 'sample': len(df_m), 'PF': pf, 'expectancy': expectancy,
            'winrate': winrate, 'total_R': total_R, 'TP': counts.get('TP', 0),
            'BE': counts.get('BE', 0), 'SL': counts.get('SL', 0), 'TIME_EXIT': counts.get('TIME_EXIT', 0)
        })
    all_results.extend(month_results)

pd.DataFrame(all_results).to_csv(output_trade_level, index=False)
df_monthly = pd.DataFrame(monthly_metrics)
df_monthly.to_csv(output_metrics, index=False)

agg_total_R = df_monthly['total_R'].sum()
agg_pf = df_monthly['PF'].mean()
agg_expectancy = df_monthly['expectancy'].mean()

aggregate = {
    'total_months': len(df_monthly), 'agg_total_R': agg_total_R,
    'agg_pf_mean': agg_pf, 'agg_expectancy_mean': agg_expectancy,
    'months_positive': int((df_monthly['total_R'] > 0).sum()),
    'months_negative': int((df_monthly['total_R'] < 0).sum())
}
with open(output_aggregate, 'w') as f:
    json.dump(aggregate, f, indent=4)

print("Recalculation with policy lock 19:45 NY finished.")
print(df_monthly)
