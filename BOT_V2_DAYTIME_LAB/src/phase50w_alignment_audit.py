import pandas as pd
import numpy as np
import os
from datetime import datetime, time
import zoneinfo

NY = zoneinfo.ZoneInfo("America/New_York")
UTC = zoneinfo.ZoneInfo("UTC")

raw_trades_path = r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase38_manipulante_deep_explainer\csv\phase38_raw_trades_enriched.csv'
df_raw = pd.read_csv(raw_trades_path)
df_raw['trade_id'] = df_raw.index

paths = {
    '2017-05': r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase50s_results\PHASE50S_201705_GEMINI_TRADE_LEVEL.csv',
    '2017-08': r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase50s_results\PHASE50S_201708_GEMINI_TRADE_LEVEL.csv',
    '2020-04': r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase50s_results\PHASE50S_202004_GEMINI_TRADE_LEVEL.csv',
    '2024-10': r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\PHASE50S_202410_TICK_AUDIT.csv'
}

output_trade_level = r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\PHASE50W_TIME_EXIT_CLOSE_TIME_TRADE_LEVEL.csv'
output_metrics = r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\PHASE50W_CLOSE_TIME_ALIGNMENT_METRICS.csv'

all_recalculated = []
metrics_results = []

for month, p in paths.items():
    if not os.path.exists(p):
        continue
    
    df_trades = pd.read_csv(p)
    col_outcome = 'tick_outcome' if 'tick_outcome' in df_trades.columns else 'outcome'
    col_R = 'tick_R' if 'tick_R' in df_trades.columns else 'R' if 'R' in df_trades.columns else 'r_result'
    
    # Merge with raw trades to get missing info (direction, entry_time)
    if 'direction' not in df_trades.columns and 'type' not in df_trades.columns:
        df_trades = pd.merge(df_trades, df_raw[['trade_id', 'type', 'entry_time', 'exit_time', 'entry_price', 'risk']], on='trade_id', how='left')
        df_trades = df_trades.rename(columns={'type': 'direction'})
        
    # Standardize column names
    if 'direction' not in df_trades.columns and 'type' in df_trades.columns:
        df_trades = df_trades.rename(columns={'type': 'direction'})
    if 'entry_time_utc' not in df_trades.columns and 'entry_time' in df_trades.columns:
        df_trades['entry_time_utc'] = pd.to_datetime(df_trades['entry_time'], utc=True)
    if 'exit_time_utc' not in df_trades.columns and 'exit_time' in df_trades.columns:
        df_trades['exit_time_utc'] = pd.to_datetime(df_trades['exit_time'], utc=True)
    if 'risk_pips' not in df_trades.columns and 'risk' in df_trades.columns:
        df_trades = df_trades.rename(columns={'risk': 'risk_pips'})
    if 'executable_entry' not in df_trades.columns and 'entry_price' in df_trades.columns:
        df_trades = df_trades.rename(columns={'entry_price': 'executable_entry'})
    
    # Only process TIME_EXIT trades for recalculation
    df_te = df_trades[df_trades[col_outcome] == 'TIME_EXIT'].copy()
    
    parquet_file = f'EURUSD_ticks_{month.replace("-", "_")}.parquet'
    parquet_path = os.path.join(r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_MARKET_DATA\tick\EURUSD\monthly', parquet_file)
    
    if not os.path.exists(parquet_path):
        continue
        
    df_ticks = pd.read_parquet(parquet_path)
    df_ticks['timestamp_utc'] = pd.to_datetime(df_ticks['timestamp_utc'], utc=True)
    df_ticks = df_ticks.sort_values('timestamp_utc')
    
    models = [
        ("19:00", time(19, 0)),
        ("19:45", time(19, 45)),
        ("20:00", time(20, 0))
    ]
    
    for _, row in df_te.iterrows():
        trade_id = row['trade_id']
        direction = row['direction']
        entry_time_utc = pd.to_datetime(row['entry_time_utc'], utc=True)
        original_exit_time_utc = pd.to_datetime(row['exit_time_utc'], utc=True)
        entry_price = row['executable_entry']
        risk_pips = row['risk_pips']
        original_R = row[col_R]
        
        entry_ny = entry_time_utc.astimezone(NY)
        trade_date_ny = entry_ny.date()
        
        for model_label, model_time in models:
            close_time_ny = datetime.combine(trade_date_ny, model_time).replace(tzinfo=NY)
            close_time_utc = close_time_ny.astimezone(UTC)
            
            mask = (df_ticks['timestamp_utc'] >= close_time_utc) & (df_ticks['timestamp_utc'] <= close_time_utc + pd.Timedelta(minutes=5))
            matching_ticks = df_ticks[mask]
            
            if matching_ticks.empty:
                all_recalculated.append({
                    'month': month,
                    'trade_id': trade_id,
                    'direction': direction,
                    'entry_time_utc': entry_time_utc,
                    'original_exit_time_utc': original_exit_time_utc,
                    'close_model': model_label,
                    'close_time_ny': str(close_time_ny),
                    'close_time_utc': close_time_utc,
                    'executable_exit_price': None,
                    'tick_R': 0.0,
                    'auditable_yes_no': 'NO',
                    'non_auditable_reason': 'NO_TICKS_AT_CLOSE_TIME',
                    'delta_vs_original_R': 0.0
                })
                continue
                
            tick = matching_ticks.iloc[0]
            exit_price = tick['bid'] if direction == 'long' or direction == 'LONG' else tick['ask']
            
            if direction == 'long' or direction == 'LONG':
                new_R = (exit_price - entry_price) / risk_pips if risk_pips > 0 else 0
            else:
                new_R = (entry_price - exit_price) / risk_pips if risk_pips > 0 else 0
                
            all_recalculated.append({
                'month': month,
                'trade_id': trade_id,
                'direction': direction,
                'entry_time_utc': entry_time_utc,
                'original_exit_time_utc': original_exit_time_utc,
                'close_model': model_label,
                'close_time_ny': str(close_time_ny),
                'close_time_utc': close_time_utc,
                'executable_exit_price': exit_price,
                'tick_R': new_R,
                'auditable_yes_no': 'YES',
                'non_auditable_reason': None,
                'delta_vs_original_R': new_R - original_R
            })

    # Metrics calculation
    df_non_te = df_trades[df_trades[col_outcome] != 'TIME_EXIT'].copy()
    
    for model_label, _ in models:
        recalc_this_model = [r for r in all_recalculated if r['month'] == month and r['close_model'] == model_label]
        df_recalc = pd.DataFrame(recalc_this_model)
        
        if df_recalc.empty:
            continue
            
        df_final_sample = pd.concat([
            df_non_te[[col_R]].rename(columns={col_R: 'R'}),
            df_recalc[df_recalc['auditable_yes_no'] == 'YES'][['tick_R']].rename(columns={'tick_R': 'R'})
        ])
        
        total_R = df_final_sample['R'].sum()
        wins = df_final_sample[df_final_sample['R'] > 0]['R'].sum()
        losses = abs(df_final_sample[df_final_sample['R'] < 0]['R'].sum())
        pf = wins / losses if losses > 0 else wins
        expectancy = df_final_sample['R'].mean()
        winrate = len(df_final_sample[df_final_sample['R'] > 0]) / len(df_final_sample) * 100
        
        orig_R = df_trades[col_R].sum()
        
        metrics_results.append({
            'month': month,
            'model': model_label,
            'sample': len(df_trades),
            'auditables': len(df_final_sample),
            'PF': pf,
            'expectancy': expectancy,
            'winrate': winrate,
            'total_R': total_R,
            'delta_vs_original': total_R - orig_R,
            'pass': 'PASS' if total_R > 0 else 'FAIL'
        })

df_all_recalc = pd.DataFrame(all_recalculated)
df_all_recalc.to_csv(output_trade_level, index=False)

df_metrics = pd.DataFrame(metrics_results)
df_metrics.to_csv(output_metrics, index=False)

print("Alignment audit finished.")
print(df_metrics)
