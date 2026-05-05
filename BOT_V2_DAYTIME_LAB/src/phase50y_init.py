import pandas as pd
import os
import json

target_months = ['2015-01', '2015-10', '2015-11', '2025-02', '2025-11']
raw_trades_path = r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase38_manipulante_deep_explainer\csv\phase38_raw_trades_enriched.csv'
output_dir = r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase50y_results'
os.makedirs(output_dir, exist_ok=True)

df_raw = pd.read_csv(raw_trades_path)
df_raw['trade_id'] = df_raw.index

checkpoint = []
raw_summary = []

for month in target_months:
    m_trades = df_raw[df_raw['year_month'].astype(str) == month]
    trade_count = len(m_trades)
    
    parquet_file = f'EURUSD_ticks_{month.replace("-", "_")}.parquet'
    parquet_path = os.path.join(r'C:\Users\alera\Desktop\Bot\BOT_MARKET_DATA\tick\EURUSD\monthly', parquet_file)
    parquet_exists = os.path.exists(parquet_path)
    
    # Raw metrics for TAREA 3
    if trade_count > 0:
        total_R = m_trades['r_result'].sum()
        pf = m_trades[m_trades['r_result'] > 0]['r_result'].sum() / abs(m_trades[m_trades['r_result'] < 0]['r_result'].sum()) if m_trades[m_trades['r_result'] < 0]['r_result'].sum() != 0 else m_trades[m_trades['r_result'] > 0]['r_result'].sum()
        raw_summary.append({
            'month': month,
            'sample': trade_count,
            'total_R': total_R,
            'PF': pf,
            'expectancy': m_trades['r_result'].mean(),
            'winrate': len(m_trades[m_trades['r_result'] > 0]) / trade_count * 100,
            'TIME_EXIT_count': len(m_trades[m_trades['outcome'].isin(['FORCED_CLOSE', 'TIME_EXIT'])])
        })
    
    checkpoint.append({
        'month': month,
        'raw_trades_found': trade_count > 0,
        'expected_trades': trade_count,
        'parquet_exists': parquet_exists,
        'tick_data_status': 'TICK_READY' if parquet_exists else 'TICK_MISSING' if trade_count > 0 else 'NO_TRADES',
        'replay_status': 'PENDING',
        'verdict': 'PENDING',
        'notes': ''
    })

with open(os.path.join(output_dir, 'PHASE50Y_EXECUTION_CHECKPOINT.json'), 'w') as f:
    json.dump(checkpoint, f, indent=4)

pd.DataFrame(raw_summary).to_csv(os.path.join(output_dir, 'PHASE50Y_RAW_MONTH_SUMMARY.csv'), index=False)

print("Checkpoint and Raw Summary created.")
print(pd.DataFrame(checkpoint))
