import pandas as pd
import os

csv_path = r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\reports\v39_manipulante4_sweep_quality\MANIPULANTE4_MICRO_PROBE_TRADES.csv'
df = pd.read_csv(csv_path)

# 1. Frequency Audit
df['entry_time_dt'] = pd.to_datetime(df['entry_time'], format='mixed')
df['date_only'] = df['entry_time_dt'].dt.date
freq = df.groupby(['config_id', 'date_only', 'phase']).size().reset_index(name='rows_count')
freq['signals_count'] = freq['rows_count'] / 2
freq['max_allowed'] = 3
freq['violation'] = freq['signals_count'] > freq['max_allowed']

audit_path = r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\reports\v39_manipulante4_sweep_quality\MANIPULANTE4_PARTIAL_TRADE_FREQUENCY_AUDIT.csv'
freq.to_csv(audit_path, index=False)

# 2. Session Audit
df['exit_time_dt'] = pd.to_datetime(df['exit_time'], format='mixed')
# Ensure UTC before converting to NY
if df['exit_time_dt'].dt.tz is None:
    df['exit_time_dt'] = df['exit_time_dt'].dt.tz_localize('UTC')
else:
    df['exit_time_dt'] = df['exit_time_dt'].dt.tz_convert('UTC')

df['exit_ny'] = df['exit_time_dt'].dt.tz_convert('America/New_York').dt.time

from datetime import time
limit = time(16, 56)
session_violations = df[df['exit_ny'] > limit]

# 3. Red Flags Analysis
summary = {
    "total_rows": len(df),
    "total_signals": len(df) / 2,
    "unique_configs": df['config_id'].nunique(),
    "months": df['month'].unique().tolist(),
    "freq_violations": int(freq['violation'].sum()),
    "session_violations": len(session_violations),
    "blown_accounts": int(df[df['ftmo_blown'] == True]['config_id'].nunique()),
    "artificial_eom_count": int(df[df['artificial_eom'] == True].shape[0]),
    "included_in_metrics_eoms": int(df[(df['artificial_eom'] == True) & (df['included_in_metrics'] == True)].shape[0])
}

print("---AUDIT RESULTS---")
print(summary)
if summary['freq_violations'] > 0:
    print("FREQUENCY VIOLATIONS DETECTED!")
    print(freq[freq['violation']].head())
if summary['session_violations'] > 0:
    print("SESSION VIOLATIONS DETECTED!")
    print(session_violations[['config_id', 'exit_time', 'exit_ny']].head())
