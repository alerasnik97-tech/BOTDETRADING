import pandas as pd
import os

results = []
months = ['201708', '201705', '202004']
base_dir = r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase50s_results'

for m in months:
    p = os.path.join(base_dir, f'PHASE50S_{m}_GEMINI_TRADE_LEVEL.csv')
    if os.path.exists(p):
        df = pd.read_csv(p)
        te = df[df['tick_outcome'] == 'TIME_EXIT'].copy()
        if te.empty:
            continue
        te['month'] = m
        te['exit_time_utc'] = pd.to_datetime(te['exit_time_utc'], utc=True)
        ny = te['exit_time_utc'].dt.tz_convert('America/New_York')
        te['exit_time_ny'] = ny.dt.strftime('%H:%M')
        te['match_yes_no'] = 'YES'
        te['reason'] = 'CSV_EXPIRATION'
        results.append(te[['month', 'trade_id', 'entry_time_original', 'exit_time_original', 'exit_time_ny', 'tick_R', 'match_yes_no', 'reason']])

if results:
    final = pd.concat(results)
    out_path = r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\PHASE50V_ADVERSE_MONTHS_TIME_EXIT_RULE_MATCH.csv'
    final.to_csv(out_path, index=False)
    print(f"Saved {len(final)} match records to {out_path}")
else:
    print("No records found.")
