import pandas as pd
import os

results = []
paths = {
    '2017-05': r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase50s_results\PHASE50S_201705_GEMINI_TRADE_LEVEL.csv',
    '2017-08': r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase50s_results\PHASE50S_201708_GEMINI_TRADE_LEVEL.csv',
    '2020-04': r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase50s_results\PHASE50S_202004_GEMINI_TRADE_LEVEL.csv',
    '2024-10': r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\PHASE50S_202410_TICK_AUDIT.csv'
}

for m, p in paths.items():
    row = {'month': m, 'trade_level_exists': os.path.exists(p)}
    if row['trade_level_exists']:
        df = pd.read_csv(p)
        col = 'tick_outcome' if 'tick_outcome' in df.columns else 'outcome'
        row['time_exit_count'] = len(df[df[col] == 'TIME_EXIT'])
        row['total_trades'] = len(df)
        parquet_file = f'EURUSD_ticks_{m.replace("-", "_")}.parquet'
        parquet_path = os.path.join(r'C:\Users\alera\Desktop\Bot\BOT_MARKET_DATA\tick\EURUSD\monthly', parquet_file)
        row['parquet_exists'] = os.path.exists(parquet_path)
    results.append(row)

final = pd.DataFrame(results)
out_path = r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\PHASE50W_INPUT_VALIDATION.csv'
final.to_csv(out_path, index=False)
print(final)
