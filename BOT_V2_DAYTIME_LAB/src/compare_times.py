import pandas as pd

df_master = pd.read_csv(r'outputs/phase38_manipulante_deep_explainer/csv/phase38_raw_trades_enriched.csv')
df_master['entry_time_utc'] = pd.to_datetime(df_master['entry_time'], utc=True)

df_batch = pd.read_csv(r'reports/manipulante_tick_historical/phase56_batches/batch_201502_201503/PHASE56_BATCH_201502_TRADE_LEVEL.csv')
df_batch['entry_time_utc'] = pd.to_datetime(df_batch['entry_time'], utc=True)

print('Master times (top 5) for 2015-02:')
print(df_master[df_master['year_month'] == '2015-02']['entry_time_utc'].head())

print('\nBatch times (top 5) for 2015-02:')
print(df_batch['entry_time_utc'].head())
