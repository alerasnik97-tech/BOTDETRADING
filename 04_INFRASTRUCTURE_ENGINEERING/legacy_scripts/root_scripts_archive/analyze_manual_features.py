import pandas as pd
import numpy as np

# Load normalized ledger
df = pd.read_csv('EURUSD_MANUAL_FEATURE_LEDGER.csv')

print(f"Total trades: {len(df)}")
print(f"Date range: {df['dateStart'].min()} to {df['dateStart'].max()}")

# Additional feature engineering
# Time blocks
df['time_block'] = np.where(df['hour_ny'] < 4, '3am-4am',
                            np.where(df['hour_ny'] < 5, '4am-5am', '5am-6am'))

# Year buckets
df['year_bucket'] = np.where(df['year'] <= 2021, '2020-2021',
                             np.where(df['year'] <= 2023, '2022-2023', '2024-2025'))

# RR buckets
df['rr_bucket'] = np.where(df['avgRiskReward'] <= 0, '0',
                           np.where(df['avgRiskReward'] <= 0.5, '0-0.5',
                                    np.where(df['avgRiskReward'] <= 1.0, '0.5-1.0', '>1.0')))

# Outcome by time block
print(f"\n=== OUTCOME BY TIME BLOCK ===")
outcome_by_timeblock = df.groupby('time_block')['outcome'].value_counts(normalize=True).unstack()
print(outcome_by_timeblock)

# Outcome by year bucket
print(f"\n=== OUTCOME BY YEAR BUCKET ===")
outcome_by_year = df.groupby('year_bucket')['outcome'].value_counts(normalize=True).unstack()
print(outcome_by_year)

# Outcome by RR bucket
print(f"\n=== OUTCOME BY RR BUCKET ===")
outcome_by_rr = df.groupby('rr_bucket')['outcome'].value_counts(normalize=True).unstack()
print(outcome_by_rr)

# Outcome by side and time block
print(f"\n=== OUTCOME BY SIDE AND TIME BLOCK ===")
outcome_by_side_time = df.groupby(['side', 'time_block'])['outcome'].value_counts(normalize=True).unstack()
print(outcome_by_side_time)

# TP rate by hour
print(f"\n=== TP RATE BY HOUR ===")
tp_by_hour = df.groupby('hour_ny')['outcome'].apply(lambda x: (x == 'TP').mean())
print(tp_by_hour)

# TP rate by weekday
print(f"\n=== TP RATE BY WEEKDAY ===")
tp_by_weekday = df.groupby('weekday_name')['outcome'].apply(lambda x: (x == 'TP').mean())
print(tp_by_weekday)

# TP rate by side
print(f"\n=== TP RATE BY SIDE ===")
tp_by_side = df.groupby('side')['outcome'].apply(lambda x: (x == 'TP').mean())
print(tp_by_side)

# Conditional win rate: TP vs SL only (excluding BE)
df_tp_sl = df[df['outcome'] != 'BE']
print(f"\n=== TP RATE (TP vs SL only) ===")
print(f"Overall TP rate (TP vs SL): {(df_tp_sl['outcome'] == 'TP').mean() * 100:.1f}%")
print(f"Long TP rate: {(df_tp_sl[df_tp_sl['side'] == 'buy']['outcome'] == 'TP').mean() * 100:.1f}%")
print(f"Short TP rate: {(df_tp_sl[df_tp_sl['side'] == 'sell']['outcome'] == 'TP').mean() * 100:.1f}%")

# Analysis of RR distribution by outcome
print(f"\n=== RR DISTRIBUTION BY OUTCOME ===")
print(df.groupby('outcome')['avgRiskReward'].describe())

# Analysis of RR by year
print(f"\n=== AVG RR BY YEAR ===")
print(df.groupby('year')['avgRiskReward'].mean())

# Trade frequency by year
print(f"\n=== TRADE FREQUENCY BY YEAR ===")
print(df.groupby('year').size())

# Save enhanced ledger
df.to_csv('EURUSD_MANUAL_FEATURE_LEDGER.csv', index=False)
print(f"\nSaved enhanced ledger to EURUSD_MANUAL_FEATURE_LEDGER.csv")
