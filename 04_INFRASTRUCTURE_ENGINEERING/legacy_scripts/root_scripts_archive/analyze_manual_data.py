import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load manual data
df = pd.read_csv('DATA MANUAL/analytics (1).csv', header=None, names=[
    'id', 'dateStart', 'dateEnd', 'pair', 'uPnL', 'rPnL', 'side', 
    'entryPrice', 'initalSL', 'maxTP', 'idealTP', 'amount', 'amountClosed', 
    'status', 'day', 'tags', 'avgClosePrice', 'avgRiskReward', 'maxRiskReward', 
    'exchangeRate', 'initialBalance', 'currentRealizedBalance'
])

print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Filter EURUSD only
df_eur = df[df['pair'].str.contains('EURUSD', na=False)].copy()
print(f"\nEURUSD rows: {len(df_eur)}")

# Parse timestamps (assume UTC, convert to NY by subtracting 5 hours for EST, 4 for EDT)
df_eur['dateStart'] = pd.to_datetime(df_eur['dateStart'])
df_eur['dateEnd'] = pd.to_datetime(df_eur['dateEnd'])

# Simple NY conversion (UTC-5 for EST, UTC-4 for EDT - using UTC-5 as approximation)
df_eur['dateStart_ny'] = df_eur['dateStart'] - pd.Timedelta(hours=5)
df_eur['dateEnd_ny'] = df_eur['dateEnd'] - pd.Timedelta(hours=5)

# Extract temporal features
df_eur['year'] = df_eur['dateStart_ny'].dt.year
df_eur['month'] = df_eur['dateStart_ny'].dt.month
df_eur['weekday'] = df_eur['dateStart_ny'].dt.dayofweek
df_eur['hour_ny'] = df_eur['dateStart_ny'].dt.hour
df_eur['minute_ny'] = df_eur['dateStart_ny'].dt.minute
df_eur['time_decimal'] = df_eur['hour_ny'] + df_eur['minute_ny'] / 60

# Convert rPnL to numeric
df_eur['rPnL'] = pd.to_numeric(df_eur['rPnL'], errors='coerce')
df_eur['uPnL'] = pd.to_numeric(df_eur['uPnL'], errors='coerce')
df_eur['avgRiskReward'] = pd.to_numeric(df_eur['avgRiskReward'], errors='coerce')
df_eur['maxRiskReward'] = pd.to_numeric(df_eur['maxRiskReward'], errors='coerce')

# Define outcome
# rPnL > 0: TP, rPnL == 0: BE, rPnL < 0: SL
df_eur['outcome'] = np.where(df_eur['rPnL'] > 0, 'TP',
                            np.where(df_eur['rPnL'] == 0, 'BE', 'SL'))

# Basic statistics
print(f"\n=== BASIC STATISTICS ===")
print(f"Date range: {df_eur['dateStart_ny'].min()} to {df_eur['dateStart_ny'].max()}")
print(f"Years: {sorted(df_eur['year'].unique())}")
print(f"\nOutcome distribution:")
print(df_eur['outcome'].value_counts())
print(f"\nWin rate: {(df_eur['outcome'] == 'TP').sum() / len(df_eur) * 100:.1f}%")

print(f"\n=== SIDE DISTRIBUTION ===")
print(df_eur['side'].value_counts())

print(f"\n=== TIME DISTRIBUTION (NY Hour) ===")
print(df_eur['hour_ny'].value_counts().sort_index())

print(f"\n=== WEEKDAY DISTRIBUTION ===")
weekday_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
df_eur['weekday_name'] = df_eur['weekday'].map(weekday_names)
print(df_eur['weekday_name'].value_counts())

print(f"\n=== RISK/REWARD DISTRIBUTION ===")
print(f"Avg RiskReward mean: {df_eur['avgRiskReward'].mean():.2f}")
print(f"Avg RiskReward median: {df_eur['avgRiskReward'].median():.2f}")
print(f"Max RiskReward mean: {df_eur['maxRiskReward'].mean():.2f}")
print(f"Max RiskReward median: {df_eur['maxRiskReward'].median():.2f}")

print(f"\n=== OUTCOME BY HOUR ===")
outcome_by_hour = df_eur.groupby('hour_ny')['outcome'].value_counts(normalize=True).unstack()
print(outcome_by_hour)

print(f"\n=== OUTCOME BY WEEKDAY ===")
outcome_by_weekday = df_eur.groupby('weekday_name')['outcome'].value_counts(normalize=True).unstack()
print(outcome_by_weekday)

print(f"\n=== OUTCOME BY SIDE ===")
outcome_by_side = df_eur.groupby('side')['outcome'].value_counts(normalize=True).unstack()
print(outcome_by_side)

# Save normalized ledger
df_eur.to_csv('EURUSD_MANUAL_FEATURE_LEDGER.csv', index=False)
print(f"\nSaved normalized ledger to EURUSD_MANUAL_FEATURE_LEDGER.csv")
