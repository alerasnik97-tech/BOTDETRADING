import pandas as pd
import numpy as np
from pathlib import Path

# Load curated sample
print("Loading curated sample...")
df_sample = pd.read_csv('EURUSD_MANUAL_ANNOTATION_SAMPLE.csv')
print(f"Sample size: {len(df_sample)}")

# Load market data for autofill
print("\nLoading market data for autofill...")
mid_path = Path('data_precision/dukascopy/EURUSD_M1_MID.csv')
df_mid = pd.read_csv(mid_path)
df_mid['timestamp'] = pd.to_datetime(df_mid['timestamp'], utc=True)
print(f"Market data rows: {len(df_mid)}")

# Convert sample timestamps to UTC for matching
df_sample['dateStart_dt'] = pd.to_datetime(df_sample['dateStart'], utc=True)
df_sample['dateEnd_dt'] = pd.to_datetime(df_sample['dateEnd'], utc=True)

# Function to calculate Asia range
def calculate_asia_range(trade_time, df_mid):
    """Calculate Asia high/low for a given trade time."""
    # Asia window: 19:00-03:00 NY (previous day)
    trade_date = trade_time.date()
    asia_start = pd.Timestamp(trade_date, tz='UTC') - pd.Timedelta(hours=5)  # 19:00 previous day NY
    asia_end = pd.Timestamp(trade_date, tz='UTC') + pd.Timedelta(hours=3)    # 03:00 same day NY
    
    mask = (df_mid['timestamp'] >= asia_start) & (df_mid['timestamp'] <= asia_end)
    asia_data = df_mid[mask]
    
    if len(asia_data) > 0:
        return asia_data['high'].max(), asia_data['low'].min()
    else:
        return np.nan, np.nan

# Function to calculate London range
def calculate_london_range(trade_time, df_mid):
    """Calculate London high/low for a given trade time."""
    # London window: 03:00-07:00 NY
    trade_date = trade_time.date()
    london_start = pd.Timestamp(trade_date, tz='UTC') + pd.Timedelta(hours=3)   # 03:00 NY
    london_end = pd.Timestamp(trade_date, tz='UTC') + pd.Timedelta(hours=7)    # 07:00 NY
    
    mask = (df_mid['timestamp'] >= london_start) & (df_mid['timestamp'] <= london_end)
    london_data = df_mid[mask]
    
    if len(london_data) > 0:
        return london_data['high'].max(), london_data['low'].min()
    else:
        return np.nan, np.nan

# Function to calculate previous day range
def calculate_previous_day_range(trade_time, df_mid):
    """Calculate previous day high/low for a given trade time."""
    # Previous day: 00:00-23:59 previous day
    trade_date = trade_time.date()
    prev_day_start = pd.Timestamp(trade_date, tz='UTC') - pd.Timedelta(days=1)
    prev_day_end = pd.Timestamp(trade_date, tz='UTC')
    
    mask = (df_mid['timestamp'] >= prev_day_start) & (df_mid['timestamp'] < prev_day_end)
    prev_day_data = df_mid[mask]
    
    if len(prev_day_data) > 0:
        return prev_day_data['high'].max(), prev_day_data['low'].min()
    else:
        return np.nan, np.nan

# Function to calculate daily open
def calculate_daily_open(trade_time, df_mid):
    """Calculate daily open for a given trade time."""
    trade_date = trade_time.date()
    day_start = pd.Timestamp(trade_date, tz='UTC')
    day_end = day_start + pd.Timedelta(hours=24)
    
    mask = (df_mid['timestamp'] >= day_start) & (df_mid['timestamp'] < day_end)
    day_data = df_mid[mask]
    
    if len(day_data) > 0:
        return day_data.iloc[0]['open']
    else:
        return np.nan

# Create annotation ledger
print("\n=== CREATING ANNOTATION LEDGER ===")

ledger_data = []
for idx, row in df_sample.iterrows():
    trade_time = row['dateStart_dt']
    entry_price = row['entryPrice']
    
    # Calculate objective features
    asia_high, asia_low = calculate_asia_range(trade_time, df_mid)
    london_high, london_low = calculate_london_range(trade_time, df_mid)
    prev_day_high, prev_day_low = calculate_previous_day_range(trade_time, df_mid)
    daily_open = calculate_daily_open(trade_time, df_mid)
    
    # Check if previous day levels were touched
    prev_day_high_touched = entry_price >= prev_day_high if not np.isnan(prev_day_high) else np.nan
    prev_day_low_touched = entry_price <= prev_day_low if not np.isnan(prev_day_low) else np.nan
    
    # Calculate distance to daily open
    daily_open_distance = entry_price - daily_open if not np.isnan(daily_open) else np.nan
    
    # Determine simple session regime
    hour_ny = row['hour_ny']
    if hour_ny < 3:
        session_regime = 'asia'
    elif hour_ny < 7:
        session_regime = 'london_open'
    elif hour_ny < 11:
        session_regime = 'early_ny'
    else:
        session_regime = 'unknown'
    
    # Build ledger row
    ledger_row = {
        'trade_id': row['id'],
        'timestamp_ny': row['dateStart_ny'],
        'outcome': row['outcome'],
        'side': row['side'],
        'hour_ny': row['hour_ny'],
        'time_block': row['time_block'],
        'year_bucket': row['year_bucket'],
        'entry_price': row['entryPrice'],
        'initial_sl': row['initalSL'],
        'max_tp': row['maxTP'],
        'avg_rr': row['avgRiskReward'],
        'max_rr': row['maxRiskReward'],
        
        # Objective autofill fields
        'asia_high': asia_high,
        'asia_low': asia_low,
        'london_high': london_high,
        'london_low': london_low,
        'prev_day_high': prev_day_high,
        'prev_day_low': prev_day_low,
        'prev_day_high_touched': prev_day_high_touched,
        'prev_day_low_touched': prev_day_low_touched,
        'daily_open': daily_open,
        'daily_open_distance': daily_open_distance,
        'session_regime': session_regime,
        
        # Human annotation fields (empty for now)
        'liquidity_source': '',
        'trigger_type': '',
        'confirmation_type': '',
        'operational_context': '',
        'entry_motive': '',
        'quality_rating': '',
        'comment': ''
    }
    
    ledger_data.append(ledger_row)
    
    if (idx + 1) % 10 == 0:
        print(f"Processed {idx + 1}/{len(df_sample)} trades...")

# Create ledger dataframe
df_ledger = pd.DataFrame(ledger_data)

# Save ledger
ledger_path = 'EURUSD_MANUAL_ANNOTATION_LEDGER.csv'
df_ledger.to_csv(ledger_path, index=False)
print(f"\nSaved annotation ledger to: {ledger_path}")

# Summary
print("\n=== LEDGER SUMMARY ===")
print(f"Total rows: {len(df_ledger)}")
print(f"Columns: {len(df_ledger.columns)}")
print(f"\nObjective fields (autofilled): {sum(1 for c in df_ledger.columns if c in ['asia_high', 'asia_low', 'london_high', 'london_low', 'prev_day_high', 'prev_day_low', 'prev_day_high_touched', 'prev_day_low_touched', 'daily_open', 'daily_open_distance', 'session_regime'])}")
print(f"Human annotation fields (empty): {sum(1 for c in df_ledger.columns if c in ['liquidity_source', 'trigger_type', 'confirmation_type', 'operational_context', 'entry_motive', 'quality_rating', 'comment'])}")

print("\n=== READY FOR HUMAN ANNOTATION ===")
print("The ledger is ready for the user to complete the human annotation fields.")
print("All objective features have been pre-filled automatically.")
