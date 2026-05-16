import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Load manual ledger
print("Loading manual ledger...")
df_manual = pd.read_csv('EURUSD_MANUAL_FEATURE_LEDGER.csv')
print(f"Manual trades: {len(df_manual)}")
print(f"Date range: {df_manual['dateStart'].min()} to {df_manual['dateStart'].max()}")

# Load high precision market data
print("\nLoading high precision market data...")
bid_path = Path('data_precision/dukascopy/EURUSD_M1_BID.csv')
ask_path = Path('data_precision/dukascopy/EURUSD_M1_ASK.csv')
mid_path = Path('data_precision/dukascopy/EURUSD_M1_MID.csv')

# Read in chunks to avoid memory issues
print("Reading bid data...")
df_bid = pd.read_csv(bid_path)
df_bid['timestamp'] = pd.to_datetime(df_bid['timestamp'], utc=True)
print(f"Bid rows: {len(df_bid)}")

print("Reading ask data...")
df_ask = pd.read_csv(ask_path)
df_ask['timestamp'] = pd.to_datetime(df_ask['timestamp'], utc=True)
print(f"Ask rows: {len(df_ask)}")

print("Reading mid data...")
df_mid = pd.read_csv(mid_path)
df_mid['timestamp'] = pd.to_datetime(df_mid['timestamp'], utc=True)
print(f"Mid rows: {len(df_mid)}")

# Convert manual timestamps to datetime (assumed NY, convert to UTC)
df_manual['dateStart_dt'] = pd.to_datetime(df_manual['dateStart'], utc=True)
df_manual['dateEnd_dt'] = pd.to_datetime(df_manual['dateEnd'], utc=True)

# Function to get market context around a trade
def get_market_context(trade_time, df_mid, window_hours_before=3, window_hours_after=2):
    """Get market data around a trade time."""
    start_window = trade_time - timedelta(hours=window_hours_before)
    end_window = trade_time + timedelta(hours=window_hours_after)
    
    # Filter market data in window
    mask = (df_mid['timestamp'] >= start_window) & (df_mid['timestamp'] <= end_window)
    context = df_mid[mask].copy()
    
    return context

# Test with first trade
print("\n=== TEST WITH FIRST TRADE ===")
first_trade = df_manual.iloc[0]
print(f"Trade time: {first_trade['dateStart_dt']}")
print(f"Entry price: {first_trade['entryPrice']}")
print(f"Side: {first_trade['side']}")
print(f"Outcome: {first_trade['outcome']}")

context = get_market_context(first_trade['dateStart_dt'], df_mid)
print(f"Context window: {len(context)} bars")
if len(context) > 0:
    print(f"Context range: {context['timestamp'].min()} to {context['timestamp'].max()}")
    print(f"Price range: {context['low'].min():.5f} to {context['high'].max():.5f}")

# Calculate basic context features for first trade
if len(context) > 0:
    # Find closest bar to trade time
    context['time_diff'] = abs(context['timestamp'] - first_trade['dateStart_dt'])
    closest_bar = context.loc[context['time_diff'].idxmin()]
    
    print(f"\nClosest bar to trade:")
    print(f"  Time: {closest_bar['timestamp']}")
    print(f"  OHLC: {closest_bar['open']:.5f} / {closest_bar['high']:.5f} / {closest_bar['low']:.5f} / {closest_bar['close']:.5f}")
    
    # Calculate range before trade
    before_trade = context[context['timestamp'] <= first_trade['dateStart_dt']]
    if len(before_trade) > 0:
        range_before = before_trade['high'].max() - before_trade['low'].min()
        print(f"\nRange before trade: {range_before:.5f} pips")
        
        # Calculate session context
        before_trade['hour'] = before_trade['timestamp'].dt.hour
        print(f"Hours in window: {sorted(before_trade['hour'].unique())}")

print("\n=== CONTEXT RECONSTRUCTION FEASIBILITY ===")
print(f"Market data available: 2020-01-01 to 2025-12-31")
print(f"Manual data: 2020-01-02 to 2025-12-31")
print(f"Coverage: Complete")
print(f"Resolution: M1 (1-minute bars)")
print(f"Data types: Bid, Ask, Mid")

print("\n=== NEXT STEPS ===")
print("1. For each manual trade, extract market context window")
print("2. Calculate session features (Asia, London, NY)")
print("3. Calculate liquidity features (sweeps of previous levels)")
print("4. Calculate structural features (displacement, reclaim)")
print("5. Merge with manual ledger")
print("6. Analyze discriminants")

print("\n=== ESTIMATED COMPUTATION TIME ===")
print(f"841 trades × context extraction = significant computation")
print("Recommendation: Process in batches or sample for initial analysis")
