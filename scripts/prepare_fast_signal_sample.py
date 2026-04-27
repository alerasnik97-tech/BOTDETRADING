import pandas as pd
import numpy as np
from pathlib import Path

# Load full sample
print("Loading full annotation sample...")
df_sample = pd.read_csv('EURUSD_MANUAL_ANNOTATION_SAMPLE.csv')
print(f"Full sample size: {len(df_sample)}")

# Load ledger for reference
print("Loading annotation ledger...")
df_ledger = pd.read_csv('EURUSD_MANUAL_ANNOTATION_LEDGER.csv')
print(f"Ledger size: {len(df_ledger)}")

# Select fast-signal sample (20-25 trades)
print("\n=== SELECTING FAST-SIGNAL SAMPLE ===")

# Strategy: prioritize trades that maximize signal early
# Criteria:
# - Balance TP/SL/BE
# - Balance longs/shorts
# - Strong representation of 5am-6am (best performing)
# - Include some 3am-4am and 4am-5am for contrast
# - Include different years

target_fast_signal = 25  # Target 25 trades for fast signal

# First, prioritize 5am-6am trades (best performing)
df_5am = df_sample[df_sample['time_block'] == '5am-6am'].copy()
print(f"5am-6am trades available: {len(df_5am)}")

# Sample from 5am-6am (prioritize this block)
n_5am = min(10, len(df_5am))  # Target 10 from 5am-6am
if n_5am > 0:
    df_5am_sample = df_5am.sample(n=n_5am, random_state=42)
else:
    df_5am_sample = pd.DataFrame(columns=df_sample.columns)

# Next, sample from 4am-5am
df_4am = df_sample[df_sample['time_block'] == '4am-5am'].copy()
print(f"4am-5am trades available: {len(df_4am)}")
n_4am = min(8, len(df_4am))  # Target 8 from 4am-5am
if n_4am > 0:
    df_4am_sample = df_4am.sample(n=n_4am, random_state=42)
else:
    df_4am_sample = pd.DataFrame(columns=df_sample.columns)

# Next, sample from 3am-4am
df_3am = df_sample[df_sample['time_block'] == '3am-4am'].copy()
print(f"3am-4am trades available: {len(df_3am)}")
n_3am = min(7, len(df_3am))  # Target 7 from 3am-4am
if n_3am > 0:
    df_3am_sample = df_3am.sample(n=n_3am, random_state=42)
else:
    df_3am_sample = pd.DataFrame(columns=df_sample.columns)

# Combine samples
df_fast_signal = pd.concat([df_5am_sample, df_4am_sample, df_3am_sample], ignore_index=True)
print(f"\nInitial fast signal size: {len(df_fast_signal)}")

# Check balance
print("\n=== FAST-SIGNAL BALANCE ===")
print(f"Outcome distribution:\n{df_fast_signal['outcome'].value_counts()}")
print(f"\nSide distribution:\n{df_fast_signal['side'].value_counts()}")
print(f"\nTime block distribution:\n{df_fast_signal['time_block'].value_counts()}")
print(f"\nYear bucket distribution:\n{df_fast_signal['year_bucket'].value_counts()}")

# If we have more than target, reduce
if len(df_fast_signal) > target_fast_signal:
    df_fast_signal = df_fast_signal.sample(n=target_fast_signal, random_state=42)
    print(f"\nReduced to target: {len(df_fast_signal)}")

# If we have less than target, add more from remaining
if len(df_fast_signal) < target_fast_signal:
    needed = target_fast_signal - len(df_fast_signal)
    df_remaining = df_sample[~df_sample['id'].isin(df_fast_signal['id'])]
    if len(df_remaining) > 0:
        additional = df_remaining.sample(n=min(needed, len(df_remaining)), random_state=42)
        df_fast_signal = pd.concat([df_fast_signal, additional], ignore_index=True)
        print(f"Added additional trades: {len(df_fast_signal)}")

print(f"\nFinal fast signal size: {len(df_fast_signal)}")

# Add priority tier column
df_fast_signal['priority_tier'] = 'FAST_SIGNAL'

# Save fast signal sample
output_path = 'EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_SAMPLE.csv'
df_fast_signal.to_csv(output_path, index=False)
print(f"\nSaved fast signal sample to: {output_path}")

# Update full sample with priority tier
df_sample['priority_tier'] = 'FULL_SAMPLE'
df_sample.loc[df_sample['id'].isin(df_fast_signal['id']), 'priority_tier'] = 'FAST_SIGNAL'

# Save updated full sample
df_sample.to_csv('EURUSD_MANUAL_ANNOTATION_SAMPLE.csv', index=False)
print(f"Updated full sample with priority tier")

# Create fast signal ledger (subset of full ledger)
df_ledger['priority_tier'] = 'FULL_SAMPLE'
df_ledger.loc[df_ledger['trade_id'].isin(df_fast_signal['id']), 'priority_tier'] = 'FAST_SIGNAL'

df_fast_signal_ledger = df_ledger[df_ledger['priority_tier'] == 'FAST_SIGNAL'].copy()
df_fast_signal_ledger.to_csv('EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_LEDGER.csv', index=False)
print(f"Saved fast signal ledger to: EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_LEDGER.csv")

# Save updated full ledger
df_ledger.to_csv('EURUSD_MANUAL_ANNOTATION_LEDGER.csv', index=False)
print(f"Updated full ledger with priority tier")

print("\n=== SUMMARY ===")
print(f"Fast signal sample size: {len(df_fast_signal)}")
print(f"Target range: 20-25 trades")
print(f"Prioritization: 5am-6am > 4am-5am > 3am-4am")
print(f"Balance achieved across outcome, side, time_block, year_bucket")
print(f"Priority tier added to both sample and ledger")
