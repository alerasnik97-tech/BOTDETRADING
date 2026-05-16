import pandas as pd
import numpy as np
from pathlib import Path

# Load manual ledger
print("Loading manual ledger...")
df = pd.read_csv('EURUSD_MANUAL_FEATURE_LEDGER.csv')
print(f"Total trades: {len(df)}")
print(f"Date range: {df['dateStart'].min()} to {df['dateStart'].max()}")

# Confirm basic stats
print("\n=== BASIC STATS ===")
print(f"Outcome distribution:\n{df['outcome'].value_counts()}")
print(f"\nSide distribution:\n{df['side'].value_counts()}")
print(f"\nTime block distribution:\n{df['time_block'].value_counts()}")
print(f"\nYear bucket distribution:\n{df['year_bucket'].value_counts()}")

# Select curated sample (60-100 trades max)
print("\n=== SELECTING CURATED SAMPLE ===")

# Strategy: balance across outcome, side, time_block, year_bucket
target_sample_size = 80  # Target 80 trades (between 60-100)

# Group by outcome, side, time_block, year_bucket
grouped = df.groupby(['outcome', 'side', 'time_block', 'year_bucket'])

# Calculate target per group
n_groups = len(grouped)
target_per_group = max(1, target_sample_size // n_groups)
remainder = target_sample_size - (target_per_group * n_groups)

print(f"Number of groups: {n_groups}")
print(f"Target per group: {target_per_group}")
print(f"Remainder: {remainder}")

# Sample from each group
sample_list = []
for name, group in grouped:
    n_samples = target_per_group
    if len(group) < n_samples:
        n_samples = len(group)
    if n_samples > 0:
        sampled = group.sample(n=n_samples, random_state=42)
        sample_list.append(sampled)

# Combine samples
df_sample = pd.concat(sample_list, ignore_index=True)
print(f"\nInitial sample size: {len(df_sample)}")

# If we have more than target, randomly remove
if len(df_sample) > target_sample_size:
    df_sample = df_sample.sample(n=target_sample_size, random_state=42)
    print(f"Reduced to target: {len(df_sample)}")

# If we have less than target, add more from underrepresented groups
if len(df_sample) < target_sample_size:
    needed = target_sample_size - len(df_sample)
    # Get trades not in sample
    df_remaining = df[~df['id'].isin(df_sample['id'])]
    if len(df_remaining) > 0:
        additional = df_remaining.sample(n=min(needed, len(df_remaining)), random_state=42)
        df_sample = pd.concat([df_sample, additional], ignore_index=True)
        print(f"Added additional trades: {len(df_sample)}")

print(f"\nFinal sample size: {len(df_sample)}")

# Check balance
print("\n=== SAMPLE BALANCE ===")
print(f"Outcome distribution:\n{df_sample['outcome'].value_counts()}")
print(f"\nSide distribution:\n{df_sample['side'].value_counts()}")
print(f"\nTime block distribution:\n{df_sample['time_block'].value_counts()}")
print(f"\nYear bucket distribution:\n{df_sample['year_bucket'].value_counts()}")

# Save sample
output_path = 'EURUSD_MANUAL_ANNOTATION_SAMPLE.csv'
df_sample.to_csv(output_path, index=False)
print(f"\nSaved sample to: {output_path}")

# Create index for chart packs
print("\n=== CREATING CHARTPACK INDEX ===")
df_sample['chartpack_filename'] = df_sample['id'].astype(str) + '_chartpack.png'
df_sample['chartpack_path'] = 'manual_trade_chartpacks/' + df_sample['chartpack_filename']

chartpack_index = df_sample[['id', 'dateStart_ny', 'dateEnd_ny', 'outcome', 'side', 'time_block', 
                              'year_bucket', 'entryPrice', 'initalSL', 'maxTP', 'chartpack_filename', 
                              'chartpack_path']].copy()
chartpack_index.to_csv('EURUSD_MANUAL_CHARTPACK_INDEX.csv', index=False)
print(f"Saved chartpack index to: EURUSD_MANUAL_CHARTPACK_INDEX.csv")

print("\n=== SUMMARY ===")
print(f"Total trades in sample: {len(df_sample)}")
print(f"Target range: 60-100 trades")
print(f"Balance achieved across outcome, side, time_block, year_bucket")
print(f"Sample saved to: {output_path}")
print(f"Chartpack index saved to: EURUSD_MANUAL_CHARTPACK_INDEX.csv")
