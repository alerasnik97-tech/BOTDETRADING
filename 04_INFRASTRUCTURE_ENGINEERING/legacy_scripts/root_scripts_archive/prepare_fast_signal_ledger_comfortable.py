import pandas as pd

print("Loading fast signal ledger...")
df = pd.read_csv('EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_LEDGER.csv')

# Reorder columns for comfort: identification + context first, human fields last
column_order = [
    # Identification
    'trade_id',
    'timestamp_ny',
    'rank',  # Will add this
    
    # Context (objective, autofilled)
    'outcome',
    'side',
    'hour_ny',
    'time_block',
    'year_bucket',
    'entry_price',
    'initial_sl',
    'max_tp',
    'avg_rr',
    'max_rr',
    'asia_high',
    'asia_low',
    'london_high',
    'london_low',
    'prev_day_high',
    'prev_day_low',
    'prev_day_high_touched',
    'prev_day_low_touched',
    'daily_open',
    'daily_open_distance',
    'session_regime',
    
    # Human annotation fields (THESE ARE WHAT YOU COMPLETE)
    'liquidity_source',
    'trigger_type',
    'confirmation_type',
    'operational_context',
    'entry_motive',
    'quality_rating',
    'comment',
    
    # Meta
    'priority_tier'
]

# Add rank column (1-25)
df['rank'] = range(1, len(df) + 1)

# Reorder columns
df = df[column_order]

# Save comfortable ledger
df.to_csv('EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_LEDGER.csv', index=False)
print("Saved comfortable ledger to: EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_LEDGER.csv")

print("\nColumn order:")
for i, col in enumerate(column_order, 1):
    print(f"{i}. {col}")

print("\n" + "=" * 60)
print("LEDGER READY FOR ANNOTATION")
print("=" * 60)
print("Open this file in Excel/Sheets:")
print("EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_LEDGER.csv")
print("\nComplete ONLY these 7 human fields:")
print("1. liquidity_source")
print("2. trigger_type")
print("3. confirmation_type")
print("4. operational_context")
print("5. entry_motive")
print("6. quality_rating")
print("7. comment")
print("\nAll other fields are already pre-filled.")
