import pandas as pd

print("Loading fast signal ledger...")
df = pd.read_csv('EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_LEDGER.csv')
print(f"Total rows: {len(df)}")

# Define column order for working ledger
column_order = [
    # Identification
    'rank',
    'trade_id',
    'timestamp_ny',
    
    # Context (objective, pre-filled)
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
    
    # Session levels (objective)
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
    
    # === HUMAN ANNOTATION FIELDS (COMPLETE THESE) ===
    'liquidity_source',
    'trigger_type',
    'confirmation_type',
    'operational_context',
    'entry_motive',
    'quality_rating',
    'comment',
    
    # Control columns (auto-calculated)
    'missing_human_fields_count',
    'annotation_status',
    
    # Meta
    'priority_tier'
]

# Calculate missing fields count
human_fields = ['liquidity_source', 'trigger_type', 'confirmation_type', 
                'operational_context', 'entry_motive', 'quality_rating', 'comment']
df['missing_human_fields_count'] = df[human_fields].isna().sum(axis=1)

# Set annotation status
df['annotation_status'] = 'PENDING'
df.loc[df['missing_human_fields_count'] == 0, 'annotation_status'] = 'READY'

# Reorder columns
df = df[column_order]

# Save working ledger
df.to_csv('EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_WORKING.csv', index=False)
print("Saved working ledger to: EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_WORKING.csv")

print("\n" + "=" * 60)
print("WORKING LEDGER READY")
print("=" * 60)
print("Open this file in Excel/Sheets:")
print("EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_WORKING.csv")
print("\nComplete ONLY the 7 human annotation fields:")
print("1. liquidity_source")
print("2. trigger_type")
print("3. confirmation_type")
print("4. operational_context")
print("5. entry_motive")
print("6. quality_rating")
print("7. comment")
print("\nControl columns (missing_human_fields_count, annotation_status)")
print("will auto-update when you run validation.")
