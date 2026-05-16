import pandas as pd
import sys

print("=" * 60)
print("MANUAL ANNOTATION VALIDATION")
print("=" * 60)

# Load working ledger
print("\nLoading working ledger...")
df = pd.read_csv('EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_WORKING.csv')
print(f"Total rows: {len(df)}")

# Define taxonomy from schema
LIQUIDITY_SOURCE_VALID = ['previous_day_high', 'previous_day_low', 'asia_high', 'asia_low', 'london_high', 'london_low', 'none_unclear']
TRIGGER_TYPE_VALID = ['sweep_reclaim', 'sweep_displacement', 'continuation_after_break', 'reversal_after_sweep', 'breakout_from_compression', 'none_unclear']
CONFIRMATION_TYPE_VALID = ['close_back_inside', 'strong_displacement_bar', 'structure_break', 'reclaim_then_go', 'immediate_rejection', 'none_unclear']
OPERATIONAL_CONTEXT_VALID = ['london_open_drive', 'london_continuation', 'london_reversal', 'pre_ny_transition', 'early_ny_followthrough', 'none_unclear']
ENTRY_MOTIVE_VALID = ['liquidity', 'displacement', 'reclaim', 'time_window', 'confluence', 'none_unclear']
QUALITY_RATING_VALID = ['A', 'B', 'C']

# Define field mappings
TAXONOMY = {
    'liquidity_source': LIQUIDITY_SOURCE_VALID,
    'trigger_type': TRIGGER_TYPE_VALID,
    'confirmation_type': CONFIRMATION_TYPE_VALID,
    'operational_context': OPERATIONAL_CONTEXT_VALID,
    'entry_motive': ENTRY_MOTIVE_VALID,
    'quality_rating': QUALITY_RATING_VALID
}

human_fields = list(TAXONOMY.keys()) + ['comment']

# Validation results
errors = []
warnings = []

# Check missing fields
missing_fields = df[human_fields].isna().sum()
for field, count in missing_fields.items():
    if count > 0:
        errors.append(f"Missing {field}: {count}/{len(df)} rows")

# Check taxonomy violations
for field, valid_values in TAXONOMY.items():
    if field in df.columns:
        invalid_mask = df[field].notna() & ~df[field].isin(valid_values)
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            invalid_values = df.loc[invalid_mask, field].unique()
            errors.append(f"Invalid values in {field}: {invalid_values} (found in {invalid_count} rows)")

# Check comment length (optional, max 200 chars)
if 'comment' in df.columns:
    # Ensure comment is treated as string for length check
    df['comment'] = df['comment'].fillna('')
    long_comments = df[df['comment'].str.len() > 200]
    if len(long_comments) > 0:
        warnings.append(f"Comments too long (>200 chars): {len(long_comments)} rows")

# Check consistency: trigger_type should not be none_unclear if confirmation_type is specific
if 'trigger_type' in df.columns and 'confirmation_type' in df.columns:
    inconsistent_mask = (df['trigger_type'] == 'none_unclear') & (df['confirmation_type'].notna()) & (df['confirmation_type'] != 'none_unclear')
    if inconsistent_mask.sum() > 0:
        warnings.append(f"Potential inconsistency: trigger_type=none_unclear but specific confirmation_type in {inconsistent_mask.sum()} rows")

# Update control columns
df['missing_human_fields_count'] = df[human_fields].isna().sum(axis=1)
df['annotation_status'] = 'PENDING'
df.loc[df['missing_human_fields_count'] == 0, 'annotation_status'] = 'READY'

# Save updated ledger
df.to_csv('EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_WORKING.csv', index=False)

# Print results
print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

if errors:
    print("\nERRORS:")
    for error in errors:
        print(f"  [ERROR] {error}")
else:
    print("\nERRORS: None")

if warnings:
    print("\nWARNINGS:")
    for warning in warnings:
        print(f"  [WARNING] {warning}")
else:
    print("\nWARNINGS: None")

# Check readiness
ready_rows = (df['annotation_status'] == 'READY').sum()
print(f"\nReady for analysis: {ready_rows}/{len(df)} rows")

# Final verdict
if errors:
    print("\n" + "=" * 60)
    print("STATUS: NOT_READY_FOR_ANALYSIS")
    print("=" * 60)
    print("Fix errors before running analysis.")
    sys.exit(1)
elif ready_rows < len(df):
    print("\n" + "=" * 60)
    print("STATUS: NOT_READY_FOR_ANALYSIS")
    print("=" * 60)
    print(f"Complete annotations first: {len(df) - ready_rows} rows pending.")
    sys.exit(1)
else:
    print("\n" + "=" * 60)
    print("STATUS: READY_FOR_ANALYSIS")
    print("=" * 60)
    print("All annotations complete and valid. You can run analysis now.")
    sys.exit(0)
