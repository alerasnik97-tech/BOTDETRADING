import pandas as pd

print("=" * 60)
print("FIXING LEDGER DTYPES FOR STRING ANNOTATIONS")
print("=" * 60)

# Load working ledger
WORKING_LEDGER = 'EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_WORKING.csv'
df = pd.read_csv(WORKING_LEDGER)

print(f"\nBefore fix:")
human_fields = ['liquidity_source', 'trigger_type', 'confirmation_type', 
                'operational_context', 'entry_motive', 'quality_rating', 'comment']
print(df[human_fields].dtypes)

# Convert human annotation fields to object dtype (string-compatible)
for field in human_fields:
    df[field] = df[field].astype(object)

print(f"\nAfter fix:")
print(df[human_fields].dtypes)

# Save with corrected dtypes
df.to_csv(WORKING_LEDGER, index=False)
print(f"\nSaved corrected ledger to: {WORKING_LEDGER}")

# Verify by reloading
df_verify = pd.read_csv(WORKING_LEDGER)
print(f"\nVerification after reload:")
print(df_verify[human_fields].dtypes)

print("\n✅ DTYPES FIXED - Human fields now support string values")
