import pandas as pd
import os
from datetime import datetime

print("=" * 60)
print("SMOKE TEST - PERSISTENCE WITH FIX")
print("=" * 60)

# Load the working ledger with the fix (convert to object dtype)
WORKING_LEDGER = 'EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_WORKING.csv'
df = pd.read_csv(WORKING_LEDGER)

# Apply the fix: convert human fields to object dtype
human_fields = ['liquidity_source', 'trigger_type', 'confirmation_type', 
                'operational_context', 'entry_motive', 'quality_rating', 'comment']
for field in human_fields:
    df[field] = df[field].astype(object)

print(f"\nBefore modification:")
print(f"  Row 0 liquidity_source: {df.iloc[0]['liquidity_source']}")
print(f"  Row 0 annotation_status: {df.iloc[0]['annotation_status']}")

# Modify row 0 with a canary value
df.iloc[0, df.columns.get_loc('liquidity_source')] = 'none_unclear'
df.iloc[0, df.columns.get_loc('trigger_type')] = 'none_unclear'
df.iloc[0, df.columns.get_loc('confirmation_type')] = 'none_unclear'
df.iloc[0, df.columns.get_loc('operational_context')] = 'none_unclear'
df.iloc[0, df.columns.get_loc('entry_motive')] = 'none_unclear'
df.iloc[0, df.columns.get_loc('quality_rating')] = 'C'
df.iloc[0, df.columns.get_loc('comment')] = 'CANARY_TEST'

# Recalculate status
missing_count = 0
df.iloc[0, df.columns.get_loc('missing_human_fields_count')] = missing_count
df.iloc[0, df.columns.get_loc('annotation_status')] = 'READY'

print(f"\nAfter modification (in memory):")
print(f"  Row 0 liquidity_source: {df.iloc[0]['liquidity_source']}")
print(f"  Row 0 annotation_status: {df.iloc[0]['annotation_status']}")

# Save
print(f"\nSaving to: {os.path.abspath(WORKING_LEDGER)}")
try:
    df.to_csv(WORKING_LEDGER, index=False)
    print("Save completed without error")
except Exception as e:
    print(f"Save failed with error: {e}")
    exit(1)

# Get timestamp after save
mtime_after = datetime.fromtimestamp(os.path.getmtime(WORKING_LEDGER)).strftime("%Y-%m-%d %H:%M:%S")
print(f"File timestamp after save: {mtime_after}")

# Verify save by reloading
df_reloaded = pd.read_csv(WORKING_LEDGER)
# Apply fix to reloaded data
for field in human_fields:
    df_reloaded[field] = df_reloaded[field].astype(object)

print(f"\nAfter reload from disk:")
print(f"  Row 0 liquidity_source: {df_reloaded.iloc[0]['liquidity_source']}")
print(f"  Row 0 annotation_status: {df_reloaded.iloc[0]['annotation_status']}")

# Check if persistence worked
if df_reloaded.iloc[0]['liquidity_source'] == 'none_unclear' and df_reloaded.iloc[0]['annotation_status'] == 'READY':
    print("\nSUCCESS: PERSISTENCE TEST PASSED")
else:
    print("\nFAILURE: PERSISTENCE TEST FAILED")
    exit(1)

# Restore original state
df_reloaded.iloc[0, df_reloaded.columns.get_loc('liquidity_source')] = None
df_reloaded.iloc[0, df_reloaded.columns.get_loc('trigger_type')] = None
df_reloaded.iloc[0, df_reloaded.columns.get_loc('confirmation_type')] = None
df_reloaded.iloc[0, df_reloaded.columns.get_loc('operational_context')] = None
df_reloaded.iloc[0, df_reloaded.columns.get_loc('entry_motive')] = None
df_reloaded.iloc[0, df_reloaded.columns.get_loc('quality_rating')] = None
df_reloaded.iloc[0, df_reloaded.columns.get_loc('comment')] = None
df_reloaded.iloc[0, df_reloaded.columns.get_loc('missing_human_fields_count')] = 7
df_reloaded.iloc[0, df_reloaded.columns.get_loc('annotation_status')] = 'PENDING'
df_reloaded.to_csv(WORKING_LEDGER, index=False)
print("Restored original state")
