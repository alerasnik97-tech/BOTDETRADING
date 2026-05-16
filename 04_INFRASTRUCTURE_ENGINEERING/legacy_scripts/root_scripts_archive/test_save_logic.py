import pandas as pd
import os
from datetime import datetime

print("=" * 60)
print("TEST SAVE LOGIC - ISOLATED")
print("=" * 60)

# Load the working ledger
WORKING_LEDGER = 'EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_WORKING.csv'
df = pd.read_csv(WORKING_LEDGER)

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
df.to_csv(WORKING_LEDGER, index=False)

# Verify save by reloading
df_reloaded = pd.read_csv(WORKING_LEDGER)
print(f"\nAfter reload from disk:")
print(f"  Row 0 liquidity_source: {df_reloaded.iloc[0]['liquidity_source']}")
print(f"  Row 0 annotation_status: {df_reloaded.iloc[0]['annotation_status']}")

# Check if persistence worked
if df_reloaded.iloc[0]['liquidity_source'] == 'none_unclear' and df_reloaded.iloc[0]['annotation_status'] == 'READY':
    print("\n✅ PERSISTENCE TEST PASSED")
else:
    print("\n❌ PERSISTENCE TEST FAILED")

# Restore original state
df.iloc[0, df.columns.get_loc('liquidity_source')] = None
df.iloc[0, df.columns.get_loc('trigger_type')] = None
df.iloc[0, df.columns.get_loc('confirmation_type')] = None
df.iloc[0, df.columns.get_loc('operational_context')] = None
df.iloc[0, df.columns.get_loc('entry_motive')] = None
df.iloc[0, df.columns.get_loc('quality_rating')] = None
df.iloc[0, df.columns.get_loc('comment')] = None
df.iloc[0, df.columns.get_loc('missing_human_fields_count')] = 7
df.iloc[0, df.columns.get_loc('annotation_status')] = 'PENDING'
df.to_csv(WORKING_LEDGER, index=False)
print("Restored original state")
