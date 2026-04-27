import pandas as pd
import os

print("=" * 60)
print("FAST SIGNAL STAGE 1 AUDIT")
print("=" * 60)

# Load ledger
print("\nLoading fast signal ledger...")
df = pd.read_csv('EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_LEDGER.csv')
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Check human fields
human_fields = ['liquidity_source', 'trigger_type', 'confirmation_type', 
                'operational_context', 'entry_motive', 'quality_rating', 'comment']
print(f"\nHuman fields completion:")
for field in human_fields:
    filled = df[field].notna().sum()
    print(f"  {field}: {filled}/{len(df)}")

# Check chart files
chart_dir = 'manual_trade_chartpacks/fast_signal'
if os.path.exists(chart_dir):
    files = [f for f in os.listdir(chart_dir) if f.endswith('.png')]
    print(f"\nChart PNG files: {len(files)}")
else:
    print(f"\nChart directory not found: {chart_dir}")

# Check alignment
df_index = pd.read_csv('EURUSD_MANUAL_CHARTPACK_INDEX.csv')
print(f"\nChartpack index rows: {len(df_index)}")

print("\n" + "=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)
