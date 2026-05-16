import os
import pandas as pd
from datetime import datetime
import hashlib

print("=" * 60)
print("PERSISTENCE FILE AUDIT")
print("=" * 60)

files = [
    'EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_WORKING.csv',
    'EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_LEDGER.csv'
]

print("\nFILE AUDIT:")
for f in files:
    if os.path.exists(f):
        size = os.path.getsize(f)
        mtime = datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{f}:")
        print(f"  EXISTS: YES")
        print(f"  SIZE: {size} bytes")
        print(f"  MODIFIED: {mtime}")
        
        # Read content to check for human annotations
        df = pd.read_csv(f)
        human_fields = ['liquidity_source', 'trigger_type', 'confirmation_type', 
                        'operational_context', 'entry_motive', 'quality_rating', 'comment']
        filled = df[human_fields].notna().sum().sum()
        total_cells = len(df) * len(human_fields)
        print(f"  HUMAN ANNOTATIONS: {filled}/{total_cells} cells filled")
        
        # Simple fingerprint
        content_hash = hashlib.md5(open(f, 'rb').read()).hexdigest()[:8]
        print(f"  HASH: {content_hash}")
    else:
        print(f"{f}: EXISTS: NO")

print("\n" + "=" * 60)
print("CURRENT WORKING DIRECTORY")
print("=" * 60)
print(f"CWD: {os.getcwd()}")
print(f"ABSOLUTE PATH TO WORKING LEDGER: {os.path.abspath('EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_WORKING.csv')}")
