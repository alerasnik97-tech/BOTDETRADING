import pandas as pd

print("=" * 60)
print("COMPARACIÓN - CHATGPT CSV VS LEDGER OFICIAL")
print("=" * 60)

chatgpt_df = pd.read_csv('EURUSD_FAST_SIGNAL_25_ANNOTATIONS_BY_CHATGPT.csv')
ledger_df = pd.read_csv('EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_WORKING.csv')

print("CHATGPT CSV:")
print(f"  Shape: {chatgpt_df.shape}")
print(f"  Columnas ({len(chatgpt_df.columns)}):")
for col in chatgpt_df.columns:
    print(f"    - {col}")

print()
print("LEDGER OFICIAL:")
print(f"  Shape: {ledger_df.shape}")
print(f"  Columnas ({len(ledger_df.columns)}):")
for col in ledger_df.columns:
    print(f"    - {col}")

print()
print("COLUMNAS COMUNES:")
common_cols = set(chatgpt_df.columns) & set(ledger_df.columns)
print(f"  {len(common_cols)} columnas en común:")
for col in sorted(common_cols):
    print(f"    - {col}")

print()
print("COLUMNAS SOLO EN CHATGPT:")
chatgpt_only = set(chatgpt_df.columns) - set(ledger_df.columns)
print(f"  {len(chatgpt_only)} columnas solo en ChatGPT:")
for col in sorted(chatgpt_only):
    print(f"    - {col}")

print()
print("COLUMNAS SOLO EN LEDGER:")
ledger_only = set(ledger_df.columns) - set(chatgpt_df.columns)
print(f"  {len(ledger_only)} columnas solo en Ledger:")
for col in sorted(ledger_only):
    print(f"    - {col}")

print()
print("MATCHING DE FILAS POR RANK:")
print(f"  ChatGPT ranks: {sorted(chatgpt_df['rank'].tolist())}")
print(f"  Ledger ranks: {sorted(ledger_df['rank'].tolist())}")
print(f"  Coinciden: {set(chatgpt_df['rank']) == set(ledger_df['rank'])}")
