import pandas as pd
import os

print("=" * 60)
print("AUDITORÍA DE ESTRUCTURA - CSV DE CHATGPT")
print("=" * 60)

path = 'EURUSD_FAST_SIGNAL_25_ANNOTATIONS_BY_CHATGPT.csv'
print(f"ARCHIVO: {path}")
print(f"TAMANO: {os.path.getsize(path)} bytes")
print()

df = pd.read_csv(path)
print(f"SHAPE: {df.shape}")
print(f"FILAS: {len(df)}")
print(f"COLUMNAS: {len(df.columns)}")
print()
print("LISTA DE COLUMNAS:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")
print()
print("DTYPES:")
print(df.dtypes)
print()
print("HUMAN FIELDS CHECK:")
human_fields = ['liquidity_source', 'trigger_type', 'confirmation_type', 
                'operational_context', 'entry_motive', 'quality_rating', 'comment']
for field in human_fields:
    if field in df.columns:
        filled = df[field].notna().sum()
        print(f"  {field}: {filled}/{len(df)} filled")
    else:
        print(f"  {field}: MISSING FROM CSV")
print()
print("SAMPLE ROWS:")
print(df.head(3))
