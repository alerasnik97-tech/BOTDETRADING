import pandas as pd

print("=" * 60)
print("VERIFICACIÓN DE ESTRUCTURA DE MUESTRAS")
print("=" * 60)

# Muestra curada total
sample = pd.read_csv('EURUSD_MANUAL_ANNOTATION_SAMPLE.csv')
print("MUESTRA CURADA TOTAL:")
print(f"  Shape: {sample.shape}")
print(f"  Filas: {len(sample)}")
print(f"  Columnas: {len(sample.columns)}")
if 'time_block' in sample.columns:
    print(f"  time_block values: {sample['time_block'].value_counts().to_dict()}")

# Fast signal sample
fast = pd.read_csv('EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_SAMPLE.csv')
print("\nFAST SIGNAL SAMPLE:")
print(f"  Shape: {fast.shape}")
print(f"  Filas: {len(fast)}")
print(f"  Columnas: {len(fast.columns)}")
if 'time_block' in fast.columns:
    print(f"  time_block values: {fast['time_block'].value_counts().to_dict()}")

# Verificar si fast_signal es subconjunto de sample
print("\nVERIFICACIÓN DE SUBCONJUNTO:")
if 'trade_id' in sample.columns and 'trade_id' in fast.columns:
    sample_ids = set(sample['trade_id'].tolist())
    fast_ids = set(fast['trade_id'].tolist())
    print(f"  Fast IDs en Sample: {len(fast_ids & sample_ids)}/{len(fast_ids)}")
    print(f"  Fast IDs NO en Sample: {len(fast_ids - sample_ids)}")
    print(f"  Sample IDs NO en Fast: {len(sample_ids - fast_ids)}")
