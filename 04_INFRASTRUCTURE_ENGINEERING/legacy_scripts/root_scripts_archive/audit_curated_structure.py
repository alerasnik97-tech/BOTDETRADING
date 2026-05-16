import pandas as pd

print("=" * 60)
print("AUDITORÍA DE ESTRUCTURA DE MUESTRA CURADA")
print("=" * 60)

# Muestra curada total
sample = pd.read_csv('EURUSD_MANUAL_ANNOTATION_SAMPLE.csv')
print("MUESTRA CURADA TOTAL (80 trades):")
print(f"  Shape: {sample.shape}")
print(f"  Filas: {len(sample)}")
print(f"  Columnas: {len(sample.columns)}")
if 'time_block' in sample.columns:
    print(f"  time_block values: {sample['time_block'].value_counts().to_dict()}")

# Fast signal sample (ETAPA 1)
fast = pd.read_csv('EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_SAMPLE.csv')
print("\nFAST SIGNAL SAMPLE (ETAPA 1 - 25 trades):")
print(f"  Shape: {fast.shape}")
print(f"  Filas: {len(fast)}")
if 'time_block' in fast.columns:
    print(f"  time_block values: {fast['time_block'].value_counts().to_dict()}")

# Ledger post-ETAPA 1
ledger = pd.read_csv('EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_WORKING.csv')
print("\nLEDGER POST-ETAPA 1:")
print(f"  Shape: {ledger.shape}")
print(f"  Filas: {len(ledger)}")

# Verificar si fast_signal es subconjunto de sample
print("\nVERIFICACIÓN DE SUBCONJUNTO:")
if 'trade_id' in sample.columns and 'trade_id' in fast.columns:
    sample_ids = set(sample['trade_id'].tolist())
    fast_ids = set(fast['trade_id'].tolist())
    print(f"  Fast IDs en Sample: {len(fast_ids & sample_ids)}/{len(fast_ids)}")
    print(f"  Fast IDs NO en Sample: {len(fast_ids - sample_ids)}")
    print(f"  Sample IDs NO en Fast: {len(sample_ids - fast_ids)}")
    
    # Identificar los 55 trades restantes
    remaining_ids = sample_ids - fast_ids
    print(f"\nTRADES RESTANTES PARA ETAPA 2: {len(remaining_ids)}")
