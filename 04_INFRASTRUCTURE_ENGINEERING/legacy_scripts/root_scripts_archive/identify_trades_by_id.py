import pandas as pd

print("=" * 60)
print("IDENTIFICACIÓN DE TRADES POR ID")
print("=" * 60)

# Muestra curada total
sample = pd.read_csv('EURUSD_MANUAL_ANNOTATION_SAMPLE.csv')
print("MUESTRA CURADA TOTAL:")
print(f"  Shape: {sample.shape}")
print(f"  IDs únicos: {sample['id'].nunique()}")

# Fast signal sample
fast = pd.read_csv('EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_SAMPLE.csv')
print("\nFAST SIGNAL SAMPLE:")
print(f"  Shape: {fast.shape}")
print(f"  IDs únicos: {fast['id'].nunique()}")

# Verificar usando id
print("\nUSANDO ID PARA IDENTIFICACIÓN:")
sample_ids = set(sample['id'].tolist())
fast_ids = set(fast['id'].tolist())
print(f"  Fast IDs en Sample: {len(fast_ids & sample_ids)}/{len(fast_ids)}")
print(f"  Fast IDs NO en Sample: {len(fast_ids - sample_ids)}")
print(f"  Sample IDs NO en Fast: {len(sample_ids - fast_ids)}")

remaining_ids = sample_ids - fast_ids
print(f"\nTRADES RESTANTES PARA ETAPA 2: {len(remaining_ids)}")
print(f"  Remaining IDs (primeros 10): {sorted(list(remaining_ids))[:10]}")

# Extraer los 55 trades restantes
remaining_trades = sample[sample['id'].isin(remaining_ids)]
print(f"\nDATAFRAME DE TRADES RESTANTES:")
print(f"  Shape: {remaining_trades.shape}")
print(f"  time_block values: {remaining_trades['time_block'].value_counts().to_dict()}")
