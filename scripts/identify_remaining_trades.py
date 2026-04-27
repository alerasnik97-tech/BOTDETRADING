import pandas as pd

print("=" * 60)
print("IDENTIFICACIÓN DE TRADES RESTANTES PARA ETAPA 2")
print("=" * 60)

# Muestra curada total
sample = pd.read_csv('EURUSD_MANUAL_ANNOTATION_SAMPLE.csv')
print("COLUMNAS EN MUESTRA CURADA:")
print(list(sample.columns))

# Fast signal sample
fast = pd.read_csv('EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_SAMPLE.csv')
print("\nCOLUMNAS EN FAST SIGNAL:")
print(list(fast.columns))

# Verificar si hay rank
if 'rank' in sample.columns and 'rank' in fast.columns:
    print("\nUSANDO RANK PARA IDENTIFICACIÓN:")
    sample_ranks = set(sample['rank'].tolist())
    fast_ranks = set(fast['rank'].tolist())
    print(f"  Fast ranks: {sorted(fast_ranks)}")
    print(f"  Sample ranks: {sorted(sample_ranks)[:25]}...{sorted(sample_ranks)[-5:]}")
    print(f"  Fast ranks en Sample: {len(fast_ranks & sample_ranks)}/{len(fast_ranks)}")
    remaining_ranks = sample_ranks - fast_ranks
    print(f"  Remaining ranks (ETAPA 2): {sorted(remaining_ranks)}")
    print(f"  Cantidad remaining: {len(remaining_ranks)}")
