import pandas as pd

print("=" * 60)
print("AUDITORÍA POST-ETAPA 1")
print("=" * 60)

# Ledger oficial post-ETAPA 1
ledger = pd.read_csv('EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_WORKING.csv')
print(f"LEDGER OFICIAL POST-ETAPA 1:")
print(f"  Shape: {ledger.shape}")
print(f"  Filas: {len(ledger)}")

human_fields = ['liquidity_source', 'trigger_type', 'confirmation_type', 
                'operational_context', 'entry_motive', 'quality_rating', 'comment']
human_filled = ledger[human_fields].notna().sum().sum()
total_cells = len(ledger) * len(human_fields)
print(f"  Human fields filled: {human_filled}/{total_cells}")
print(f"  annotation_status: {ledger['annotation_status'].value_counts().to_dict()}")

# Muestra curada total
curated = pd.read_csv('EURUSD_MANUAL_ANNOTATION_SAMPLE.csv')
print(f"\nMUESTRA CURADA TOTAL:")
print(f"  Shape: {curated.shape}")
print(f"  Filas: {len(curated)}")

# Fast signal sample
fast_signal = pd.read_csv('EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_SAMPLE.csv')
print(f"\nFAST SIGNAL SAMPLE:")
print(f"  Shape: {fast_signal.shape}")
print(f"  Filas: {len(fast_signal)}")

# Diferencia
remaining = len(fast_signal) - len(ledger)
print(f"\nTRADUES RESTANTES PARA ETAPA 2: {remaining}")
