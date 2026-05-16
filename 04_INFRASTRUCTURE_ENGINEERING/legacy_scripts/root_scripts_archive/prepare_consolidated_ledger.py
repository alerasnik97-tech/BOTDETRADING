import pandas as pd
from datetime import datetime
from shutil import copy2

print("=" * 60)
print("PREPARACIÓN DE LEDGER CONSOLIDADO DE 80 TRADES")
print("=" * 60)

# Backup del ledger post-ETAPA 1
ledger_path = 'EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_WORKING.csv'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = f'EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_WORKING_BACKUP_{timestamp}.csv'

print(f"BACKUP DEL LEDGER POST-ETAPA 1:")
print(f"  Origen: {ledger_path}")
print(f"  Destino: {backup_path}")
copy2(ledger_path, backup_path)
print(f"  BACKUP COMPLETADO")

# Cargar muestra curada total
sample = pd.read_csv('EURUSD_MANUAL_ANNOTATION_SAMPLE.csv')
print(f"\nMUESTRA CURADA TOTAL:")
print(f"  Shape: {sample.shape}")

# Cargar ledger post-ETAPA 1
ledger = pd.read_csv(ledger_path)
print(f"\nLEDGER POST-ETAPA 1:")
print(f"  Shape: {ledger.shape}")

# Preparar ledger consolidado
# Necesitamos mapear columnas entre sample y ledger
# Sample usa 'id', ledger usa 'trade_id'
# Necesitamos crear un ledger de 80 trades con las columnas correctas

# Primero, vamos a crear el ledger consolidado basado en el schema del ledger
print(f"\nCREANDO LEDGER CONSOLIDADO:")

# Usar la muestra curada total como base, pero con las columnas del ledger
# El ledger tiene columnas adicionales que no están en sample
# Necesitamos preservar las 25 anotaciones de ETAPA 1 y preparar las 55 restantes

# Por ahora, vamos a crear un CSV de los 55 trades restantes para anotación
# Similar a como se hizo con ETAPA 1

fast = pd.read_csv('EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_SAMPLE.csv')
fast_ids = set(fast['id'].tolist())
sample_ids = set(sample['id'].tolist())
remaining_ids = sample_ids - fast_ids

# Extraer los 55 trades restantes
remaining_trades = sample[sample['id'].isin(remaining_ids)].copy()
print(f"  Trades restantes extraídos: {len(remaining_trades)}")

# Guardar como CSV para anotación (similar a ETAPA 1)
output_path = 'EURUSD_MANUAL_ANNOTATION_STAGE2_REMAINDER.csv'
remaining_trades.to_csv(output_path, index=False)
print(f"  Guardado en: {output_path}")
print(f"  Shape: {remaining_trades.shape}")
print(f"  time_block values: {remaining_trades['time_block'].value_counts().to_dict()}")
