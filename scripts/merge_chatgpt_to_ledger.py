import pandas as pd
import os
from datetime import datetime
from shutil import copy2

print("=" * 60)
print("BACKUP Y MERGE CONTROLADO")
print("=" * 60)

# Backup del ledger oficial
ledger_path = 'EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_WORKING.csv'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = f'EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_WORKING_BACKUP_{timestamp}.csv'

print(f"CREANDO BACKUP:")
print(f"  Origen: {ledger_path}")
print(f"  Destino: {backup_path}")
copy2(ledger_path, backup_path)
print(f"  BACKUP COMPLETADO: {backup_path}")

# Cargar ambos archivos
chatgpt_df = pd.read_csv('EURUSD_FAST_SIGNAL_25_ANNOTATIONS_BY_CHATGPT.csv')
ledger_df = pd.read_csv(ledger_path)

# Mapear campos humanos de ChatGPT al ledger
human_fields = ['liquidity_source', 'trigger_type', 'confirmation_type', 
                'operational_context', 'entry_motive', 'quality_rating', 'comment']

# Convertir campos humanos a object dtype en ledger para permitir strings
for field in human_fields:
    ledger_df[field] = ledger_df[field].astype(object)

print()
print("INTEGRANDO ANOTACIONES DE CHATGPT:")
print(f"  ChatGPT filas: {len(chatgpt_df)}")
print(f"  Ledger filas: {len(ledger_df)}")

# Para cada rank, actualizar campos humanos en el ledger
for idx, row in chatgpt_df.iterrows():
    rank = row['rank']
    # Encontrar fila correspondiente en ledger por rank
    ledger_idx = ledger_df[ledger_df['rank'] == rank].index[0]
    
    # Actualizar campos humanos
    for field in human_fields:
        ledger_df.at[ledger_idx, field] = row[field]
    
    # Recalcular missing_human_fields_count
    missing_count = sum([1 for f in human_fields if pd.isna(ledger_df.at[ledger_idx, f])])
    ledger_df.at[ledger_idx, 'missing_human_fields_count'] = missing_count
    
    # Actualizar annotation_status
    if missing_count == 0:
        ledger_df.at[ledger_idx, 'annotation_status'] = 'READY'
    else:
        ledger_df.at[ledger_idx, 'annotation_status'] = 'PENDING'

print(f"  CAMPOS HUMANOS ACTUALIZADOS: 7 campos x 25 filas")

# Verificar taxonomía antes de guardar
print()
print("VERIFICANDO TAXONOMÍA:")
TAXONOMY = {
    'liquidity_source': ['previous_day_high', 'previous_day_low', 'asia_high', 'asia_low', 'london_high', 'london_low', 'none_unclear'],
    'trigger_type': ['sweep_reclaim', 'sweep_displacement', 'continuation_after_break', 'reversal_after_sweep', 'breakout_from_compression', 'none_unclear'],
    'confirmation_type': ['close_back_inside', 'strong_displacement_bar', 'structure_break', 'reclaim_then_go', 'immediate_rejection', 'none_unclear'],
    'operational_context': ['london_open_drive', 'london_continuation', 'london_reversal', 'pre_ny_transition', 'early_ny_followthrough', 'none_unclear'],
    'entry_motive': ['liquidity', 'displacement', 'reclaim', 'time_window', 'confluence', 'none_unclear'],
    'quality_rating': ['A', 'B', 'C']
}

taxonomy_issues = []
for field in human_fields[:-1]:  # Excluir comment
    valid_values = TAXONOMY[field]
    for idx, row in ledger_df.iterrows():
        val = row[field]
        if pd.notna(val) and val not in valid_values:
            taxonomy_issues.append(f"  Rank {row['rank']} {field}: '{val}' no está en taxonomía")

if taxonomy_issues:
    print("  ERRORES DE TAXONOMÍA:")
    for issue in taxonomy_issues:
        print(issue)
else:
    print("  TAXONOMÍA OK - Todos los valores válidos")

# Guardar ledger actualizado
ledger_df.to_csv(ledger_path, index=False)
print()
print(f"LEDGER ACTUALIZADO GUARDADO: {ledger_path}")

# Verificar estado final
print()
print("ESTADO FINAL DEL LEDGER:")
human_filled = ledger_df[human_fields].notna().sum().sum()
total_cells = len(ledger_df) * len(human_fields)
print(f"  CELLS HUMANAS LLENAS: {human_filled}/{total_cells}")
print(f"  ANNOTATION_STATUS: {ledger_df['annotation_status'].value_counts().to_dict()}")
