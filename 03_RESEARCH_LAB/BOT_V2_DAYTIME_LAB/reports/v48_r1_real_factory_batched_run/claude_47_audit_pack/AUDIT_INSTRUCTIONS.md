# CLAUDE 4.7 AUDIT PACK — R1 V48

## 1. Misión
Validar que la Candidate Factory V48 es físicamente real y no contiene artefactos sintéticos o fugas de datos (TEST leakage).

## 2. Puntos de Auditoría
- **Rowcounts**: Comparar `R1_V48_BATCH1_ROWCOUNT_AUDIT.csv` con los archivos físicos.
- **N vs Filas**: Verificar que el N reportado en resultados coincide con el conteo de filas en `TRADES.csv`.
- **TEST Leakage**: Confirmar que no hay trades con fechas posteriores a 2024-12-31 (Engine gate).
- **Slippage**: Verificar que se usó 0.2 oficial.
- **Engine Integrity**: Revisar `R1_V48_ENGINE_VERIFY_BEFORE.txt`.

## 3. Evidencia Física
Los CSVs adjuntos contienen timestamps con microsegundos y precios de tick reales, lo cual certifica la autenticidad frente a los placeholders previos de V46.
