# LAB OUTPUT EVIDENCE CONTRACT

## 1. Scope
Este contrato define los requisitos obligatorios para cualquier evidencia (reportes, CSVs, métricas) generada durante las fases de laboratorio autorizadas. Cualquier salida que no cumpla con este contrato será considerada INVÁLIDA e INAUDITABLE.

## 2. Allowed Output Directories
Solo se permite escribir resultados en:
- `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/<run_id>/`
- `03_RESEARCH_LAB/research_lab/reports/<run_id>/`

## 3. Forbidden Output Directories
Está terminantemente prohibido escribir resultados en:
- Raíz del proyecto (`/`).
- `05_MARKET_DATA_VAULT/` (Solo lectura).
- `07_BACKUPS/` (Solo lectura/archivo).
- `01_CORE_PRODUCTION/` (Congelado).
- Cualquier carpeta que contenga `quarantine`, `legacy` o `temp` en su nombre.

## 4. Required Evidence Files per Run
Cada ejecución exitosa de laboratorio debe generar como mínimo:
1. `MANIFEST.json`: Metadatos completos (RunID, Git Hash, Config Hash, Timestamps).
2. `TRADES.csv`: Lista completa de trades ejecutados con columnas de costo auditables.
3. `RANKING.csv`: Resumen de métricas por configuración (solo métricas TRAIN).
4. `LOGS.txt`: Registro de ejecución con marcas de tiempo y auditoría de guards.

## 5. Manifest Schema (Minimum Requirements)
- `run_id`: UUID único.
- `train_only`: `True`.
- `holdout_touched`: `False`.
- `max_timestamp_cutoff`: `2024-12-31`.
- `git_commit_sha`: Hash del código utilizado.
- `input_data_hashes`: Hashes de los archivos OHLCV utilizados.

## 6. No-Leakage Enforcement
- El sistema de reporte debe verificar que no existan trades con fecha >= 2025-01-01.
- El sistema debe fallar (`FAIL-CLOSED`) si intenta escribir sobre un `run_id` existente.

## 7. Evidence Permanence
- Una vez finalizada una corrida de laboratorio y aprobada por auditoría, los resultados deben ser inmutables.
- No se permite el uso de `git push --force` sobre ramas de evidencia.
