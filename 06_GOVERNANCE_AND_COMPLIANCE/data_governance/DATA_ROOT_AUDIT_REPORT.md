# DATA ROOT AUDIT REPORT

## 1. Status
**DATA_ROOT_AUDIT_AND_MOVES_APPLIED_SAFE**

## 2. Executive Summary
Se ha completado la auditoría de los datasets remanentes en la raíz. Se han identificado 7 ítems principales de datos (Legacy, Manual y Derived) con un tamaño total aproximado de 160MB. Se ha generado un manifiesto de hashes SHA256 para todos los archivos. Los datasets han sido movidos a sus ubicaciones canónicas dentro de `05_MARKET_DATA_VAULT/`.

## 3. Data Root Inventory
| Item | Files | Size | Status |
| :--- | :--- | :--- | :--- |
| `DATA MANUAL` | 9 | 3MB | MOVED |
| `data_usdjpy_2016_2019` | 4 | 32MB | MOVED |
| `data_usdjpy_2016_2021` | 4 | 48MB | MOVED |
| `data_usdjpy_2022_2025` | 4 | 32MB | MOVED |
| `ecb_stage2_checkpoints` | 6 | 7MB | MOVED |
| `scbi_*_checkpoints` | 5 | <1MB | MOVED |
| `reports` | 101 | 0.6MB | **BLOCKED** |

## 4. Data Classification
- **MANUAL_DATA:** `DATA MANUAL`.
- **LEGACY_DATA:** `data_usdjpy_*`.
- **DERIVED_DATA:** `ecb_stage2_checkpoints`, `scbi_*_checkpoints`.
- **IMPORT_PATH_BLOCKED:** `reports`.

## 5. Manifest Summary
Manifiesto generado en:
`05_MARKET_DATA_VAULT/manifests/ROOT_DATA_AUDIT_MANIFEST.csv`
Incluye hashes SHA256 para todos los archivos (todos menores a 100MB).

## 6. Reference Audit
- `data_usdjpy_*`: Referenciados en `research_lab/config.py`. Se requiere actualización de paths en Phase D.
- `ecb_stage2_checkpoints`: Referenciada en autopilot scripts.
- `reports`: Referenciada en >20 scripts activos. Movimiento bloqueado hasta Phase D (Path Migration).

## 7. Moves Applied
| Source | Destination | Tracked | Command |
| :--- | :--- | :--- | :--- |
| `DATA MANUAL` | `05_MARKET_DATA_VAULT/manual_data/DATA_MANUAL/` | YES | git mv |
| `data_usdjpy_2016_2019` | `05_MARKET_DATA_VAULT/legacy_data/data_usdjpy_2016_2019/` | YES | git mv |
| `data_usdjpy_2016_2021` | `05_MARKET_DATA_VAULT/legacy_data/data_usdjpy_2016_2021/` | YES | git mv |
| `data_usdjpy_2022_2025` | `05_MARKET_DATA_VAULT/legacy_data/data_usdjpy_2022_2025/` | YES | git mv |
| `ecb_stage2_checkpoints` | `05_MARKET_DATA_VAULT/derived_data/ecb_checkpoints/` | YES | git mv |
| `scbi_checkpoints` | `05_MARKET_DATA_VAULT/derived_data/scbi_checkpoints/` | YES | git mv |

## 8. Moves Blocked
| Item | Reason |
| :--- | :--- |
| `reports` | Alto riesgo de rotura de scripts activos de reporte y auditoría. Requiere Phase D. |

## 9. ZIP Policy In Data
- No se detectaron archivos ZIP dentro de las carpetas de datos auditadas.

## 10. Root After
- Datasets legacy eliminados de raíz.
- Pendientes carpetas de infraestructura y scripts.

## 11. Safety Verification
- data_deleted: NO
- data_modified: NO
- raw_data_touched_as_content: NO
- validation_process_run: NO
- holdout_process_run: NO
- 2025_2026_used_for_analysis: NO
- backtest_run: NO
- strategy_run: NO

## 12. Remaining Work
- Actualizar `research_lab/config.py` para apuntar a los nuevos paths en el vault.
- Migrar `reports` una vez que el motor de reportes sea desacoplado de la raíz.

## 13. Copy-Paste Summary for ChatGPT
`STATUS: DATA_MOVES_APPLIED, ITEMS: 6, VAULT_SYNC: OK, MANIFEST: CREATED, BLOCKED: reports, RISK: Path updates required in Phase D.`
