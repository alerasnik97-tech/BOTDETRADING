# ROOT ORDER READ-ONLY AUDIT

**Fecha de Auditoría:** 2026-04-27
**Veredicto:** **CLEANUP_REQUIRED** (Higiene baja, referencias obsoletas críticas)

## 1. Lo más importante
La estructura jerárquica (documentos 00, 01, 02, 03) está bien implementada, pero el directorio raíz físico es un "cementerio" de reportes, scripts y carpetas temporales. Lo más grave son los **cientos de scripts en `src` que tienen rutas absolutas harcodeadas** apuntando a `Bot V2`, lo que garantiza que fallarán al ejecutarse en la nueva raíz.

## 2. Hallazgos Detallados

### 2.1 Archivos Sueltos en Raíz
| Tipo | Cantidad Estimada | Clasificación | Recomendación |
|------|-------------------|---------------|---------------|
| `.md` | 100+ | `CLEANUP_RECOMMENDED` | Mover a `REPORTS` o `ARCHIVE_SUPERSEDED`. |
| `.json` | 30+ | `CLEANUP_RECOMMENDED` | Mover a carpetas de status o `REPORTS`. |
| `.csv` | 10+ | `CLEANUP_RECOMMENDED` | Mover a `DATA` o `OUTPUTS`. |
| `.py` | 5+ | `CLEANUP_RECOMMENDED` | Mover a `scripts` o `TOOLS`. |

### 2.2 Carpetas Temporales Detectadas
| Carpeta | Clasificación | Riesgo |
|---------|---------------|--------|
| `temp_zip_staging` | `TEMP_FOLDER_RISK` | Basura de proceso de compresión. |
| `_staging_final` | `TEMP_FOLDER_RISK` | Basura de proceso de despliegue. |
| `_zip_clean_1032631789` | `TEMP_FOLDER_RISK` | Basura de proceso de limpieza. |
| `results_REHEARSAL` | `NO_ACTION_NEEDED` | Probablemente útil para comparación. |

### 2.3 ZIPs y Duplicidad
- **Hallazgo:** `000_PARA_CHATGPT.zip` existe en la raíz y en `BOT_V2_DAYTIME_LAB`.
- **Clasificación:** `DUPLICATE_RISK`.
- **Riesgo:** Confusión sobre cuál es el bundle canónico global.

### 2.4 Referencias Obsoletas (Crítico)
- **Hallazgo:** Cientos de archivos en `BOT_V2_DAYTIME_LAB\src\` contienen la cadena `C:\Users\alera\Desktop\Bot\Bot V2`.
- **Clasificación:** `OBSOLETE_REFERENCE`.
- **Impacto:** Los scripts no funcionarán en la nueva raíz `BOT DE TRADING ultimo` sin una edición masiva de rutas.

## 3. Estado de Documentos Maestros
- **00_READ_THIS_FIRST.md:** `OK`. Las referencias a la raíz son correctas.
- **ZIP_CONTENTS_MANIFEST.md:** `OK`. Refleja la intención de la estructura, pero no menciona que la raíz física sigue sucia (lo cual es correcto para un manifiesto de entrega).

## 4. Riesgos Reales
1. **Inoperabilidad de Scripts:** El harcodeo de rutas a carpetas archivadas rompe el entorno de investigación.
2. **Confusión Documental:** Es difícil identificar los reportes de autoridad entre cientos de archivos `ADMISSION_REPORT` y `AM_` sueltos.
3. **Mantenimiento Imposible:** Una IA nueva tardará mucho tiempo en distinguir la "verdad" del "ruido".

## 5. Recomendaciones Post-Auditoría
1. **Surgical Path Repair:** Ejecutar un script que reemplace todas las instancias de `C:\Users\alera\Desktop\Bot\Bot V2` por la nueva raíz oficial.
2. **Root Sanitization:** Mover todos los archivos `.md` y `.json` que no sean `00`, `01`, `02`, `03` o `ZIP_CONTENTS_MANIFEST` a una carpeta de `ARCHIVE_SUPERSEDED\legacy_reports\`.
3. **Temp Cleanup:** Eliminar carpetas que empiezan con `temp_`, `_staging` o `_zip`.
4. **CSV Consolidation:** Mover ledgers sueltos a sus respectivas carpetas de `results` o `data`.

---
**Confirmación de Solo Lectura:**
- No se modificó ningún archivo.
- No se corrieron backtests.
- No se interactuó con MT5 ni trading real.
