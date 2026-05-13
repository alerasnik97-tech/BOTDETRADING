# POLÍTICA OBLIGATORIA DE RECONSTRUCCIÓN ATÓMICA DEL ZIP OFICIAL
**ID:** `POL-ZIP-ATOMIC-001`  
**Estado:** OBLIGATORIO / ENFORCED  
**Fecha de Implementación:** 2026-05-13  

## 1. Prohibición de Borrado Directo
Queda estrictamente prohibido eliminar de forma directa el archivo oficial único `000_PARA_CHATGPT.zip` antes de generar su reemplazo (ej. usar `Remove-Item` o `os.remove` secuencial previo a la compresión). Este patrón de diseño es inseguro y puede dejar la raíz del proyecto desprovista del artefacto de entrega si el proceso de construcción del nuevo ZIP es interrumpido o sufre fallos.

## 2. Flujo de Trabajo Atómico Obligatorio
Todo script o agente que requiera actualizar o regenerar el ZIP oficial para ChatGPT debe respetar de forma inmutable la siguiente secuencia:

### Fase A: Construcción en Aislamiento Temporal
1. El nuevo empaquetado debe realizarse exclusivamente sobre un archivo con extensión y sufijo temporal claramente demarcado:
   `000_PARA_CHATGPT_BUILDING.tmp.zip`
2. Durante esta fase, el archivo oficial anterior (`000_PARA_CHATGPT.zip`) debe permanecer intacto y disponible en la raíz.

### Fase B: Auditoría de Integridad y Cumplimiento (Pre-flight Check)
El archivo temporal debe ser inspeccionado mediante el módulo `zipfile` de Python para certificar:
- `testzip() == None` (Cero corrupción estructural).
- Ausencia total de datos crudos (`tick/`, `raw/`, archivos `.parquet`).
- Ausencia total de entornos virtuales (`venv/`, `venv_v37/`, `.venv/`).
- Ausencia de repositorios y cachés (`.git/`, `__pycache__/`, `.pytest_cache/`).
- Ausencia de archivos de secretos (`.env`, `kaggle.json`, credenciales).
- Presencia de las carpetas institucionales clave y reportes de gobernanza.
- `no_internal_zips == True` (Prohibido anidar archivos `.zip`).

### Fase C: Reemplazo Atómico (Fail-Safe Promotion)
1. **Promoción:** Solo si la Fase B es exitosa en todos sus criterios, se procederá a renombrar/sobrescribir el archivo temporal para convertirlo en el oficial:
   `000_PARA_CHATGPT_BUILDING.tmp.zip` $\rightarrow$ `000_PARA_CHATGPT.zip`
2. **Fail-Close:** Si la Fase B falla en cualquiera de sus aserciones, el archivo temporal debe ser eliminado o puesto en cuarentena con reporte explícito, y **el ZIP oficial anterior jamás debe ser borrado ni alterado**.
