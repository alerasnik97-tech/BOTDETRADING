# Forensic Audit Report: Sunday Session Gap Hypothesis

**Veredicto:** `SUNDAY_GAP_PRESENT_MATERIAL`

## 1. Pregunta Auditada
¿El laboratorio está descartando o perdiendo sistemáticamente las velas del domingo, y eso está contaminando la apertura de Asia de los lunes, los cálculos weekly y/o la lectura institucional de las estrategias?

## 2. Metodología y Datasets
- **Datasets:** `data_free_2020/prepared/` y `data_candidates_2022_2025/prepared/`.
- **Scripts Inspeccionados:** `research_lab/data_loader.py`, `institutional_research_candidate_lab/data_io.py`, `baseline_truth_model.py`.
- **Timezone:** US/Eastern (NY) para lógica de sesiones, UTC para almacenamiento en CSV.

## 3. Hallazgos Forenses
### A. Presencia de Datos (Conteo)
- Las barras de domingo **existen físicamente** en los archivos `prepared`.
- Sin embargo, hay una **pérdida parcial estructural** en la apertura: se observan de 2 a 4 barras por domingo en lugar de las 7 esperadas (17:00-23:00 NY).
- El loader de `research_lab` (main) filtra los domingos vía `fx_market_mask`.
- El loader de `institutional_research_candidate_lab` (candidatos) **NO filtra** los domingos.

### B. Contaminación del Lunes (Sunday -> Monday Asia)
- **Confirmado:** El laboratorio institucional asigna a las barras del domingo la fecha "Sunday".
- **Impacto Crítico:** El Lunes toma al **Domingo como "Día Previo"** para el cálculo de PDH/PDL.
- **Evidencia:** En la semana del 2026-04-13, el Lunes usó un rango de PDH/PDL de ~26 pips (Domingo) en lugar del rango real de ~62 pips (Viernes).

### C. Impacto Weekly
- El VWAP semanal se desplaza al perder las primeras 3-4 horas de la apertura dominical.
- Los niveles semanales (PWH/PWL) son menos afectados pero sufren de falta de precisión en la captura del gap de apertura.

### D. Impacto en la Estrategia Actual
- **Material.** La estrategia opera el lunes con niveles de referencia irrelevantes (rango del domingo).
- Esto invalida la lectura de sweeps y contextos del lunes en todo el historial del laboratorio institucional.

## 4. Localización Exacta del Problema
- **Archivo:** `institutional_research_candidate_lab/baseline_truth_model.py`
- **Función:** `compute_session_levels`
- **Bloque:** Líneas 142-152.
- **Falla:** El bucle itera sobre `dates = sorted(frame["date"].unique())`. Si el domingo está presente, `idx - 1` para el lunes apunta al domingo.
- **Causa Raíz:** Inconsistencia entre loaders. La capa de candidatos recibe domingos que la capa de investigación filtra, pero la lógica de la estrategia no está preparada para manejarlos como sesión de baja liquidez.

## 5. Veredicto Final
`SUNDAY_GAP_PRESENT_MATERIAL`

## 6. Próximo Paso Único
**Implementar `REMEDIATION_PLAN_MINIMAL`**: Modificar `compute_session_levels` para que, si el `current_date` es Lunes, busque el Viernes previo (o colapse el Domingo en el Lunes) para los cálculos de PDH/PDL.

---
**Generado por Auditoría Forense Antigravity - 2026-04-24**
