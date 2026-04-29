# PHASE40 ROCKI AM RESCUE REPORT

## 1. Lo mas importante
Se ha rescatado con éxito la estrategia **SCBI_M5_GLOBAL**, identificada como el candidato óptimo para la sesión de madrugada de NY (Overnight). La estrategia ha sido renombrada operativamente como **ROCKI AM** y se ha organizado toda su evidencia técnica (métricas, datasets, scripts de validación) en una nueva estructura de carpetas aislada de MANIPULANTE. El bot queda "congelado" y documentado para su futura implementación en un entorno VPS.

## 2. Veredicto final exacto
**ROCKI_AM_RESCUED_AND_ARCHIVED**

## 3. Estrategia Encontrada
- **Nombre Historico**: SCBI_M5_GLOBAL.
- **Nuevo Nombre**: ROCKI AM.
- **Horario**: 00:00 – 04:00 NY (approx).
- **Instrumento**: EURUSD.
- **Timeframe**: M5.

## 4. Evidencia Encontrada
- **Reporte de Validacion**: `scratch/scbi_global_validation_results.json` (Aprobado 5/5).
- **Dataset de Trades**: `scratch/real_htf_filter_ab_results.json` (N=1080).
- **Script de Auditoria**: `scratch/run_scbi_global_validation.py`.
- **Rutas**: Todos los archivos críticos fueron copiados a `ROCKI_AM/`.

## 5. Metricas Principales (Baseline)
- **Sample**: 1080 trades.
- **Profit Factor**: **2.44**.
- **Expectancy**: **0.43R**.
- **Win Rate**: **62.1%**.
- **Max Drawdown**: -9.6R.
- **Frecuencia**: ~22 trades/mes.

## 6. ¿Por que no se usa ahora?
La estrategia opera principalmente entre las **00:00 y 04:00 AM NY**, lo que hace inviable su operación manual consistente sin afectar la salud y el desempeño diurno en **MANIPULANTE**. Requiere automatización total en VPS.

## 7. ¿Que falta para futuro VPS?
- Auditoría de costos netos (Comisiones Prop Firm).
- Integración de News Gate nocturno.
- Infraestructura VPS y runner independiente.

## 8. Seguridad
- **MANIPULANTE**: Intacta. No se modificó ningún archivo del bot oficial.
- **MT5**: No se interactuó con la terminal.
- **Ejecucion**: ROCKI AM está archivada y no tiene lanzador activo.

---
*Fase completada. ROCKI AM está lista para ser el segundo bot oficial del laboratorio en el futuro.*
