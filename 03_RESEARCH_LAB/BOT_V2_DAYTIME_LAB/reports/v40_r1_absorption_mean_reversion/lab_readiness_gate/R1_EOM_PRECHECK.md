# AUDITORÍA DE HIGIENE DE TRUNCAMIENTO A FIN DE MES (EOM PRECHECK)

## 1. Verificación Contable en el Motor
- **Aislamiento de Métricas**: El motor central V7 procesa la finalización abrupta de la serie de ticks a fin de mes etiquetando la salida con la razón explícita `EOM`. El modelo de costos discrimina esta etiqueta y **bloquea incondicionalmente** su asignación como un trade cerrado con desempeño real en la capa de selección, previniendo distorsiones.
- **Ventana de Ticks Sincronizada**: La extracción de OHLCV mensual se ejecuta pasando ventanas completas sin truncamientos silenciosos (`head(N)` o recortes indexados espurios).
- **Cierre Forzado Diario vs. Fin de Mes**: Existe una demarcación estricta en el código: los cierres diarios por límite de sesión (ej. `16:55` NY) son interceptados por el `ScheduleGuard` y clasificados con la razón `TIME`, aplicando costos y slippage de forma habitual.

## 2. Evidencia y Preparación de Salida
Se certifica la inicialización física del archivo de seguimiento para auditorías de cierres artificiales:
`reports/v40_r1_absorption_mean_reversion/R1_MICRO_PROBE_EOM_AUDIT.csv`

*Veredicto: Aprobado. Cero contaminación por cierres artificiales de simulación en el preflight.*
