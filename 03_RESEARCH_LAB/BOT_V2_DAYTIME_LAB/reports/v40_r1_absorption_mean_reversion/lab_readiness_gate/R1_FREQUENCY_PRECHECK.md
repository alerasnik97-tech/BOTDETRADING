# AUDITORÍA DE FRECUENCIA OPERATIVA E HIGIENE CAUSAL (FREQUENCY PRECHECK)

## 1. Verificación del Orquestador
- **Implementación del Throttler**: El script `run_r1_micro_probe.py` inicializa un diccionario independiente de seguimiento de operaciones diarias (`daily_trades`) para cada configuración paramétrica evaluada.
- **Incondicionalidad de Causalidad**: Las señales candidatas provenientes del detector de absorción (`R1AbsorptionDetector`) son procesadas en estricto orden cronológico (`itertuples()`). Al alcanzar la cuota diaria configurada (`max_trades_per_day = 3`), el ciclo descarta de forma inmediata y automática cualquier señal posterior surgida en la misma fecha calendario NY.
- **Ausencia de Look-Ahead Bias**: Se certifica la inexistencia de lógicas de ordenamiento futuro (ej. *seleccionar los 3 trades con mayor R neto del día*). El algoritmo opera a ciegas de los resultados futuros, garantizando una simulación física real.

## 2. Preparación de Evidencia de Salida
Para permitir una verificación independiente del cumplimiento de este límite a lo largo de los 76 meses, el orquestador tiene designada la ruta de salida para la auditoría de frecuencia transaccional:
`reports/v40_r1_absorption_mean_reversion/R1_MICRO_PROBE_TRADE_FREQUENCY_AUDIT.csv`

*Veredicto: Aprobado. Ninguna configuración en el preflight superó las 3 operaciones diarias.*
