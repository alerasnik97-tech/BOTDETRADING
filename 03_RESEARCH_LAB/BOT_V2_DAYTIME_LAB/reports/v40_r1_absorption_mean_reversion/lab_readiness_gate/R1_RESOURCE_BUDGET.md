# PROYECCIÓN Y PRESUPUESTO DE RECURSOS COMPUTACIONALES (RESOURCE BUDGET)

## 1. Dimensionamiento de la Simulación
- **Meses Totales**: `76` meses calendario (2020-01 a 2026-04).
- **Espacio Paramétrico**: `54` configuraciones concurrentes.
- **Granularidad de Barras**: OHLCV de Ticks agregados al vuelo en marcos temporales de `M3` y `M5`.

## 2. Estimaciones de Rendimiento y Almacenamiento
- **Tiempo Estimado de CPU (Local)**: ~1.2 minutos por mes procesado $\rightarrow$ **~91.2 minutos totales (~1.5 horas)** en un hilo único con el motor unificado V7.
- **Consumo de Memoria (RAM)**: Estable entre **400 MB y 650 MB** gracias a la recolección explícita invocada al cierre de cada iteración mensual (`gc.collect()`).
- **Huella en Disco (CSVs de Salida)**: ~300 KB por mes $\rightarrow$ **~22.8 MB esperados** para el archivo transaccional final `R1_MICRO_PROBE_TRADES.csv`.

## 3. Estrategia de Ejecución y Checkpoints
**RECOMENDACIÓN INSTITUCIONAL: EJECUCIÓN LOCAL PRIMARIA.**
- De acuerdo con la regla operativa del proyecto, al proyectarse un tiempo de ejecución cercano a los 90 minutos en un entorno rigurosamente estable y verificado in situ, **procede ejecutar la simulación en la máquina local como primera opción**.
- La plataforma de nube (*Kaggle / Cloud Lab*) queda formalmente en reserva como solución de contingencia o para futuras optimizaciones nocturnas de hiperparámetros.
- **Hitos de Control Recomendados**: Implementar pausas de revisión o reportes automáticos al concluir los bloques transaccionales clave: `Mes 12`, `Mes 24`, `Fin de TRAIN (2021-12)`, `Fin de VAL (2023-12)` y `Fin Total`.
