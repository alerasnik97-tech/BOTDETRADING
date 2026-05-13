# R1_PREFLIGHT_CHECKPOINT

## Estado: PREFLIGHT_SUCCESS
- **Runner**: `run_r1_micro_probe.py`
- **Integridad**:
  - Código V7 (motor, utils) recuperado intacto tras borrado accidental.
  - Fix de importaciones en motor y scripts finalizado.
  - CostModelConfig adaptado a régimen FTMO con 5 USD/lot round-turn y slippage de 0 y 0.2.
  - Columnas OHLC estándar unificadas.
  - Ejecución verificada sin look-ahead, sin crashes.
- **Rendimiento preflight**: Se procesaron los primeros 5 meses (2020-01 a 2020-05) a una velocidad aproximada de 1.2 minutos por mes para 54 configuraciones.
- **Recomendación**: El runner es estable. Se recomienda programar la corrida pesada overnight o transferir a entorno cloud, ya que los 76 meses completos tardarán aproximadamente 90 minutos de CPU local.
