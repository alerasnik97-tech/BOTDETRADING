# CLAUDE 4.7 OPUS HIGH AUDIT BRIEFING — R1 FINAL CONFIRMATION

## 1. Misión del Auditor
Se requiere una auditoría final y definitiva sobre la candidata `cfg_r1_factory_opt_001`. Tu misión es validar si esta configuración es apta para iniciar una fase de incubación (paper trading).

## 2. Foco Forense
- **Congelación de Parámetros**: Verifica en `R1_FINAL_CONFIRMATION_CONFIG_FREEZE.json` que no haya habido "ajustes finos" oportunistas.
- **Robustez al Slippage**: Evalúa si el paso de 0.2 a 0.3 pips en TEST es lo suficientemente sólido.
- **Estabilidad Temporal**: Revisa `R1_FINAL_CONFIRMATION_SUBPERIOD_ROBUSTNESS.csv`. ¿Hay algún año que explique más del 30% del retorno?
- **Fuga de Información**: ¿Hay algún indicio de que la fase de Factory (V43) contaminó la validez de la muestra TEST en esta fase V44?

## 3. Veredicto Requerido
Si detectas cualquier anomalía o fragilidad, reporta `R1_FINAL_CONFIRMATION_RED`. Si la evidencia es suficiente, sanciona la preparación para papel.
