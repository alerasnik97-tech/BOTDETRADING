# KAGGLE_V49_7C_RESOURCE_BUDGET

## Estimación de Cómputo
- **Configs**: 600 (mínimo) a 1200 (máximo).
- **Tiempo estimado por config**: 30-60 segundos.
- **Tiempo total estimado**: 5 a 20 horas de cómputo continuo.
- **Memoria RAM**: Se estima un uso de 4-8 GB.

## Riesgos y Mitigación
- **Timeout**: Las sesiones de Kaggle suelen durar 12 horas. Se requiere el uso de checkpoints cada 50 configs.
- **CPU Throttling**: Kaggle puede reducir la prioridad tras varias horas. No afecta la calidad, solo el tiempo.
- **Desconexión**: El sistema de checkpoints garantiza que no se pierda más de una hora de trabajo ante una caída de sesión.
