# GITHUB_ACTIONS_DO_NOT_USE_FOR

GitHub Actions **no sirve para**:
- **Backtests pesados**: Consumen demasiados minutos de acción y CPU.
- **Parquet grandes**: No subir datos masivos al repositorio de Git para ser usados en CI.
- **Ticks completos**: Imposible de manejar por límites de almacenamiento.
- **Sweeps nocturnos**: No es una plataforma de cómputo genérico persistente.
- **Datos privados pesados**: Riesgo de que queden en los logs de GitHub o en el runner.
