# KAGGLE_SMOKE_TEST_PLAN

## Objetivo del primer notebook
- Confirmar que el entorno de Kaggle funciona correctamente.
- Confirmar acceso seguro al repositorio privado (vía PAT) o al Kaggle Input.
- Confirmar capacidad de escritura en `/kaggle/working`.
- Confirmar que los outputs pueden ser empaquetados y descargados.
- **IMPORTANTE**: No correr ninguna estrategia de trading real.

## Checks Críticos
- [ ] **Python OK**: Versión compatible instalada.
- [ ] **Repo/Input OK**: Código accesible en la ruta esperada.
- [ ] **Visibilidad**: Carpeta `08_CLOUD_FREE_RUN_LAB` visible tras el clonado/carga.
- [ ] **Output Manifest**: Archivo JSON de estado creado correctamente.
- [ ] **No Secretos**: Confirmar que no se han impreso tokens o passwords en los logs.
- [ ] **No Datos Crudos**: Verificar que no se han cargado gigabytes de datos innecesarios.
- [ ] **No Backtest**: Confirmar que no se ha ejecutado el runner de estrategia.

## Estados Posibles
- **KAGGLE_SMOKE_READY**: Todo verificado, entorno apto para preparación de paquete real.
- **KAGGLE_SMOKE_BLOCKED_REPO_ACCESS**: Fallo al clonar el repo privado.
- **KAGGLE_SMOKE_BLOCKED_INPUT_MISSING**: Dataset de entrada no encontrado.
- **KAGGLE_SMOKE_BLOCKED_SECURITY**: Se detectó riesgo de filtración de secretos o uso indebido.
