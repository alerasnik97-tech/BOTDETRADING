# KAGGLE_V49_7C_SECURITY_POLICY

- **No hardcodear tokens**: El uso de `GH_TOKEN` debe ser estrictamente a través de `UserSecretsClient` de Kaggle.
- **No imprimir secretos**: Queda prohibido cualquier print o log que exponga el contenido de tokens o credenciales.
- **No subir .env ni kaggle.json**: Estos archivos no deben formar parte del repositorio ni del notebook.
- **No usar brokers**: No se permite la conexión a APIs de brokers (OANDA, MetaTrader) desde Kaggle.
- **No tocar producción**: La ejecución se limita a `03_RESEARCH_LAB` y `08_CLOUD_FREE_RUN_LAB`.
- **No usar TEST 2025-2026**: Queda prohibido el acceso o lectura de los datos de TEST finales para evitar fugas (leakage).
- **No generar ZIPs**: El flujo de trabajo debe ser transparente mediante archivos sueltos auditables o commit directo.
- **No subir datos pesados**: No subir archivos `.parquet` masivos a GitHub.
- **Auditoría**: Todo output generado debe ser auditable antes de su integración final en el laboratorio local.
