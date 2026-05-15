# KAGGLE_SMOKE_FAILURE_RECOVERY

## Escenarios de Fallo y Acciones:

- **Fallo en Clonado**: Verificar conexión a internet en el Notebook de Kaggle o validez de la URL del repositorio.
- **Fallo en Imports**: Puede deberse a dependencias faltantes en el entorno base de Kaggle. Documentar los paquetes faltantes y proceder a instalarlos vía `!pip install` en una celda adicional si es necesario.
- **Detección de Seguridad (Safety Scan FAILED)**: Si se detecta un archivo o término prohibido, detener la prueba inmediatamente y auditar el contenido del repositorio local antes de re-intentar.
- **Error de Escritura**: Verificar permisos en `/kaggle/working/`.
