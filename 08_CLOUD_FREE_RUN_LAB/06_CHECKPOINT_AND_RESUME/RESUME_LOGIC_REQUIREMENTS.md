# RESUME_LOGIC_REQUIREMENTS

- **Detección Automática**: Al iniciar, el runner debe buscar un archivo `checkpoint.json`.
- **Carga de Estado**: Si existe, cargar los parámetros y métricas para continuar desde el punto exacto.
- **Validación de Datos**: Asegurar que los datos crudos cargados coinciden con los del checkpoint (mismo símbolo, misma temporalidad).
- **Logs de Re-inicio**: Registrar claramente en los logs que se ha retomado una corrida desde el checkpoint X.
