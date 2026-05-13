# CHECKPOINT_REQUIREMENTS

Para ejecuciones cloud, el motor debe soportar:
- **Guardado de Estado**: Guardar el progreso actual (último trade, última barra procesada, métricas acumuladas) en un archivo `checkpoint.json`.
- **Frecuencia**: El guardado debe ocurrir cada X minutos o cada Y barras (configurable).
- **Integridad**: El archivo de checkpoint debe escribirse de forma atómica (escribir temporal y luego renombrar) para evitar corrupción si el proceso se corta a mitad de escritura.
- **Formato**: JSON legible y auditable.
