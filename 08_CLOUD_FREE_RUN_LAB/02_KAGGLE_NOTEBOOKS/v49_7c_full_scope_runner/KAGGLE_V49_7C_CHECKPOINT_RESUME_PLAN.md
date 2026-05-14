# KAGGLE_V49_7C_CHECKPOINT_RESUME_PLAN

## Estrategia de Checkpoints
- Se guardará el estado cada **50 configuraciones** procesadas.
- El archivo `checkpoint_v49_7c.json` contendrá el índice de la última configuración exitosa y las métricas acumuladas.
- En caso de desconexión de Kaggle, el runner buscará este archivo para reanudar.

## Procedimiento de Re-inicio
1. Reiniciar la sesión de Kaggle.
2. Clonar/Cargar el repo de nuevo.
3. El runner detectará el checkpoint y preguntará (o iniciará automáticamente) desde la configuración N+1.
4. Se debe verificar que no existan duplicados en los archivos de resultados finales tras un re-inicio.
