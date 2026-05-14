# KAGGLE_V49_7C_OUTPUT_HANDOFF

Este documento explica cómo reintegrar los resultados de Kaggle al proyecto local.

## Opciones de Handoff

### Opción A: Push Directo desde Kaggle (Recomendada si GH_TOKEN está listo)
1. Configurar `git user.name` y `git user.email` en el Notebook.
2. `git add` de los resultados en las carpetas institucionales correspondientes.
3. `git commit -m "[cloud] integrate v49.7c results from kaggle"`.
4. `git push origin clean-sync-branch`.

### Opción B: Descarga Manual
1. Comprimir la carpeta de resultados en Kaggle.
2. Descargar el archivo a la PC local.
3. Colocar en `08_CLOUD_FREE_RUN_LAB/10_CLOUD_OUTPUT_INBOX/`.
4. Notificar a Antigravity para que realice la auditoría e integración.

## Entregables Esperados
- Archivos de configuración final.
- Logs de ejecución (`run.log`).
- Reportes de trades y resultados (Train/Val).
- Rankings de mejores configuraciones.
- Auditorías de rowcount, duplicados y fechas.
- Archivo de decisión final.
