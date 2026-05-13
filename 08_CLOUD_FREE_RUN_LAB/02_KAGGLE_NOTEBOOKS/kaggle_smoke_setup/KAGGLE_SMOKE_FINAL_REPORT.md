## Estado
KAGGLE_SMOKE_SETUP_READY

## Qué se preparó
- **Carpeta institucional**: `08_CLOUD_FREE_RUN_LAB\02_KAGGLE_NOTEBOOKS\kaggle_smoke_setup\`.
- **Plan de Smoke Test**: Definición de objetivos y checks de seguridad.
- **Política de Acceso Seguro**: Protocolo para usar GitHub PAT vía Kaggle Secrets y recomendación de Dataset Privado.
- **Celdas de Notebook**: 7 celdas listas para copiar y ejecutar, cubriendo desde el check de entorno hasta la limpieza de secretos.
- **Plan de Paquete Futuro**: Especificación de lo que debe contener el primer paquete real de Kaggle.
- **Políticas de Entrega**: Definición de cómo manejar los outputs que regresan de la nube.

## Qué debe hacer el usuario en Kaggle
1. **No ingresar contraseña**: Nunca escribir credenciales directamente en el notebook.
2. **Crear Kaggle Secret**: Generar un Personal Access Token (PAT) en GitHub y guardarlo en Kaggle como `GH_TOKEN`.
3. **Ejecutar Celdas**: Copiar y pegar el contenido de `KAGGLE_NOTEBOOK_CELLS.md` en un nuevo notebook privado.
4. **Descargar Outputs**: Al finalizar, descargar la carpeta `KAGGLE_SMOKE_OUTPUTS`.
5. **Pasar Output a ChatGPT**: Entregar el `KAGGLE_SMOKE_MANIFEST.json` para validación de Readiness.

## Seguridad
Confirmado:
- no token escrito: OK (se usa Secrets)
- no datos subidos: OK (solo smoke test de entorno)
- no broker: OK (deshabilitado por política)
- no backtest: OK (deshabilitado en smoke test)
- no sweep: OK (deshabilitado en smoke test)

## Próximo paso
Si el smoke test en Kaggle arroja `KAGGLE_SMOKE_READY`, procederemos a preparar el primer **Cloud Package** real para una micro-corrida de MANIPULANTE 4, una vez que la sesión local actual finalice.
