# KAGGLE_MICRO_PROBE_RUNBOOK

## Paso 1: Configuración
- Crear nuevo Notebook en Kaggle.
- Configurar "Internet On" (si se necesitan dependencias externas).
- Configurar "Persistence" si se desea mantener archivos entre sesiones (con cuidado).

## Paso 2: Datos
- Subir el `CLOUD_PACKAGE` como un Dataset privado de Kaggle.
- Añadir el Dataset al Notebook.

## Paso 3: Ejecución
- Usar celdas de script para lanzar el runner.
- Redirigir output a `/kaggle/working/`.
- Usar la opción "Save Version" -> "Run All (Save & Run All)" para ejecución en segundo plano (hasta 12-30h).

## Paso 4: Resultados
- Una vez finalizada la corrida, descargar los archivos de la sección "Output" de la versión guardada.
- Mover a `10_CLOUD_OUTPUT_INBOX`.
