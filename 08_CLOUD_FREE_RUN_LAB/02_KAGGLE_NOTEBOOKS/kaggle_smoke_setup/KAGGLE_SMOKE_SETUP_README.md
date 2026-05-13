# KAGGLE_SMOKE_SETUP_README

Este documento detalla el procedimiento para realizar un "Smoke Test" (prueba de humo) en Kaggle. El objetivo es validar que el entorno es apto para futuras corridas sin ejecutar lógica de trading todavía.

## Objetivos
1. Validar el acceso seguro al repositorio privado (vía Kaggle Secrets).
2. Verificar la estructura de carpetas necesaria.
3. Comprobar la capacidad de escritura y generación de artefactos en `/kaggle/working`.
4. Asegurar que no hay filtración de secretos en el entorno.

## Procedimiento
1. Crear un Personal Access Token (PAT) en GitHub con permisos de lectura.
2. Añadirlo a Kaggle como un Secret llamado `GH_TOKEN`.
3. Copiar las celdas de `KAGGLE_NOTEBOOK_CELLS.md` en un nuevo Notebook de Kaggle.
4. Ejecutar y verificar los resultados.
