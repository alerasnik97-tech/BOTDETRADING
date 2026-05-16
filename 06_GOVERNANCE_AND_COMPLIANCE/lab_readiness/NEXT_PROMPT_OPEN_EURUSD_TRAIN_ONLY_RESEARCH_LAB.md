Actuá como ingeniero cuantitativo senior, auditor institucional de evidencia y oficial de ejecución del laboratorio de estrategias.

OBJETIVO GENERAL:
Ejecutar la apertura operacional del laboratorio:

EURUSD_OPEN_FOR_RESEARCH_TRAIN_ONLY

Esta tarea consiste en ejecutar el primer "Controlled Smoke Run" para certificar que el laboratorio produce evidencia limpia bajo los nuevos guards.

TAREAS:
1. Seleccionar un único candidato (ej: F06 - Bollinger Mean Reversion) del registro.
2. Configurar una corrida estrictamente TRAIN-ONLY (2015-2024).
3. Ejecutar el `lab_preflight.py` antes del run.
4. Generar el primer `run_id` y su correspondiente `MANIFEST.json`.
5. Validar que el resultado se guarde en `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/`.
6. Verificar que NO haya ningún trade o dato posterior a 2024-12-31.

REGLAS:
- NO optimizar.
- NO correr 60 estrategias.
- NO usar holdout.
- NO usar noticias.

PRÓXIMO PASO:
Si este smoke run es exitoso y el manifiesto es válido, el laboratorio quedará habilitado para el `F06 Clean Rerun` masivo de Fase 3.
