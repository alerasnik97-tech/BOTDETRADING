# NEXT_PROMPT: CLEAN_SMOKE_OUTPUTS_AND_REOPEN_LAB

## Contexto
El reporte `SMOKE_RUN_INCIDENT_AUDIT_20260516.md` ha identificado que el commit `32420260` contaminó el historial de Git con archivos de salida pesados y mutó el motor sin una auditoría aislada. Aunque la rama actual de gobernanza está limpia, el repositorio local aún contiene archivos que podrían ser stageados accidentalmente.

## Instrucciones para Antigravity
1. **Quarantine local reports**: Mover el contenido de `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/eurusd_train_only_smoke/` a `07_BACKUPS/quarantine_smoke_outputs_20260516/` para evitar `git add .` accidentales.
2. **Verify .gitignore**: Asegurarse de que `*.csv`, `*.png` y `*.zip` estén correctamente ignorados en las subcarpetas de reportes para evitar recurrencias.
3. **Engine Lockdown Audit**: Realizar una revisión final de las líneas 703-708 de `03_RESEARCH_LAB/research_lab/engine.py` para asegurar que el fallback de señal es estable y no introduce riesgos de ejecución.
4. **Official Lab Re-Opening**: Una vez limpio el entorno local y auditado el motor, declarar el laboratorio como **STABLE_FOR_STRATEGY_CONVERSION**.

## Prohibiciones
- NO borrar evidencias locales sin antes asegurar el backup en 07_BACKUPS.
- NO tocar datos crudos en 05_MARKET_DATA_VAULT.
- NO usar `git add .`. Usar stage selectivo por archivo.
