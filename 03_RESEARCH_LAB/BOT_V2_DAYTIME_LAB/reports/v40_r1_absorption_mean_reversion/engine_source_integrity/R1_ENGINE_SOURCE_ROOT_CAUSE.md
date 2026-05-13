# ROOT CAUSE ANALYSIS: R1 ENGINE SOURCE INTEGRITY COMPROMISE

## 1. Descripción del Problema
Durante el preflight de la estrategia **R1 (EURUSD NY Open Absorption)**, se constató que el código fuente del motor de backtesting y utilidades causales (`src/v6_utils` y `src/v7_engine`) no correspondía a la versión canónica institucional, habiendo sido reconstruido parcialmente desde el contexto de memoria del agente.

## 2. Causa Raíz (Root Cause)
La pérdida de los archivos fuente `.py` se originó en la transición hacia la rama `clean-sync-branch` (creada en el commit `cc7eed4` para propósitos de sincronización institucional con GitHub). 
En dicho procedimiento de limpieza quirúrgica:
1. Se excluyeron o eliminaron los archivos fuente `.py` de las carpetas del motor en el working tree de Research Lab, dejando únicamente los archivos binarios compilados `.pyc` en las subcarpetas `__pycache__` como dependencias empaquetadas.
2. Al iniciar el desarrollo de la fase R1 en esta nueva rama, el intérprete de Python arrojó `ModuleNotFoundError` al no hallar los fuentes `.py` para la ejecución en modo interactivo/debug o al requerir ciertas firmas explícitas.
3. En lugar de ejecutar una restauración de Git desde la rama canónica anterior estable (`agent/research-manipulante4-sweep-quality`), el agente procedió a re-escribir los archivos faltantes (`bars.py`, `engine.py`, `cost_model.py`, etc.) basándose en resúmenes de contexto. Esto generó implementaciones sub-estándar y simplificadas que rompieron la paridad forense del motor.

## 3. Análisis de Impacto
- **Sobre M3 y M4**: Nulo en el histórico oficial, ya que dichas fases fueron selladas y validadas en sus respectivas ramas antes de la creación de la rama de sincronización limpia.
- **Sobre R1**: Crítico. Las métricas generadas durante el smoke test de 5 meses carecían de validez institucional al utilizar un modelo de costos simplificado y un motor que omitía las salvaguardas avanzadas de V7 (como las colas de eventos internas y auditoría MAE completa).
- **Sobre el resto del repositorio**: Nulo. El problema quedó estrictamente confinado al working tree local de `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/src/`. Las carpetas de producción (`01_CORE_PRODUCTION`) y datos (`05_MARKET_DATA_VAULT`) permanecen en estado READ-ONLY e inmutables.

## 4. Medidas de Prevención (Remediation & Guardrails)
1. **Regla de Inmutabilidad del Motor**: Prohibición absoluta de recrear código de `v6_utils` o `v7_engine` mediante herramientas de escritura sin un diff explícito aprobado contra la rama canónica.
2. **Restauración Exclusiva por Git**: Ante cualquier ausencia de dependencias del motor, la única vía de recuperación autorizada es `git restore` o el copiado forense desde commits inmutables de Git.
3. **Auditoría Pre-Sweep**: Incorporar un paso estricto de verificación de hashes del motor antes de iniciar cualquier micro-probe walk-forward.
