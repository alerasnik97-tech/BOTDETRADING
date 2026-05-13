# PLAN DE RESTAURACIÓN CANÓNICA: R1 ENGINE SOURCE INTEGRITY

## Decisión Adoptada
**A. RESTORE_CORE_FROM_GIT**

Se determina que el motor central de backtesting cuantitativo (`UnifiedV7Engine` y librerías `v6_utils`) debe regresar inmediatamente a su estado canónico inmutable para asegurar la estricta comparabilidad forense de los resultados contra el benchmark institucional (Fases M3/M4).

## Justificación
Las modificaciones y recreaciones introducidas en la sesión previa no corresponden a mejoras algorítmicas planificadas, sino a re-implementaciones de emergencia ante un error de rutas en la rama de sincronización. Mantener un motor divergente destruye la confianza en los costos de FTMO, slippage y la causalidad estricta de la simulación.

## Protocolo de Ejecución
1. **Limpieza del Working Tree**: Eliminar las versiones `.py` no confiables presentes en `src/v6_utils` y `src/v7_engine`.
2. **Restauración Canónica**: Ejecutar el comando de extracción directa desde el árbol inmutable de la rama estable de la fase previa (`agent/research-manipulante4-sweep-quality`), la cual contiene la última certificación completa de paridad del motor:
   ```powershell
   git checkout agent/research-manipulante4-sweep-quality -- 03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/src/v6_utils 03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/src/v7_engine
   ```
3. **Preservación de Lógica R1**: Los detectores específicos de la estrategia (`src/R1/r1_detector.py` y `src/R1/r1_levels.py`) se mantendrán intactos, ya que no violan fronteras del motor y operan puramente sobre DataFrames OHLC causales externos.
4. **Adaptación del Orquestador**: Modificar `run_r1_micro_probe.py` para que consuma las firmas exactas del motor restaurado (ej. inyectar correctamente los objetos `CostModelConfig` y parámetros de calendario nativos sin alterar el código de `engine.py`).

## Criterios de Éxito
- Hashes idénticos en los fuentes del motor respecto a la rama estable.
- Aprobación del 100% de la suite de tests targeted de V7.
