# NEXT PROMPT: EURUSD Priority A Formal Train-Only Backtest

Actúa como **Institutional Quantitative Researcher**. Tu misión es ejecutar la fase de **Evaluación Formal en Entrenamiento** para las estrategias Priority A.

## Objetivo
Generar un dossier de performance completo para cada una de las 4 estrategias en todo el periodo de entrenamiento (2015-2024), una por una.

## Alcance
- **Dataset**: EURUSD M1 Train (2015-01-01 a 2024-12-31).
- **Estrategias**:
  1. `mr01_anchor_elastic`
  2. `mr02_vwap_stretch_reversion`
  3. `tp01_london_ny_momentum_pullback`
  4. `ve_orb_volatility_expansion`

## Reglas de Ejecución y Rigor Cuantitativo
1. **Una Estrategia por Vez**: No ejecutar sweeps globales ni buscar campeones de forma simultánea. Se requiere aislamiento total.
2. **Fixed Params**: Utilizar estrictamente `DEFAULT_PARAMS`. Queda **estrictamente prohibida** cualquier optimización paramétrica o búsqueda de rejilla en esta fase.
3. **Dossier Individual Exhaustivo**: Generar curvas de equidad detalladas, métricas de drawdown, y matrices de performance mensual y anual para cada estrategia.
4. **Evaluación Multicosto**: Evaluar cada estrategia bajo los perfiles de costos autorizados (`base`, `conservative`, `stress`) para medir la sensibilidad a comisiones, spreads ampliados y slippage de rollover.
5. **Estabilidad Anual**: Analizar la consistencia de los retornos año con año en los 10 años de entrenamiento para evitar el sesgo por sub-periodos excepcionales.
6. **Distribución de Trades**: Evaluar la estabilidad de la frecuencia operativa (que no se concentren todos los trades en un solo mes o año).
7. **No Selección Simplista por PF**: No declarar edges definitivas basadas puramente en el Profit Factor (PF) global; se debe sopesar la esperanza matemática, la tasa de acierto (WR) y la desviación máxima.

## Gobernanza e Integridad de Datos
1. **Blindaje de Holdout (2025-2026)**: Queda **estrictamente prohibido** mapear, leer o cargar cualquier dato posterior al 31 de diciembre de 2024.
2. **Reconciliación de Metadatos**: El runner formal debe registrar con precisión matemática absoluta en el manifiesto (`RUN_MANIFEST.json`) y en los snapshots de configuración el rango completo real de 10 años (`2015-01-01` a `2024-12-31`). No se aceptarán discrepancias documentales en la fase formal.
3. **News Fail-Closed**: El filtro de noticias debe configurarse explícitamente en modo `enabled = false` (desactivado por defecto) a menos que se apruebe una fase de reconstrucción específica.
4. **Archivos de Salida en Git**: Asegurar que las reglas de exclusión en `.gitignore` sigan activas para no subir CSVs de trades pesados al repositorio remoto. Solo se permite subir manifiestos, configuraciones y resúmenes JSON livianos.

## Decisiones y Clasificación
- No seleccionar ningún "Champion" ni promover a "Incubation" hasta que las 4 estrategias Priority A cuenten con su Dossier Formal completo.
