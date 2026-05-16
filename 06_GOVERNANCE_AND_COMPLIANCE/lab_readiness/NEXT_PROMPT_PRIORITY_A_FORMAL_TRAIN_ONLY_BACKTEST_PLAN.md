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

## Protocolo
1. **Dossier Individual**: Cada estrategia debe tener su propio reporte exhaustivo (curvas de equidad, drawdown, estadísticas mensuales/anuales).
2. **Fixed Params**: Usar `DEFAULT_PARAMS`. No optimizar todavía.
3. **Reproducibilidad**: Registrar run_id, commit y hashes de configuración.
4. **Data Governance**: Validar que no se acceda a datos de 2025/2026.

## Tareas
- Ejecutar el backtest formal de 10 años para la primera estrategia de la lista.
- Analizar la estabilidad del performance a través de los años.
- Documentar el comportamiento en diferentes regímenes de volatilidad detectados en el periodo train.
- NO realizar selección de champion hasta que las 4 tengan su dossier completo.

## Restricciones
- No holdout.
- No news.
- No high precision.
- No modifications to engine or data.
