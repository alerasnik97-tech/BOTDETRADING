# Resumen de Auditoría e Implementación: VSE V0.1

He completado la fase de implementación y baseline oficial de la estrategia **VSE_EURUSD_M5_NY_V0_1**.

## A. Archivos creados o modificados
- `research_lab/strategies/strategy_vse.py` (Nuevo: Lógica Squeeze + Limit Entry)
- `research_lab/strategies/__init__.py` (Modificado: Registro de estrategia)
- `research_lab/config.py` (Modificado: Habilitación en CLI)
- `research_lab/tests/test_strategy_vse.py` (Nuevo: Suite de validación)

## B. Resumen de implementación
- **Squeeze Engine**: Detección de BB(20,2) dentro de KC(20, 1.5) por 5+ velas.
- **Limit Entry Mod**: Implementación de una orden limitada "frozen" en el nivel de la banda rota de la vela de ruptura.
- **Gestión**: SL estructural en banda opuesta congelada. TP fijo a 0.8 ATR. BE a 0.4 ATR.
- **Frecuencia**: Limitado estrictamente a 1 trade por día.

## C. Tests nuevos
- `test_vse_valid_long_retest`: **PASS** (Verificada entrada por retest).
- `test_vse_tp_cancellation_works`: **PASS** (Verificada cancelación si toca TP antes de entry).
- `test_vse_body_filter_fail`: **PASS** (Verificado rechazo de velas con cuerpo < 50%).

## D. Resultados Baseline en normal_mode

| Métrica | Dev (2020-23) | Val (2024) | Hold (2025) |
| :--- | :--- | :--- | :--- |
| **Total Trades** | 109 | 22 | 19 |
| **Net PnL (Est.)** | -15.56% | -4.26% | -3.30% |
| **Profit Factor** | **0.18** | **0.09** | **0.14** |
| **Win Rate** | 33.9% | 27.3% | 42.1% |
| **Max Drawdown** | 15.6% | 4.3% | 3.6% |
| **Qty SL** | 27 | 5 | 5 |
| **Qty TP** | 56 | 9 | 11 |
| **Qty Max Hold** | 26 | 8 | 3 |

## E. Desglose por lado (Dev 2020-2023)
- **Longs**: 87 trades, WR 36.8%, PF 0.23
- **Shorts**: 22 trades, WR 22.7%, PF 0.06

## F. Resultado de decisión
**NACE MUERTA.**

## G. Sin resultados en conservative / high_precision
No se ejecutan por no superar el umbral mínimo de viabilidad en `normal_mode` (PF < 1.0).

## H. Explicación de alcance
Se modificaron `__init__.py` y `config.py` exclusivamente para registrar la estrategia. El motor (`engine.py`) **no fue tocado**, ya que la lógica de "Limit Order" se resolvió íntegramente dentro del módulo de la estrategia mediante una ventana de búsqueda retrospectiva controlada por `limit_expiry_bars`.

## I. Veredicto honesto
La hipótesis de "Expansion post-Squeeze con retest" no sobrevive a la fricción operativa. El target de `0.8 ATR` (~5-7 pips) es demasiado vulnerable al spread y comisiones, mientras que el SL estructural en la banda opuesta suele ser más amplio, destruyendo el R:R. El "whipsaw" no se resuelve con la entrada limitada; simplemente ocurre un poco después del fill. VSE no tiene edge en este entorno.
