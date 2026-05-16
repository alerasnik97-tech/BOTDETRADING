# EURUSD Institutional Hypothesis Backlog (2026-05-16)

Este documento centraliza las estrategias algorítmicas extraídas de la investigación externa. Todas las estrategias operan bajo el protocolo **TRAIN-ONLY** y están diseñadas para la ventana de liquidez de Nueva York (07:00-19:00 NY).

## Taxonomía de Familias
- **MR**: Mean Reversion
- **VE**: Volatility Expansion
- **TP**: Trend Pullback
- **SD**: Session Dynamics
- **ED**: Event Driven
- **SE**: Statistical Edge
- **HY**: Hybrid

---

## 1. Anchor Elastic (MR-01)
- **Familia**: Mean Reversion
- **Gatillo**: Desviación extrema (>1.2 ATR o 1.8σ) respecto al ancla de precio medio (APM/VWAP) sin régimen de tendencia fuerte (ADX < 22).
- **Entrada**: Cierre de vela M1 de reingreso a la banda después del extremo.
- **Salida**: TP en APM (máx 1.5R), SL en extremo de excursión + buffer.
- **Scoring**: 95/100
- **Prioridad**: A (Implementar Smoke Test)

## 2. RV Shock Break (VE-01)
- **Familia**: Volatility Expansion
- **Gatillo**: Ruptura de canal Donchian de 30m tras compresión de Realized Volatility (rv15 <= p30).
- **Entrada**: Cierre M5 por encima de canal + shock de rv5 (>= 2x mediana).
- **Salida**: TP 2.0R, SL debajo de barra de ruptura.
- **Scoring**: 92/100
- **Prioridad**: A (Implementar Smoke Test)

## 3. Trend Day EMA Pullback (TP-01)
- **Familia**: Trend Pullback
- **Gatillo**: Calificación de Trend Day (07:00-09:30). Primer retroceso que toca EMA20 sin cerrar debajo de EMA50.
- **Entrada**: Cierre M1 confirmando giro a favor de tendencia.
- **Salida**: TP 2.0R o trailing bajo EMA20.
- **Scoring**: 91/100
- **Prioridad**: A (Implementar Smoke Test)

## 4. Europe Extreme Failure (SD-01)
- **Familia**: Session Dynamics
- **Gatillo**: Falsa extensión del extremo europeo (02:00-07:00 NY). Excursión < 0.08 ATR seguida de reingreso rápido.
- **Entrada**: Reingreso en máximo 3 velas M1.
- **Salida**: TP en midpoint europeo.
- **Scoring**: 89/100
- **Prioridad**: A (Implementar Smoke Test)

## 5. Post-News Stabilization (ED-01)
- **Familia**: Event Driven
- **Gatillo**: Continuación posnoticia tras normalización de spread (10-15 min después del release).
- **Entrada**: Ruptura de máximo de consolidación posnoticia en M1.
- **Salida**: TP 1.8R, SL lado opuesto de consolidación.
- **Scoring**: 88/100
- **Prioridad**: A (Implementar Smoke Test)

## 6. London Session H/L Breakout (SD-02)
- **Familia**: Session Dynamics
- **Gatillo**: Ruptura del máximo/mínimo de la sesión de Londres (03:00-07:00 UTC) con filtro de rango estrecho pre-sesión.
- **Entrada**: Cierre M1 por fuera del rango LDN.
- **Salida**: TP 2x ATR(14) M5, SL 1x ATR.
- **Scoring**: 87/100
- **Prioridad**: B

## 7. VWAP Stretch Reversion (MR-02)
- **Familia**: Mean Reversion
- **Gatillo**: Reversión desde ±2.25 SD del VWAP anclado.
- **Entrada**: RSI(14) cruzando 28/72 tras toque de banda extrema.
- **Salida**: TP en línea central VWAP o ±1 SD.
- **Scoring**: 86/100
- **Prioridad**: B

## 8. Institutional EMA Pullback (TP-02)
- **Familia**: Trend Pullback
- **Gatillo**: Retroceso a EMA50 o EMA200 en M15 durante tendencia definida (EMA50 > EMA200).
- **Entrada**: Cierre M5 confirmando pullback respetado.
- **Salida**: TP 2.5x riesgo, SL debajo de mínimo local.
- **Scoring**: 85/100
- **Prioridad**: B

## 9. BB Squeeze Momentum (VE-02)
- **Familia**: Volatility Expansion
- **Gatillo**: Estrechamiento drástico de Bandas de Bollinger (Bandwidth < SMA50) seguido de ruptura con ADX ascendente.
- **Entrada**: Cierre M15 fuera de bandas con ADX > 25.
- **Salida**: TP 1:2 RR, SL en SMA20.
- **Scoring**: 84/100
- **Prioridad**: B

## 11. Initial Balance Failure (SD-04)
- **Familia**: Session Dynamics
- **Gatillo**: Fracaso en sostener el rompimiento del Initial Balance (07:00-08:30 NY).
- **Entrada**: Cierre M5 reingresando al IB tras ruptura fallida.
- **Salida**: TP 1x ATR, SL lado opuesto del IB.
- **Scoring**: 82/100
- **Prioridad**: B

## 12. London Close Mean Reversion (MR-03)
- **Familia**: Mean Reversion
- **Gatillo**: Desviación del VWAP diario al cierre de Londres (11:30-12:00 NY).
- **Entrada**: Vela M1 alcista/bajista tras desviación > 5 pips.
- **Salida**: TP 8 pips, SL 10 pips.
- **Scoring**: 81/100
- **Prioridad**: C

## 13. Friday Reversion (SE-01)
- **Familia**: Statistical Edge
- **Gatillo**: Reversión por toma de beneficios los viernes al mediodía (12:00 NY) tras tendencia semanal > 0.8%.
- **Entrada**: Ejecución directa a las 12:00 NY con stop catastrófico.
- **Salida**: Cierre por tiempo a las 15:45 NY.
- **Scoring**: 80/100
- **Prioridad**: C

## 14. London Lunch Fade (SE-02)
- **Familia**: Statistical Edge
- **Gatillo**: Ruptura fallida del rango de almuerzo (11:45-13:30 London local).
- **Entrada**: Reingreso al rango en 2 min tras excursión > 0.12 ATR.
- **Salida**: TP en midpoint del rango, SL extremo de excursión.
- **Scoring**: 79/100
- **Prioridad**: C

## 15. GARCH Adaptive (HY-01)
- **Familia**: Hybrid
- **Gatillo**: Cambio de régimen detectado vía GARCH(1,1) o HMM sobre retornos M15.
- **Entrada**: Ruptura Donchian en tendencia o Reversión Bollinger en rango.
- **Salida**: Variable según régimen.
- **Scoring**: 78/100
- **Prioridad**: C (Alta complejidad)

## 16. Keltner Snapback (MR-04)
- **Familia**: Mean Reversion
- **Gatillo**: Ruptura de Canales de Keltner (20, 2.0 ATR) sin soporte de volumen/tick count.
- **Entrada**: Cierre M5 dentro de canales tras excursión.
- **Salida**: TP en EMA central, SL extremo previo.
- **Scoring**: 77/100
- **Prioridad**: C

## 17. ATR Compression-Expansion (VE-03)
- **Familia**: Volatility Expansion
- **Gatillo**: Ciclo de ATR (percentil < 30) seguido de pico explosivo.
- **Entrada**: Cierre M5 fuera de rango de compresión con ATR ascendente.
- **Salida**: TP proyectado 2.0R.
- **Scoring**: 76/100
- **Prioridad**: C

## 18. Fibonacci 61.8% Pullback (TP-03)
- **Familia**: Trend Pullback
- **Gatillo**: Retroceso al nivel 61.8% de un impulso M5 con ADX(M15) > 25.
- **Entrada**: Rechazo confirmado en M5 tras tocar nivel Fib.
- **Salida**: TP 1.5x SL, SL bajo swing previo.
- **Scoring**: 75/100
- **Prioridad**: C

## 19. Breakout-Retest Structural (TP-04)
- **Familia**: Trend Pullback
- **Gatillo**: Ruptura de H/L de sesión previa seguida de retesteo limpio (±3 pips).
- **Entrada**: Vela M1 confirmatoria tras retesteo sin sombras profundas.
- **Salida**: TP 15 pips, SL 10 pips.
- **Scoring**: 74/100
- **Prioridad**: C

## 20. Post-News Volatility Reversion (ED-02)
- **Familia**: Event Driven
- **Gatillo**: Reversión de spike inicial (ATR > 2x pre-news) tras 10 min del release.
- **Entrada**: Vela M1 de giro que rompe el máximo/mínimo de la vela de shock.
- **Salida**: TP 0.75x ATR_pre, SL 1.0x ATR_pre.
- **Scoring**: 73/100
- **Prioridad**: C

## 21. PNMC-15 Momentum (ED-03)
- **Familia**: Event Driven
- **Gatillo**: Estabilización de 15 min posnoticia (StdDev < 0.5x ATR).
- **Entrada**: Ruptura del rango de estabilización en dirección del shock inicial.
- **Salida**: TP 2x ATR(15m), SL 1x ATR(15m).
- **Scoring**: 72/100
- **Prioridad**: C

## 22. NY Mid-Day Volatility Expansion (VE-04)
- **Familia**: Volatility Expansion
- **Gatillo**: Ruptura de rango estrecho de mediodía (11:30-12:00 NY).
- **Entrada**: Cierre M1 fuera del rango con spread < 1.5 pips.
- **Salida**: TP 2x ATR, SL 1x ATR.
- **Scoring**: 71/100
- **Prioridad**: C

## 23. HVFTF Trend Following (HY-02)
- **Familia**: Hybrid
- **Gatillo**: Giro de SuperTrend(10,3) en M5 filtrado por ADX y SuperTrend M15.
- **Entrada**: Apertura de vela M5 tras confirmación de tendencia.
- **Salida**: TP 2.5R, SL en valor SuperTrend.
- **Scoring**: 70/100
- **Prioridad**: C

## 24. M15 Trend + VWAP MR (HY-03)
- **Familia**: Hybrid
- **Gatillo**: Desviación de VWAP (>3 pips) a favor de la tendencia M15 (MA10 > MA30).
- **Entrada**: Vela reversal M1 tras desviación extrema.
- **Salida**: TP1 en VWAP, TP2 extensión.
- **Scoring**: 69/100
- **Prioridad**: C

## 25. Programmable Structure Break + Fill (SD-05)
- **Familia**: Session Dynamics
- **Gatillo**: Ruptura de swing H/L intradía seguida de llenado de imbalance estructural.
- **Entrada**: Cierre M5 tras mitigación de zona de valor.
- **Salida**: Variable según estructura.
- **Scoring**: 68/100
- **Prioridad**: C
