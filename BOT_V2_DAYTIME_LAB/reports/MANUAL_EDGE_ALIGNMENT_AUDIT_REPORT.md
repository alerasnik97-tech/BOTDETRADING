# MANUAL EDGE ALIGNMENT AUDIT REPORT (OFFICIAL)

## 1. Executive Summary
Esta auditoría compara la operativa manual ("Barrido 1H + CHoCH 3M") del usuario contra la implementación automatizada del bot. 

**Hallazgo Principal**: La operativa manual concentra su ventaja en una ventana de tiempo extremadamente estrecha (08:00 - 11:00 NY) y utiliza una definición de "Barrido" más dinámica que la del bot.

## 2. Aclaración Discrepancia Profit Factor (1.64 vs 1.53)
Se ha auditado el origen de la cifra 1.64 reportada anteriormente:
- **PF 1.64 (Bruto)**: Calculado como `Suma(Beneficios Reales) / Abs(Suma(Pérdidas Reales))` usando la columna `rPnL` del archivo `analytics.csv`. Este valor refleja el retorno monetario real.
- **PF 1.53 (Normalizado R)**: Calculado asignando una pérdida de **-1.0 R** a cada SL y usando el `avgRiskReward` para los TPs. 
- **Conclusión**: La diferencia sugiere que el usuario promedia retornos levemente inferiores en R que los nominales de TP, o que el tamaño de las pérdidas reales fluctúa por encima de la unidad de riesgo teórica. La Phase 18 utilizará el modelo de R normalizado para mayor rigor estadístico.

## 3. Performance Metrics (Manual - Normalizado)
- **Muestra**: 841 trades (2020-2025).
- **Expectancy**: +0.25 R por trade.
- **Concentración Temporal**: 100% de los trades en 08:00–11:00 NY.

## 4. Matching Analysis (Manual vs Auto)
- **Matches Exactos**: 66 trades (7.8%).
- **Ganadores Perdidos (Bot Missed)**: **287 trades**.
- **Hipótesis**: Estos 287 ganadores ocurren por barridos de fractales H1 que el bot actual no detecta.

## 5. Proposed Action Plan (Phase 18)
1. Implementar fractales H1 dinámicos.
2. Restringir ventana a 08:00-11:00 NY.
3. Validar captura de los 287 ganadores.
