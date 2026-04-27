# VPS FORWARD GATE PLAN (DEMO ONLY)

Este plan define las métricas y reglas para la fase de validación en tiempo real de los candidatos Phase 7 y Phase 8.

## 1. Reglas de Ejecución
- **Entorno:** VPS con MetaTrader 5 (Cuenta DEMO).
- **Símbolo:** EURUSD.
- **Riesgo:** 1% por operación.
- **Modificaciones:** PROHIBIDAS. Se respetan las reglas de backtest al 100%.

## 2. Checkpoints y Métricas
Se realizarán cortes de evaluación cada N trades.

### Métricas por Corte:
- **Sample:** Número de operaciones.
- **Profit Factor:** Ratio de ganancias/pérdidas.
- **Expectancy:** Ganancia media por trade en R.
- **Max Drawdown:** Caída máxima de la equidad.
- **Execution Error:** Número de fallos técnicos (spread, noticias, desconexión).

## 3. Criterios de Suspensión (STOP)
- **Account Mode Change:** Si se detecta cuenta real o `allow_live=true`.
- **Divergencia Crítica:** Si el resultado real difiere significativamente del backtest (±30% en métricas clave).
- **Violación de News Guard:** Si se abre un trade durante una noticia bloqueada.
- **Riesgo Descontrolado:** Trade abierto sin SL o con lotaje incorrecto.

## 4. Veredictos de Gate
- **FORWARD_CONTINUE:** Métricas dentro de rango. Seguir hasta el siguiente checkpoint.
- **FORWARD_REVIEW_REQUIRED:** Anomalías leves detectadas. Auditar logs.
- **FORWARD_REJECTED:** Estrategia no replica el edge en tiempo real. Volver a laboratorio.
- **FORWARD_READY_FOR_NEXT_GATE:** Meta de trades alcanzada con éxito. Preparar Live Gate (futuro).
