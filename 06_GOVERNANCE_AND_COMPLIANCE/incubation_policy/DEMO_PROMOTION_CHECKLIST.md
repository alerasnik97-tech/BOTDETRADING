# DEMO PROMOTION CHECKLIST

Para pasar de una fase de Paper Trading a una Demo Seria / Evaluación FTMO, se deben marcar todos los puntos siguientes:

## Historial de Incubación
- [ ] Mínimo de 8 a 12 semanas de Forward Test completadas.
- [ ] Mínimo de trades estadísticamente representativo (según la frecuencia de la estrategia).
- [ ] Net Expectancy positiva durante el periodo de incubación.
- [ ] Drawdown observado en Forward dentro de los parámetros esperados del Backtest.

## Estabilidad Operativa
- [ ] Cero fallos técnicos críticos no resueltos en las últimas 4 semanas.
- [ ] El Kill Switch fue probado y funciona correctamente.
- [ ] Los logs de ejecución están completos y auditados.
- [ ] Discrepancia Backtest vs Forward Test analizada y aceptable (< 20% desviación).

## Gestión de Riesgos
- [ ] Módulo News Fail-Close verificado con eventos reales.
- [ ] Registro de slippage real dentro de los márgenes de rentabilidad.
- [ ] Alertas (Telegram/Email) configuradas y operativas.

## Gobernanza y Aprobación
- [ ] Shadow Ledger actualizado al día.
- [ ] Revisión de resultados realizada por ChatGPT/Claude para detectar sesgos.
- [ ] **APROBACIÓN EXPLÍCITA DEL USUARIO**.

> [!CAUTION]
> Queda estrictamente prohibido pasar directamente de Backtest a Cuenta Real o Fondeo sin completar esta checklist.
