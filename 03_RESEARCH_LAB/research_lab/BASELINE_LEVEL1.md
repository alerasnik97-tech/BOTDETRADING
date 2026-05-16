# Baseline oficial Nivel 1

Estado congelado antes de Nivel 2:

- sistema: `APTO PARA TESTEAR ESTRATEGIAS`
- noticias: `OFF`
- loader / data: `APROBADO`
- horario / timezone / DST: `APROBADO`
- motor base: `APROBADO`
- tests: `APROBADO`

Limitación estructural explícita:

- la fuente histórica operativa sigue siendo `OHLC BID`
- no hay `ASK` histórico real
- spread, slippage y comisión se modelan de forma sintética y auditable

Regla de conservación:

- `normal_mode` debe preservar el comportamiento aprobado del Nivel 1
- cualquier cambio de realismo adicional debe entrar como capa opcional
- no se modifica la baseline salvo necesidad crítica demostrable
