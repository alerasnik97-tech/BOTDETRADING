# MANIPULANTE — PROP FIRM PLAN

## Estrategia Autoridad
**MANIPULANTE (Phase25 + Global Weekend Hard Close)**

## Empresas Evaluadas

### FTMO 2-Step Swing (RUTA PRIMARIA)
- **Daily loss**: 5%
- **Max loss**: 10%
- **Weekend**: Permite mantener, pero MANIPULANTE cierra viernes 16:55 NY de todos modos (regla global para proteger capital).
- **Status**: Paper ready.
- **Risk Recomendado**: 0.50% base (Max defendible: 0.75%).
- **Observación**: Es la ruta más limpia y recomendada.

### FundedNext Stellar Lite 10K
- **Daily loss**: 4%
- **Max loss**: 8%
- **Weekend**: Prohíbe weekend holding. Resuelto automáticamente por la regla global de MANIPULANTE.
- **Status**: Viable paper/free-trial a 0.50%.
- **Risk Recomendado**: 0.50% (0.75% no es defendible matemáticamente aquí, 1.00% está descartado).
- **Observación**: Ruta alternativa de bajo coste, requiere ejecución estricta del cierre de viernes.

### FTMO 1-Step
- NO es preferida. El modelo 2-Step Swing ofrece más resiliencia.

## Reglas Universales (Independientes de la Firma)
1. **GLOBAL_HARD_CLOSE_BEFORE_MARKET_CLOSE**: Viernes 16:55 NY cierre obligatorio. Sin excepciones.
2. **Preservación de Capital**: Prioridad número uno. Si hay duda sobre una noticia o data, NO SE OPERA (Fail-Closed).
