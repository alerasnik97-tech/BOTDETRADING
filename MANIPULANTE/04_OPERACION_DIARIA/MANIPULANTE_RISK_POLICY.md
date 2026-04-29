# MANIPULANTE — RISK POLICY

## Estrategia Autoridad
**PHASE25** = TP 1.4R / BE 0.4R / BF 70%

## Riesgo Recomendado
- **Base**: 0.50% por operación
- **Stress paper**: 0.60% (solo paper/demo)
- **Agresivo stress**: 0.75% (solo paper experimental, NO recomendado)
- **Prohibido**: 1.00% (rechazado)

## Risk Rules
1. Máximo 1 trade por día
2. No compounding
3. SL fijo en sweep extreme + 0.5 pips buffer
4. No mover SL manualmente (excepto BE trigger a 0.4R)
5. No añadir a posiciones perdedoras
6. No promediar
7. No revenge trading

## REGLA GLOBAL DE CIERRE VIERNES
**VIERNES 16:55 NY: CIERRE OBLIGATORIO — REGLA UNIVERSAL**

- No mantener operaciones abiertas durante el fin de semana
- No tomar señal si existe riesgo de no poder cerrar antes del fin de semana
- No override manual
- Aplica a FTMO, FundedNext, paper/demo, free trial y CUALQUIER prop firm
- No operar si hay duda

## Drawdown Limits
- Daily loss warning: -2R → pause
- Weekly drawdown: -2R → review
- Monthly drawdown: -3R → mandatory review
- Máx pérdidas consecutivas antes de pause: 4

## Prop Firm Compliance
- Daily loss limit (FTMO/FundedNext): respetado por risk sizing
- Max loss limit: respetado por risk sizing
- Weekend rule: cubierta por GLOBAL_HARD_CLOSE_BEFORE_MARKET_CLOSE
- News rule: cubierta por News Fortress (fail-closed)

## Validación Histórica
- PF post-policy: 2.8097
- Expectancy: 0.2824R
- DD: -5.5839R
- Weekend violations: 0
- Edge no degradado
