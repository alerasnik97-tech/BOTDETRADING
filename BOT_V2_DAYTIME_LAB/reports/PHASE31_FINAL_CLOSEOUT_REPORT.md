# PHASE31 FINAL CLOSEOUT REPORT

## Objetivo
Close PHASE31_PROP_FIRM_SURVIVAL_SIMULATOR and prepare PHASE32_FTMO_PAPER_EVALUATION_PLAN.

## Veredicto
PHASE31_PROP_FIRM_READY_CONSERVATIVE_RISK

## Reglas simuladas
- FTMO 2-Step: Challenge 10%, daily loss 5%, max loss 10%, min trading days 4, periodo ilimitado.
- Verification: target 5%, daily loss 5%, max loss 10%, min trading days 4.
- Funded: sin profit target; supervivencia bajo daily loss y max loss.

## Estrategias comparadas
- Phase25 autoridad: TP1.4 / BE0.4 / BF70.
- TP1.4_BE0.5_BF70: shadow comparator; unica diferencia BE 0.4R a BE 0.5R.

## Resumen de resultados
- A 0.75%: Challenge historico 98.53% pass, Verification 99.26% pass, funded 12m survival 100%, daily loss breach 0%, max loss breach 0%.
- A 1.00%: funded survival cae a 82.35% y aparecen breaches por daily loss.

## Riesgo recomendado
- Challenge paper: 0.50% a 0.75%.
- Verification paper: 0.50% a 0.75%.
- Funded paper: 0.50% prudente; 0.75% solo con confirmacion forward.
- Max not exceed: 0.75%.

## Por que 1.00% no es riesgo base
- Empieza riesgo material de breach por daily loss.
- La supervivencia funded 12m baja a 82.35%.
- El objetivo de fondeo es preservacion y continuidad, no velocidad.

## Por que 0.75% es techo defendible
- No tuvo daily loss breach ni max loss breach en Phase31.
- Mantiene pass rates altos en Challenge y Verification.
- Debe tratarse como techo, no como obligacion.

## Por que 0.50% es mas prudente en fondeada
- Reduce presion contra daily loss.
- Mejora margen operativo para errores y costos reales.
- Es mas compatible con supervivencia en cuenta fondeada.

## Daily loss como riesgo principal
El daily loss domina la decision de riesgo. El max loss no fue el primer punto de fallo en el riesgo recomendado.

## Non-win streak vs pure SL streak
La racha non-win mide sequia psicologica incluyendo BE. La racha pure SL mide perdida monetaria real. Para fondeo, la racha monetaria y el equity intraday importan mas que la sequia de TP.

## Limitaciones
- Equity intraday aproximada con MAE/SL proxy, no path tick-by-tick completo.
- Ledgers en R no incorporan comisiones/swaps exactos.
- Reglas FTMO/prop firm requieren revision manual antes de cualquier real.

## Estado operativo
- Phase25 sigue autoridad.
- TP1.4_BE0.5_BF70 sigue shadow comparator.
- No real.
- No MT5.
- No evaluacion real automatica.
- Phase32 sera paper evaluation plan.

## Siguiente paso unico
PHASE32_FTMO_PAPER_EVALUATION_PLAN only; no real and no MT5.
