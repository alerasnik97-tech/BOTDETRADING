# PHASE32 FTMO PAPER EVALUATION PLAN

## Objetivo
Ejecutar una evaluacion paper tipo FTMO sin real, sin MT5 y sin broker, usando Phase25 como autoridad y TP1.4_BE0.5_BF70 como shadow ledger.

## Cuenta simulada
- Modelo: FTMO paper Challenge / Verification / funded survival.
- Fase: paper only.
- Profit target simulado Challenge: 10%.
- Profit target simulado Verification: 5%.
- Max daily loss simulado: 5%.
- Max loss simulado: 10%.
- Min trading days: 4.
- Reset diario: 00:00 CE(S)T proxy / Europe-Prague.
- Equity intraday: usar MAE/SL proxy hasta tener path real mas preciso.

## Estrategias
- Ledger A autoridad: Phase25 TP1.4 / BE0.4 / BF70.
- Ledger B shadow: TP1.4 / BE0.5 / BF70.
- No reemplazo automatico.

## Riesgo
- Escenario prudente: 0.50% por trade.
- Escenario techo paper: 0.75% por trade.
- 1.00% prohibido como base.

## Gates obligatorios
- News Fortress debe estar en ALLOW.
- Data Quality Mask debe estar en ALLOW.
- No trade si no hay ALLOW.
- No trade si hay duda.
- No trade si hay bloqueo por daily loss.
- No trade fuera de horario.
- Reglas FTMO reales deben revisarse manualmente antes de cualquier real.
