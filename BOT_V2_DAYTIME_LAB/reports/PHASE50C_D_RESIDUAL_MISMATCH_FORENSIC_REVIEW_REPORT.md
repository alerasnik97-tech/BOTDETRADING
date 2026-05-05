# PHASE50C-D RESIDUAL MISMATCH FORENSIC REVIEW

Verdict: JAN2025_REAUDIT_NOT_READY_FOR_NEXT_MONTH

Residual trades: 2312, 2313, 2316, 2320, 2327

Counts:
- confirmed_matches: 15
- residual_mismatches: 5
- mismatches_explained: 4
- bugs: 0
- not_auditable: 1
- match_rate_confirmed: 0.75
- effective_reliability_score: 0.95

Classifications:
- 2312: bar=BE tick=SL category=ENTRY_PRICE_FEED_DIFFERENCE cause=El precio historico de entrada queda lejos del bid/ask tick cercano.
- 2313: bar=FORCED_CLOSE tick=NONE category=EXIT_TIME_MISSING_OR_UNCLEAR cause=No hubo TP/SL/BE por tick; el cierre forzado requiere capa de salida horaria.
- 2316: bar=BE tick=NONE category=DATA_GAP_OR_LOW_TICK_DENSITY cause=No hay ticks cargados en la ventana entry-10m a exit+10m.
- 2320: bar=SL tick=BE category=BE_SEQUENCE_DIFFERENCE cause=El tick muestra activacion de BE antes del toque posterior, mientras la barra queda en SL.
- 2327: bar=TP tick=NONE category=ENTRY_PRICE_FEED_DIFFERENCE cause=El precio historico de entrada queda lejos del bid/ask tick cercano.

Safety:
- MANIPULANTE not modified.
- Strategy, TP, BE, BF, schedules, MT5, orders, real, Exness, 2024 not touched.
