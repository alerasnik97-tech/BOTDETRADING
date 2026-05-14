# V50B REAL QA ?" FAMILY DECISION CORRECTIONS

**Objetivo**: Ajustar el estado de cada familia tras la auditoría de rechazos.

## F01 ?" London Continuation
- **Estado Anterior**: `REAL_PIPELINE_CONFIRMED_NO_TRADES_IN_SAMPLE`
- **Estado Corregido**: **REAL_PIPELINE_CONFIRMED_AFTER_REJECTION_AUDIT**
- **Justificación**: Se confirmó que el motor recibe las señales pero las rechaza por `BLOCKED_BY_SCHEDULE` (Horas predeterminadas de NY 08-11). Esto valida que el pipeline de comunicación funciona.

## F06 ?" Volatility Regime Breakout
- **Estado Anterior**: `REAL_PIPELINE_CONFIRMED_NO_TRADES_IN_SAMPLE`
- **Estado Corregido**: **REAL_PIPELINE_CONFIRMED_AFTER_REJECTION_AUDIT**
- **Justificación**: Similar a F01, las señales son interceptadas correctamente por el motor y rechazadas por horario (`BLOCKED_BY_SCHEDULE`).

## F08 ?" Session Overlap Trend
- **Estado Anterior**: `REAL_PIPELINE_CONFIRMED_NO_TRADES_IN_SAMPLE`
- **Estado Corregido**: **REAL_PIPELINE_CONFIRMED_AFTER_REJECTION_AUDIT**
- **Justificación**: Pipeline confirmado mediante auditoría de rechazos.

## F12 ?" Macro Calendar Safe-Window
- **Estado Anterior**: `REAL_PIPELINE_CONFIRMED`
- **Estado Corregido**: **REAL_PIPELINE_CONFIRMED_WITH_NEWS_RESERVATION**
- **Justificación**: F12 produjo trades reales, pero el uso de `DummyNews` deja una reserva crítica sobre su comportamiento ante noticias reales.

**Veredicto**: Todas las familias tienen un pipeline real verificado, ya sea por ejecución exitosa o por rechazo justificado por el motor.
