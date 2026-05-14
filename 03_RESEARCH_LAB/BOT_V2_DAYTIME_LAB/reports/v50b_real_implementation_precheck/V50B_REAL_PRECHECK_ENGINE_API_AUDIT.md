# V50B REAL PRECHECK ENGINE API AUDIT

**Objetivo**: Documentar el proceso de integracin con `UnifiedV7Engine`.

## Puntos de Entrada
- **`execute_signal(side, signal_bar_close, ticks_after, ...)`**: 
  - Realiza validaciones de `TestLeakageGuard`.
  - Verifica `ScheduleGuard`, `NewsCalendar`, `FtmoCompliance` y `PositionThrottler`.
  - Retorna un `FillResult` o una razón de rechazo.
- **`close_position_with_costs(fill, sl_price, tp_price, ticks_during, ...)`**:
  - Simula la salida de la operacin.
  - Aplica el `CostModel` (spread, comisión, swap).
  - Retorna un `TradeRecord`.

## Requisitos de Integracin
- **Seİales**: Deben tener `signal_bar_close` (pd.Timestamp) real.
- **Datos**: Requiere `ticks_after` y `ticks_during` (pd.DataFrame) reales del Vault.
- **Causality**: Cada llamada a `execute_signal` queda registrada en el `causal_log` del motor.

## Blindaje
- El motor invoca `self.leak_guard.verify_timestamp(signal_bar_close)` en cada ejecucin.
- Cualquier intento de usar fechas >= 2025 dispararǭ una `TestLeakageViolation`.

**Veredicto**: API clara y lista para integracin real fuera del core.
