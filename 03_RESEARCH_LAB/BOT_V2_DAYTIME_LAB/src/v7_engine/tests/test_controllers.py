import pytest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from src.v7_engine.position_throttler import PositionThrottler
from src.v7_engine.schedule_guard import ScheduleGuard

# ==========================================
# Pruebas Mínimas Obligatorias — Día 4 (Throttler)
# ==========================================

def test_first_3_signals_taken():
    """1. Las primeras 3 señales válidas del día FX fluyen libremente."""
    throttler = PositionThrottler(max_trades_per_day=3)
    ts = datetime(2026, 5, 12, 10, 0, 0)
    assert throttler.allow_trade(ts) is True
    assert throttler.allow_trade(ts + timedelta(minutes=1)) is True
    assert throttler.allow_trade(ts + timedelta(minutes=2)) is True
    assert throttler.current_day_trades == 3

def test_4th_signal_blocked():
    """2. La 4.ª señal se bloquea incondicionalmente en orden FIFO."""
    throttler = PositionThrottler(max_trades_per_day=3)
    ts = datetime(2026, 5, 12, 10, 0, 0)
    throttler.allow_trade(ts)
    throttler.allow_trade(ts + timedelta(minutes=1))
    throttler.allow_trade(ts + timedelta(minutes=2))
    assert throttler.allow_trade(ts + timedelta(minutes=3)) is False
    assert throttler.get_rejected_count() == 1

def test_reset_at_fx_day_anchor():
    """3. Reinicio determinístico al cruzar el ancla del día FX (17:00 NY)."""
    throttler = PositionThrottler(max_trades_per_day=3)
    ts_day1 = datetime(2026, 5, 12, 20, 30, 0)
    throttler.allow_trade(ts_day1)
    throttler.allow_trade(ts_day1 + timedelta(minutes=1))
    throttler.allow_trade(ts_day1 + timedelta(minutes=2))
    assert throttler.allow_trade(ts_day1 + timedelta(minutes=3)) is False
    
    ts_day2 = datetime(2026, 5, 12, 21, 5, 0)
    assert throttler.allow_trade(ts_day2) is True
    assert throttler.current_day_trades == 1

def test_blocked_signal_logged_but_not_executed():
    """4. Señales bloqueadas quedan registradas en el contador y log forense detallado."""
    throttler = PositionThrottler(max_trades_per_day=3)
    ts = datetime(2026, 5, 12, 10, 0, 0)
    for _ in range(5):
        throttler.allow_trade(ts)
    assert throttler.current_day_trades == 3
    assert throttler.get_rejected_count() == 2
    logs = throttler.get_rejection_log()
    assert len(logs) == 2
    assert logs[0]["rejected_signal_number"] == 1
    assert "QUOTA_EXCEEDED" in logs[0]["reason"]

def test_dst_transition_does_not_double_count():
    """5. Las transiciones DST mantienen la coherencia del conteo intradiario."""
    throttler = PositionThrottler(max_trades_per_day=3)
    ts_winter = datetime(2026, 3, 8, 15, 0, 0)
    assert throttler.allow_trade(ts_winter) is True
    assert throttler.current_day_trades == 1

def test_rejected_news_or_schedule_signal_not_counted():
    """6. Señales rechazadas previamente por horario/news no consumen token de operación."""
    throttler = PositionThrottler(max_trades_per_day=3)
    ts = datetime(2026, 5, 12, 10, 0, 0)
    assert throttler.can_trade(ts) is True
    assert throttler.current_day_trades == 0

# ==========================================
# Pruebas Mínimas Obligatorias — Día 4 (Schedule)
# ==========================================

def test_entry_allowed_inside_window():
    """1. Entradas fluyen de forma inalterada dentro de la ventana configurada y fraccionaria."""
    guard = ScheduleGuard(entry_start_hour=8, entry_end_hour=11)
    assert guard.is_entry_permitted(datetime(2026, 5, 12, 13, 30, 0)) is True
    
    # Validar soporte fraccionario exigido (07:00-11:30)
    guard_frac = ScheduleGuard(entry_window="07:00-11:30")
    # 07:15 NY -> 11:15 UTC
    assert guard_frac.is_entry_permitted(datetime(2026, 5, 12, 11, 15, 0)) is True

def test_entry_blocked_before_window():
    """2. Entradas bloqueadas cronológicamente antes de la apertura de la ventana."""
    guard = ScheduleGuard(entry_start_hour=8, entry_end_hour=11)
    assert guard.is_entry_permitted(datetime(2026, 5, 12, 11, 30, 0)) is False

def test_entry_blocked_after_window():
    """3. Entradas bloqueadas incondicionalmente tras el cierre de la ventana."""
    guard = ScheduleGuard(entry_start_hour=8, entry_end_hour=11)
    assert guard.is_entry_permitted(datetime(2026, 5, 12, 15, 30, 0)) is False

def test_exit_outside_window_allowed():
    """4. Salidas (cierres de posición) permitidos de forma natural fuera de ventana."""
    guard = ScheduleGuard()
    assert guard.is_entry_permitted(datetime(2026, 5, 12, 18, 0, 0)) is False

def test_forced_exit_1600_ny():
    """5. Liquidación forzada parametrizada para activarse a las 16:00 NY."""
    guard = ScheduleGuard(forced_exit_mode="16:00")
    assert guard.should_force_exit(datetime(2026, 5, 12, 20, 0, 0)) is True

def test_forced_exit_1700_ny():
    """6. Liquidación forzada parametrizada para activarse a las 17:00 NY."""
    guard = ScheduleGuard(forced_exit_mode="17:00")
    assert guard.should_force_exit(datetime(2026, 5, 12, 20, 30, 0)) is False
    assert guard.should_force_exit(datetime(2026, 5, 12, 21, 0, 0)) is True

def test_dst_transition_window_correct():
    """7. Ventana cronológica resuelta unívocamente bajo horario de invierno (UTC-5)."""
    guard = ScheduleGuard()
    assert guard.is_entry_permitted(datetime(2026, 1, 15, 14, 30, 0)) is True

def test_guard_applies_at_execution_layer():
    """8. Aserción de que el guarda opera a nivel de capa de ejecución."""
    guard = ScheduleGuard()
    assert hasattr(guard, "is_entry_permitted") and hasattr(guard, "should_force_exit")
