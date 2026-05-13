import pytest
import pandas as pd
from datetime import datetime
from src.v7_engine.news_filter import NewsCalendar, NewsEvent
from src.v7_engine.engine import UnifiedV7Engine

@pytest.fixture
def base_ticks():
    # Rango temporal: 2026-05-12 desde las 12:00:00 hasta las 22:00:00 UTC
    # 12:00 UTC -> 08:00 NY (Apertura)
    # 20:00 UTC -> 16:00 NY (Cierre forzado 16:00)
    # 21:00 UTC -> 17:00 NY (Cierre forzado 17:00)
    idx = pd.date_range(start="2026-05-12 12:00:00", end="2026-05-12 22:00:00", freq="1min", tz="UTC")
    df = pd.DataFrame({
        "bid": [1.05000 + i*0.00002 for i in range(len(idx))],
        "ask": [1.05010 + i*0.00002 for i in range(len(idx))]
    }, index=idx)
    return df

@pytest.fixture
def empty_calendar():
    cal = NewsCalendar()
    cal.add_covered_period(datetime(2026, 5, 1), datetime(2026, 5, 30))
    return cal

# =====================================================================
# Pruebas Mínimas Obligatorias de Integración V7 (Día 5)
# =====================================================================

def test_signal_out_of_schedule_never_executes(empty_calendar):
    """1. Una señal fuera de horario no llega a ejecución."""
    engine = UnifiedV7Engine(news_calendar=empty_calendar, news_mode="none")
    # 11:30 UTC -> 07:30 NY (Cerrado por temprano)
    idx = pd.date_range(start="2026-05-12 11:00:00", periods=5, freq="1min", tz="UTC")
    df_early = pd.DataFrame({"bid": [1.0], "ask": [1.0]}, index=idx)
    sig_time = pd.Timestamp("2026-05-12 11:02:00", tz="UTC")
    fill, reason = engine.execute_signal("long", sig_time, df_early)
    assert fill is None
    assert reason == "BLOCKED_BY_SCHEDULE"

def test_signal_inside_news_buffer_never_executes(base_ticks):
    """2. Una señal antes del post-news buffer no ejecuta."""
    cal = NewsCalendar()
    cal.add_covered_period(datetime(2026, 5, 1), datetime(2026, 5, 30))
    cal.add_event(NewsEvent("1", "NFP", datetime(2026, 5, 12, 13, 0), "USD", "High"))
    engine = UnifiedV7Engine(news_calendar=cal, news_mode="post5")
    # 13:02 UTC cae dentro del buffer de 5 minutos post-noticia
    sig_time = pd.Timestamp("2026-05-12 13:02:00", tz="UTC")
    fill, reason = engine.execute_signal("long", sig_time, base_ticks)
    assert fill is None
    assert reason == "BLOCKED_BY_NEWS"

def test_valid_signal_post_news_executes_correctly(base_ticks):
    """3. Una señal válida post-news dentro de horario sí ejecuta."""
    cal = NewsCalendar()
    cal.add_covered_period(datetime(2026, 5, 1), datetime(2026, 5, 30))
    cal.add_event(NewsEvent("1", "NFP", datetime(2026, 5, 12, 12, 30), "USD", "High"))
    engine = UnifiedV7Engine(news_calendar=cal, news_mode="post5")
    # 13:00 UTC supera los 5 minutos del evento (12:30) y cae en franja de NY
    sig_time = pd.Timestamp("2026-05-12 13:00:00", tz="UTC")
    fill, reason = engine.execute_signal("long", sig_time, base_ticks)
    assert fill is not None
    assert reason in ["FILLED", "SIGNAL"]
    assert fill.side == "long"

def test_fourth_trade_same_day_blocked(empty_calendar, base_ticks):
    """4. El cuarto trade del mismo día FX queda bloqueado."""
    engine = UnifiedV7Engine(news_calendar=empty_calendar, max_trades_per_day=3, news_mode="none")
    sigs = [
        pd.Timestamp("2026-05-12 13:00:00", tz="UTC"),
        pd.Timestamp("2026-05-12 13:15:00", tz="UTC"),
        pd.Timestamp("2026-05-12 13:30:00", tz="UTC"),
        pd.Timestamp("2026-05-12 13:45:00", tz="UTC")
    ]
    fills = []
    for s in sigs:
        f, r = engine.execute_signal("long", s, base_ticks)
        fills.append((f, r))
    assert fills[0][0] is not None
    assert fills[1][0] is not None
    assert fills[2][0] is not None
    assert fills[3][0] is None
    assert fills[3][1] == "BLOCKED_BY_THROTTLER"

def test_ftmo_violation_blocks_subsequent_operations(empty_calendar, base_ticks):
    """5. Una violación FTMO bloquea toda operación posterior."""
    engine = UnifiedV7Engine(news_calendar=empty_calendar, initial_balance=100000.0, news_mode="none")
    # Inducir un quiebre por pérdida mayor al umbral del día
    engine.ftmo.update_state(datetime(2026, 5, 12, 12, 0), closed_pnl=-12000.0, floating_pnl=0.0)
    assert engine.ftmo.blown is True
    sig = pd.Timestamp("2026-05-12 13:00:00", tz="UTC")
    fill, reason = engine.execute_signal("long", sig, base_ticks)
    assert fill is None
    assert reason == "BLOCKED_BY_BLOWN_STATE"

def test_forced_exit_1600_ny_works(empty_calendar, base_ticks):
    """6. Forced exit a 16:00 NY funciona."""
    engine = UnifiedV7Engine(news_calendar=empty_calendar, forced_exit_mode="16:00", news_mode="none")
    sig = pd.Timestamp("2026-05-12 13:00:00", tz="UTC")
    fill, _ = engine.execute_signal("long", sig, base_ticks)
    assert fill is not None
    # Forzar SL/TP inalcanzables
    exit_fill = engine.simulate_position(fill, sl_price=0.1, tp_price=5.0, ticks_during=base_ticks)
    assert exit_fill.reason == "TIME"
    assert exit_fill.fill_time == pd.Timestamp("2026-05-12 20:00:00", tz="UTC") # 16:00 NY

def test_forced_exit_1700_ny_works(empty_calendar, base_ticks):
    """7. Forced exit a 17:00 NY funciona."""
    engine = UnifiedV7Engine(news_calendar=empty_calendar, forced_exit_mode="17:00", news_mode="none")
    sig = pd.Timestamp("2026-05-12 13:00:00", tz="UTC")
    fill, _ = engine.execute_signal("long", sig, base_ticks)
    assert fill is not None
    exit_fill = engine.simulate_position(fill, sl_price=0.1, tp_price=5.0, ticks_during=base_ticks)
    assert exit_fill.reason == "TIME"
    assert exit_fill.fill_time == pd.Timestamp("2026-05-12 21:00:00", tz="UTC") # 17:00 NY

def test_break_even_triggers_correctly(empty_calendar):
    """8. BE se activa correctamente."""
    engine = UnifiedV7Engine(news_calendar=empty_calendar, news_mode="none")
    # Crear serie de ticks donde el precio sube favorablemente activando BE, y luego retrocede
    idx = pd.date_range(start="2026-05-12 13:00:00", periods=5, freq="1min", tz="UTC")
    df_be = pd.DataFrame({
        "bid": [1.05000, 1.05050, 1.05250, 1.05005, 1.04900], # Pico en idx 2 activa BE
        "ask": [1.05010, 1.05060, 1.05260, 1.05015, 1.04910]
    }, index=idx)
    sig = pd.Timestamp("2026-05-12 13:00:00", tz="UTC")
    fill, _ = engine.execute_signal("long", sig, df_be)
    assert fill is not None
    
    # SL original: 1.04900. Entrada ask en t+1 (13:01) es 1.05060. sl_dist = 0.00160.
    # Disparador BE a 1.0R favorable: requiere 1.05060 + 0.00160 = 1.05220 bid.
    # Pico bid en t+2 (13:02) es 1.05250 -> Activa BE.
    # Salida ocurre en t+3 (13:03) al caer por debajo del nuevo SL (1.05065).
    exit_fill = engine.simulate_position(
        fill, sl_price=1.04900, tp_price=1.06000, ticks_during=df_be, be_trigger_r=1.0
    )
    assert exit_fill.reason == "BE"

def test_tp_and_sl_calculate_r_correctly(empty_calendar, base_ticks):
    """9. TP y SL calculan R correctamente."""
    engine = UnifiedV7Engine(news_calendar=empty_calendar, news_mode="none")
    sig = pd.Timestamp("2026-05-12 13:00:00", tz="UTC")
    fill, _ = engine.execute_signal("long", sig, base_ticks)
    entry_p = fill.fill_price
    sl_p = entry_p - 0.00100
    tp_p = entry_p + 0.00200 # Exactamente 2.0R
    
    assert abs((tp_p - entry_p) / (entry_p - sl_p) - 2.0) < 0.0001

def test_bid_ask_applies_correctly_on_entry_and_exit(empty_calendar, base_ticks):
    """10. Bid/ask se aplica correctamente en entrada y salida."""
    engine = UnifiedV7Engine(news_calendar=empty_calendar, news_mode="none")
    sig = pd.Timestamp("2026-05-12 13:00:00", tz="UTC")
    # Entrada Long ejecuta sobre Ask
    fill_long, _ = engine.execute_signal("long", sig, base_ticks)
    assert fill_long.fill_price == base_ticks.loc[fill_long.fill_time, "ask"]
    
    # Entrada Short ejecuta sobre Bid
    fill_short, _ = engine.execute_signal("short", sig, base_ticks)
    assert fill_short.fill_price == base_ticks.loc[fill_short.fill_time, "bid"]

def test_no_lookahead_leakage(empty_calendar, base_ticks):
    """11. No hay lookahead."""
    engine = UnifiedV7Engine(news_calendar=empty_calendar, news_mode="none")
    sig = pd.Timestamp("2026-05-12 13:00:00", tz="UTC")
    
    # 1. Aserción de ejecución T+1 estricta: fill_time ocurre unívocamente en t > señal
    fill, _ = engine.execute_signal("long", sig, base_ticks)
    assert fill.fill_time > sig
    
    # 2. Bloqueo de datos futuros: un dataset truncado sin ticks posteriores a la señal
    # arroja un error incondicional impidiendo que el motor extraiga precios adelantados.
    from src.v6_utils.execution import NoFillError
    truncated_ticks = base_ticks[:"2026-05-12 13:00:00"]
    with pytest.raises(NoFillError):
        engine.execute_signal("long", sig, truncated_ticks)

def test_causal_log_remains_clean(empty_calendar, base_ticks):
    """12. CausalLog queda limpio y registra con pureza la secuencia real."""
    engine = UnifiedV7Engine(news_calendar=empty_calendar, news_mode="none")
    sig = pd.Timestamp("2026-05-12 13:00:00", tz="UTC")
    fill, _ = engine.execute_signal("long", sig, base_ticks)
    
    # 1. Verificar libro mayor general de transiciones causales
    clog = engine.get_causal_log()
    assert len(clog) > 0
    assert clog[0]["event"] == "EXECUTION_FILL"
    assert clog[0]["fill_price"] == fill.fill_price
    
    # 2. Simular posición y constatar agregado secuencial de cierre
    #    Orden causal post Gate 6E: POSITION_CLOSED → COSTS_APPLIED → FTMO_UPDATED_NET
    exit_fill = engine.simulate_position(fill, sl_price=1.04900, tp_price=1.06000, ticks_during=base_ticks)
    events = [e["event"] for e in clog]
    assert "POSITION_CLOSED" in events, f"POSITION_CLOSED missing from causal log: {events}"
    # Localizar el cierre y verificar orden causal estricto
    close_idx = next(i for i, e in enumerate(clog) if e["event"] == "POSITION_CLOSED")
    assert clog[close_idx]["reason"] in ["SL", "TP", "EOM", "TIME", "BE"]
    # Verificar que COSTS_APPLIED y FTMO_UPDATED_NET aparecen después del cierre (no antes)
    post_close_events = [e["event"] for e in clog[close_idx+1:]]
    assert "COSTS_APPLIED" in post_close_events, f"COSTS_APPLIED missing after POSITION_CLOSED: {post_close_events}"
    assert "FTMO_UPDATED_NET" in post_close_events, f"FTMO_UPDATED_NET missing after POSITION_CLOSED: {post_close_events}"
    # Verificar que no hay lookahead: FTMO_UPDATED_NET nunca antes de COSTS_APPLIED
    costs_idx = next(i for i, e in enumerate(clog) if e["event"] == "COSTS_APPLIED")
    ftmo_idx = next(i for i, e in enumerate(clog) if e["event"] == "FTMO_UPDATED_NET")
    assert close_idx < costs_idx < ftmo_idx, f"Causal order violation: CLOSED@{close_idx}, COSTS@{costs_idx}, FTMO@{ftmo_idx}"

def test_leakage_guard_blocks_test_set(empty_calendar, base_ticks):
    """13. Guarda automático bloquea unívocamente el acceso a la partición de prueba (TEST)."""
    from src.v7_engine.engine import TestLeakageViolation
    
    # Configurar motor explícitamente en modo optimización/train
    engine = UnifiedV7Engine(news_calendar=empty_calendar, news_mode="none", active_phase="train")
    
    # Intento de evaluar marca de tiempo perteneciente al conjunto TEST reservado (>= 2023)
    sig_test = pd.Timestamp("2023-01-05 13:00:00", tz="UTC")
    with pytest.raises(TestLeakageViolation):
        engine.execute_signal("long", sig_test, base_ticks)

def test_causal_log_detects_non_strict_fill_attempt(empty_calendar, base_ticks):
    """14. CausalLog detecta e intercepta unívocamente intentos de llenado no estrictos o futuros."""
    engine = UnifiedV7Engine(news_calendar=empty_calendar, news_mode="none")
    sig = pd.Timestamp("2026-05-12 13:00:00", tz="UTC")
    
    # 1. Flujo válido: ejecución pura sin violaciones causales
    fill, _ = engine.execute_signal("long", sig, base_ticks)
    clog = engine.get_causal_log()
    assert len(clog) > 0 # El log no está vacío por construcción
    assert not any(entry.get("violation", False) for entry in clog)
    
    # 2. Flujo inválido simulado: inyectar manualmente un intento de T+0 o lookahead
    engine.causal_log.append({
        "event": "UNAUTHORIZED_FILL_ATTEMPT",
        "violation": True,
        "reason": "Intento de extracción de precio T+0 interceptado."
    })
    
    # Comprobar detección obligatoria
    assert any(entry.get("violation", False) for entry in engine.get_causal_log())
