import pytest
import pandas as pd
from datetime import datetime
from src.v6_utils.execution import next_bar_execute, simulate_exit, simulate_exit_with_be, NoFillError, FillResult
from src.v7_engine.engine import UnifiedV7Engine
from src.v7_engine.news_filter import NewsCalendar

@pytest.fixture
def base_causal_ticks():
    # Ticks precisos y ordenados para simulación rigurosa
    ts_idx = pd.to_datetime([
        "2026-05-12 13:00:00", # Coincide con la vela de señal
        "2026-05-12 13:00:01", # Primer tick post-señal
        "2026-05-12 13:00:02",
        "2026-05-12 13:05:00",
        "2026-05-12 15:59:59",
        "2026-05-12 16:00:00", # Instante de forced exit NY
        "2026-05-12 16:00:01"
    ], utc=True)
    
    return pd.DataFrame({
        "bid": [1.05000, 1.05010, 1.05020, 1.05500, 1.05400, 1.05390, 1.05380],
        "ask": [1.05010, 1.05020, 1.05030, 1.05510, 1.05410, 1.05400, 1.05390]
    }, index=ts_idx)

def test_execution_rejects_when_only_past_or_equal_ticks_exist(base_causal_ticks):
    """6.1 Impide fill si solo existen ticks en o antes del timestamp de la señal."""
    sig_time = pd.Timestamp("2026-05-12 13:00:00", tz="UTC")
    truncated = base_causal_ticks[:"2026-05-12 13:00:00"]
    with pytest.raises(NoFillError):
        next_bar_execute("long", sig_time, truncated)

def test_execution_uses_first_tick_strictly_after_signal_time(base_causal_ticks):
    """6.2 Garantiza que fill_time > signal_time usando estrictamente el primer tick posterior."""
    sig_time = pd.Timestamp("2026-05-12 13:00:00", tz="UTC")
    fill = next_bar_execute("long", sig_time, base_causal_ticks)
    assert fill.fill_time == pd.Timestamp("2026-05-12 13:00:01", tz="UTC")
    assert fill.fill_price == 1.05020 # Ask de 13:00:01

def test_buy_market_entry_uses_ask_not_bid_or_mid(base_causal_ticks):
    """6.3 Entrada compradora a mercado toma unívocamente el lado Ask del libro."""
    sig_time = pd.Timestamp("2026-05-12 13:00:00", tz="UTC")
    fill = next_bar_execute("long", sig_time, base_causal_ticks)
    ask_expected = base_causal_ticks.loc[fill.fill_time, "ask"]
    bid_avoided = base_causal_ticks.loc[fill.fill_time, "bid"]
    assert fill.fill_price == ask_expected
    assert fill.fill_price != bid_avoided

def test_sell_market_entry_uses_bid_not_ask_or_mid(base_causal_ticks):
    """6.4 Entrada vendedora a mercado toma unívocamente el lado Bid del libro."""
    sig_time = pd.Timestamp("2026-05-12 13:00:00", tz="UTC")
    fill = next_bar_execute("short", sig_time, base_causal_ticks)
    bid_expected = base_causal_ticks.loc[fill.fill_time, "bid"]
    ask_avoided = base_causal_ticks.loc[fill.fill_time, "ask"]
    assert fill.fill_price == bid_expected
    assert fill.fill_price != ask_avoided

def test_exit_resolution_follows_tick_order_not_bar_high_low():
    """6.5 Resolución de salida sigue el orden real de ticks, previniendo llenados fantasma en barras."""
    ts_idx = pd.to_datetime(["2026-05-12 13:01:00", "2026-05-12 13:02:00", "2026-05-12 13:03:00"], utc=True)
    
    # Escenario A: SL se toca primero temporalmente
    ticks_sl_first = pd.DataFrame({
        "bid": [1.04900, 1.06000, 1.05500], # Toca SL (1.04950) en el primer tick
        "ask": [1.04910, 1.06010, 1.05510]
    }, index=ts_idx)
    
    fill_time = pd.Timestamp("2026-05-12 13:00:00", tz="UTC")
    res_sl = simulate_exit("long", 1.05000, sl_price=1.04950, tp_price=1.05950, ticks_during=ticks_sl_first, fill_time=fill_time)
    assert res_sl.reason == "SL"
    
    # Escenario B: TP se toca primero temporalmente
    ticks_tp_first = pd.DataFrame({
        "bid": [1.06000, 1.04900, 1.05500], # Toca TP (1.05950) en el primer tick
        "ask": [1.06010, 1.04910, 1.05510]
    }, index=ts_idx)
    res_tp = simulate_exit("long", 1.05000, sl_price=1.04950, tp_price=1.05950, ticks_during=ticks_tp_first, fill_time=fill_time)
    assert res_tp.reason == "TP"

def test_break_even_activates_only_after_real_mfe_tick():
    """6.6 Break-even se activa solo tras un tick de MFE favorable, sin asunciones adelantadas."""
    ts_idx = pd.to_datetime(["2026-05-12 13:01:00", "2026-05-12 13:02:00", "2026-05-12 13:03:00"], utc=True)
    ticks_be = pd.DataFrame({
        "bid": [1.05100, 1.05300, 1.04990], # Tick 1 no llega al trigger (1.05200). Tick 2 arma BE. Tick 3 sale en BE-SL.
        "ask": [1.05110, 1.05310, 1.05000]
    }, index=ts_idx)
    
    fill_time = pd.Timestamp("2026-05-12 13:00:00", tz="UTC")
    res = simulate_exit_with_be(
        "long", entry_price=1.05000, sl_price=1.04800,
        be_trigger=1.05200, be_new_sl=1.05005, tp_price=1.06000,
        ticks_during=ticks_be, fill_time=fill_time
    )
    assert res.reason == "BE-SL"
    assert res.fill_price == 1.05005

def test_forced_exit_uses_configured_ny_time_without_lookahead(base_causal_ticks):
    """6.7 Cierre forzado se dispara en o tras la estampa configurada sin espiar ticks previos."""
    fill_time = pd.Timestamp("2026-05-12 13:00:01", tz="UTC")
    exit_target = pd.Timestamp("2026-05-12 16:00:00", tz="UTC")
    
    res = simulate_exit("long", 1.05020, sl_price=1.00000, tp_price=2.00000, ticks_during=base_causal_ticks, fill_time=fill_time, time_exit=exit_target)
    assert res.reason == "TIME"
    assert res.fill_time >= exit_target
    # Asegurar que no cerró prematuramente en 15:59:59
    assert res.fill_time != pd.Timestamp("2026-05-12 15:59:59", tz="UTC")

def test_no_bar_high_low_shortcut_for_exit():
    """6.8 Demuestra que no existe atajo de resolución optimista por extremos agregados de barra."""
    ts_idx = pd.to_datetime(["2026-05-12 13:01:00", "2026-05-12 13:02:00"], utc=True)
    # Ticks reales que simulan el interior de una vela de alta volatilidad
    ticks_real = pd.DataFrame({
        "bid": [1.04500, 1.06500], # Primero cruza el SL inferior de forma destructiva
        "ask": [1.04510, 1.06510]
    }, index=ts_idx)
    
    fill_time = pd.Timestamp("2026-05-12 13:00:00", tz="UTC")
    res = simulate_exit("long", 1.05000, sl_price=1.04800, tp_price=1.06000, ticks_during=ticks_real, fill_time=fill_time)
    # Si usara un atajo optimista de high/low simultáneo, podría declarar TP erróneamente.
    # Al seguir el flujo estricto T+1 tick a tick, se garantiza el dictamen verídico de SL.
    assert res.reason == "SL"

# =====================================================================
# SECCIÓN DE REFUERZO DE ESTRÉS CAUSAL (CORRECCIÓN D)
# =====================================================================

def test_tick_order_with_millisecond_burst_sl_first():
    """D1. Ráfaga sub-milisegundo resuelve verídicamente SL antes que TP."""
    ts_idx = pd.to_datetime([
        "2026-05-12 13:00:00.001",
        "2026-05-12 13:00:00.002",
        "2026-05-12 13:00:00.003"
    ], utc=True)
    ticks_burst = pd.DataFrame({
        "bid": [1.05000, 1.04890, 1.05500], # Toca SL (1.04900) en el ms 002
        "ask": [1.05010, 1.04900, 1.05510]
    }, index=ts_idx)
    fill_time = pd.Timestamp("2026-05-12 13:00:00.000", tz="UTC")
    res = simulate_exit("long", 1.05000, sl_price=1.04900, tp_price=1.05400, ticks_during=ticks_burst, fill_time=fill_time)
    assert res.reason == "SL"
    assert res.fill_time == pd.Timestamp("2026-05-12 13:00:00.002", tz="UTC")

def test_tick_order_with_millisecond_burst_tp_first():
    """D2. Ráfaga sub-milisegundo resuelve verídicamente TP antes que SL."""
    ts_idx = pd.to_datetime([
        "2026-05-12 13:00:00.001",
        "2026-05-12 13:00:00.002",
        "2026-05-12 13:00:00.003"
    ], utc=True)
    ticks_burst = pd.DataFrame({
        "bid": [1.05000, 1.05400, 1.04890], # Toca TP (1.05400) en el ms 002
        "ask": [1.05010, 1.05410, 1.04900]
    }, index=ts_idx)
    fill_time = pd.Timestamp("2026-05-12 13:00:00.000", tz="UTC")
    res = simulate_exit("long", 1.05000, sl_price=1.04900, tp_price=1.05400, ticks_during=ticks_burst, fill_time=fill_time)
    assert res.reason == "TP"

def test_same_second_microsecond_order_is_preserved():
    """D3. Preservación estricta de orden en colisiones de un mismo segundo variando microsegundos."""
    ts_idx = pd.to_datetime([
        "2026-05-12 13:00:01.000010",
        "2026-05-12 13:00:01.000020"
    ], utc=True)
    ticks_us = pd.DataFrame({
        "bid": [1.04900, 1.05500],
        "ask": [1.04910, 1.05510]
    }, index=ts_idx)
    fill_time = pd.Timestamp("2026-05-12 13:00:01.000000", tz="UTC")
    res = simulate_exit("long", 1.05000, sl_price=1.04950, tp_price=1.05450, ticks_during=ticks_us, fill_time=fill_time)
    assert res.reason == "SL"

def test_variable_spread_does_not_use_mid_price():
    """D4. Horquilla de spread extremadamente asimétrica valida pureza de fill excluyendo mid-price."""
    ts_idx = pd.to_datetime(["2026-05-12 13:00:01"], utc=True)
    # Spread gigante de 20 pips donde el mid price dispararía TP erróneamente
    ticks_wide = pd.DataFrame({
        "bid": [1.04000],
        "ask": [1.06000] # Mid price sería 1.05000
    }, index=ts_idx)
    fill_time = pd.Timestamp("2026-05-12 13:00:00", tz="UTC")
    res = simulate_exit("long", 1.04500, sl_price=1.04100, tp_price=1.04900, ticks_during=ticks_wide, fill_time=fill_time)
    # Posición larga sale por SL porque el bid (1.04000) cruza 1.04100
    assert res.reason == "SL"
    assert res.fill_price == 1.04100

def test_be_not_triggered_by_bar_high_without_tick():
    """D5. El umbral de Break-Even no se asume por máximos de vela sin un tick causal explícito."""
    ts_idx = pd.to_datetime(["2026-05-12 13:00:01", "2026-05-12 13:00:02"], utc=True)
    ticks_be = pd.DataFrame({
        "bid": [1.05190, 1.04900], # Se queda a 1 pipette del trigger (1.05200) y cae
        "ask": [1.05200, 1.04910]
    }, index=ts_idx)
    fill_time = pd.Timestamp("2026-05-12 13:00:00", tz="UTC")
    res = simulate_exit_with_be(
        "long", entry_price=1.05000, sl_price=1.04950,
        be_trigger=1.05200, be_new_sl=1.05005, tp_price=1.06000,
        ticks_during=ticks_be, fill_time=fill_time
    )
    assert res.reason == "SL" # Salió en SL original porque BE jamás armó

def test_forced_exit_uses_first_tick_after_boundary():
    """D6. Cierre forzado captura con precisión sub-segundo el primer tick post-límite."""
    ts_idx = pd.to_datetime([
        "2026-05-12 15:59:59.999",
        "2026-05-12 16:00:00.001"
    ], utc=True)
    ticks_bnd = pd.DataFrame({"bid": [1.05000, 1.04990], "ask": [1.05010, 1.05000]}, index=ts_idx)
    fill_time = pd.Timestamp("2026-05-12 13:00:00", tz="UTC")
    res = simulate_exit("long", 1.05000, sl_price=1.00000, tp_price=2.00000, ticks_during=ticks_bnd, fill_time=fill_time, time_exit=pd.Timestamp("2026-05-12 16:00:00", tz="UTC"))
    assert res.reason == "TIME"
    assert res.fill_time == pd.Timestamp("2026-05-12 16:00:00.001", tz="UTC")

def test_sell_exit_uses_correct_bid_ask_side():
    """D7. Operación corta resuelve sus paradas empleando unívocamente la columna Ask."""
    ts_idx = pd.to_datetime(["2026-05-12 13:00:01"], utc=True)
    ticks_side = pd.DataFrame({"bid": [1.04900], "ask": [1.05100]}, index=ts_idx)
    fill_time = pd.Timestamp("2026-05-12 13:00:00", tz="UTC")
    res = simulate_exit("short", 1.05000, sl_price=1.05050, tp_price=1.04000, ticks_during=ticks_side, fill_time=fill_time)
    assert res.reason == "SL" # Ask (1.05100) cruzó SL (1.05050)

def test_buy_exit_uses_correct_bid_ask_side():
    """D8. Operación larga resuelve sus paradas empleando unívocamente la columna Bid."""
    ts_idx = pd.to_datetime(["2026-05-12 13:00:01"], utc=True)
    ticks_side = pd.DataFrame({"bid": [1.04900], "ask": [1.05100]}, index=ts_idx)
    fill_time = pd.Timestamp("2026-05-12 13:00:00", tz="UTC")
    res = simulate_exit("long", 1.05000, sl_price=1.04950, tp_price=1.06000, ticks_during=ticks_side, fill_time=fill_time)
    assert res.reason == "SL" # Bid (1.04900) cruzó SL (1.04950)
