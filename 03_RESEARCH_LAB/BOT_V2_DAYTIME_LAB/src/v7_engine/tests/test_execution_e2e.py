import pytest
import pandas as pd
from src.v6_utils.execution import next_bar_execute, simulate_exit, simulate_exit_with_be, NoFillError

@pytest.fixture
def e2e_ticks():
    ts = pd.to_datetime([
        "2026-05-12 13:00:00.000",
        "2026-05-12 13:00:00.001",
        "2026-05-12 13:00:00.002",
        "2026-05-12 13:00:01.000",
        "2026-05-12 15:59:59.999",
        "2026-05-12 16:00:00.001"
    ], utc=True)
    return pd.DataFrame({
        "bid": [1.05000, 1.05010, 1.05020, 1.05500, 1.05400, 1.05390],
        "ask": [1.05010, 1.05020, 1.05030, 1.05510, 1.05410, 1.05400]
    }, index=ts)

def test_e2e_1_only_past_ticks_no_fill(e2e_ticks):
    """1. Impide fill si solo se le suministran ticks estrictamente pasados."""
    sig = pd.Timestamp("2026-05-12 13:00:02", tz="UTC")
    truncated = e2e_ticks[:"2026-05-12 13:00:01"]
    with pytest.raises(NoFillError):
        next_bar_execute("long", sig, truncated)

def test_e2e_2_tick_equal_signal_time_no_fill(e2e_ticks):
    """2. Impide fill en el mismo instante T+0 coincidente con la señal."""
    sig = pd.Timestamp("2026-05-12 13:00:00", tz="UTC")
    truncated = e2e_ticks[e2e_ticks.index <= sig]
    with pytest.raises(NoFillError):
        next_bar_execute("long", sig, truncated)

def test_e2e_3_first_valid_tick_after_signal_fill(e2e_ticks):
    """3. Llenado unívoco consumiendo el primer tick T+1 posterior."""
    sig = pd.Timestamp("2026-05-12 13:00:00", tz="UTC")
    fill = next_bar_execute("long", sig, e2e_ticks)
    assert fill.fill_time == pd.Timestamp("2026-05-12 13:00:00.001", tz="UTC")

def test_e2e_4_buy_uses_ask(e2e_ticks):
    """4. Compras atacan invariablemente el lado Ask del libro."""
    sig = pd.Timestamp("2026-05-12 13:00:00", tz="UTC")
    fill = next_bar_execute("long", sig, e2e_ticks)
    assert fill.fill_price == 1.05020 # Ask en 001 ms

def test_e2e_5_sell_uses_bid(e2e_ticks):
    """5. Ventas atacan invariablemente el lado Bid del libro."""
    sig = pd.Timestamp("2026-05-12 13:00:00", tz="UTC")
    fill = next_bar_execute("short", sig, e2e_ticks)
    assert fill.fill_price == 1.05010 # Bid en 001 ms

def test_e2e_6_spread_variable_does_not_use_mid():
    """6. Exclusión formal de precios medios ante horquillas asimétricas."""
    ts = pd.to_datetime(["2026-05-12 13:00:01"], utc=True)
    df = pd.DataFrame({"bid": [1.04000], "ask": [1.06000]}, index=ts)
    fill = next_bar_execute("long", pd.Timestamp("2026-05-12 13:00:00", tz="UTC"), df)
    assert fill.fill_price == 1.06000
    assert fill.fill_price != 1.05000

def test_e2e_7_sl_first_then_tp():
    """7. Secuencialidad verídica capturando SL con anterioridad al TP."""
    ts = pd.to_datetime(["2026-05-12 13:00:01", "2026-05-12 13:00:02"], utc=True)
    df = pd.DataFrame({"bid": [1.04900, 1.06000], "ask": [1.04910, 1.06010]}, index=ts)
    res = simulate_exit("long", 1.05000, 1.04950, 1.05950, df, pd.Timestamp("2026-05-12 13:00:00", tz="UTC"))
    assert res.reason == "SL"

def test_e2e_8_tp_first_then_sl():
    """8. Secuencialidad verídica capturando TP con anterioridad al SL."""
    ts = pd.to_datetime(["2026-05-12 13:00:01", "2026-05-12 13:00:02"], utc=True)
    df = pd.DataFrame({"bid": [1.06000, 1.04900], "ask": [1.06010, 1.04910]}, index=ts)
    res = simulate_exit("long", 1.05000, 1.04950, 1.05950, df, pd.Timestamp("2026-05-12 13:00:00", tz="UTC"))
    assert res.reason == "TP"

def test_e2e_9_same_second_microsecond_preserved():
    """9. Estabilidad cronológica ante colisiones variando microsegundos."""
    ts = pd.to_datetime(["2026-05-12 13:00:01.000005", "2026-05-12 13:00:01.000010"], utc=True)
    df = pd.DataFrame({"bid": [1.04900, 1.05500], "ask": [1.04910, 1.05510]}, index=ts)
    res = simulate_exit("long", 1.05000, 1.04950, 1.05450, df, pd.Timestamp("2026-05-12 13:00:01", tz="UTC"))
    assert res.reason == "SL"

def test_e2e_10_millisecond_burst_preserved():
    """10. Ráfagas densas de milisegundos retienen el orden subyacente sin agrupamiento."""
    ts = pd.to_datetime(["2026-05-12 13:00:00.001", "2026-05-12 13:00:00.002"], utc=True)
    df = pd.DataFrame({"bid": [1.05500, 1.04500], "ask": [1.05510, 1.04510]}, index=ts)
    res = simulate_exit("long", 1.05000, 1.04800, 1.05400, df, pd.Timestamp("2026-05-12 13:00:00", tz="UTC"))
    assert res.reason == "TP"

def test_e2e_11_be_not_active_before_mfe():
    """11. Break-even inactivo si no se alcanza la cota estricta de ganancia máxima."""
    ts = pd.to_datetime(["2026-05-12 13:00:01"], utc=True)
    df = pd.DataFrame({"bid": [1.05190], "ask": [1.05200]}, index=ts) # Queda corto de 1.05200 en Bid
    res = simulate_exit_with_be("long", 1.05000, 1.04900, 1.05200, 1.05005, 1.06000, df, pd.Timestamp("2026-05-12 13:00:00", tz="UTC"))
    # No sale en BE porque no activó. Al no cruzar tampoco SL ni TP, retorna EOM por defecto.
    assert res.reason == "EOM"

def test_e2e_12_be_active_after_real_tick_only():
    """12. Activación confirmada de Break-even tras consumir un tick causal de MFE."""
    ts = pd.to_datetime(["2026-05-12 13:00:01", "2026-05-12 13:00:02"], utc=True)
    df = pd.DataFrame({"bid": [1.05250, 1.04990], "ask": [1.05260, 1.05000]}, index=ts)
    res = simulate_exit_with_be("long", 1.05000, 1.04800, 1.05200, 1.05005, 1.06000, df, pd.Timestamp("2026-05-12 13:00:00", tz="UTC"))
    assert res.reason == "BE-SL"
    assert res.fill_price == 1.05005

def test_e2e_13_forced_exit_first_valid_tick():
    """13. Cierre forzado captura el instante físico subsecuente en el libro."""
    ts = pd.to_datetime(["2026-05-12 15:59:59.999", "2026-05-12 16:00:00.001"], utc=True)
    df = pd.DataFrame({"bid": [1.05000, 1.04950], "ask": [1.05010, 1.04960]}, index=ts)
    res = simulate_exit("long", 1.05000, 1.00000, 2.00000, df, pd.Timestamp("2026-05-12 13:00:00", tz="UTC"), time_exit=pd.Timestamp("2026-05-12 16:00:00", tz="UTC"))
    assert res.reason == "TIME"
    assert res.fill_time == pd.Timestamp("2026-05-12 16:00:00.001", tz="UTC")

def test_e2e_14_missing_ticks_after_signal_controlled_no_fill():
    """14. Ausencia total de cotizaciones post-señal aborta de forma controlada."""
    ts = pd.to_datetime(["2026-05-12 13:00:00"], utc=True)
    df = pd.DataFrame({"bid": [1.05], "ask": [1.051]}, index=ts)
    with pytest.raises(NoFillError):
        next_bar_execute("long", pd.Timestamp("2026-05-12 13:00:00", tz="UTC"), df)

def test_e2e_15_no_bar_high_low_shortcut():
    """15. Certifica ausencia de atajos optimistas por extremos fractales agregados."""
    ts = pd.to_datetime(["2026-05-12 13:01:00", "2026-05-12 13:02:00"], utc=True)
    df = pd.DataFrame({"bid": [1.04500, 1.06500], "ask": [1.04510, 1.06510]}, index=ts)
    res = simulate_exit("long", 1.05000, 1.04800, 1.06000, df, pd.Timestamp("2026-05-12 13:00:00", tz="UTC"))
    assert res.reason == "SL"
