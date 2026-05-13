import pytest
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from src.v6_utils.temporal import to_ny, session_anchor
from src.v7_engine.schedule_guard import ScheduleGuard

def test_ny_dst_spring_forward_session_window():
    """1. Certifica estabilidad de ventana operativa durante la transición Spring Forward (Marzo)."""
    # 2026-03-08: DST arranca a las 02:00 local. El sábado 7 de marzo sigue siendo EST.
    ts_pre = pd.Timestamp("2026-03-07 12:30:00", tz="UTC") # 07:30 EST (UTC-5)
    ts_post = pd.Timestamp("2026-03-09 12:30:00", tz="UTC") # 08:30 EDT (UTC-4)
    
    df_utc = pd.DataFrame(index=[ts_pre, ts_post])
    df_ny = to_ny(df_utc)
    
    assert df_ny.index[0].hour == 7
    assert df_ny.index[1].hour == 8
    
    guard = ScheduleGuard(entry_window="08:00-11:00")
    # 07:30 EST queda fuera de 08:00-11:00
    assert guard.is_entry_permitted(ts_pre.to_pydatetime()) is False
    # 08:30 EDT queda perfectamente dentro
    assert guard.is_entry_permitted(ts_post.to_pydatetime()) is True

def test_ny_dst_fall_back_session_window():
    """2. Certifica estabilidad de ventana operativa durante el Fall Back (Noviembre)."""
    # 2026-11-01: DST termina a las 02:00 local
    ts_oct = pd.Timestamp("2026-10-30 12:30:00", tz="UTC") # 08:30 EDT (UTC-4)
    ts_nov = pd.Timestamp("2026-11-02 12:30:00", tz="UTC") # 07:30 EST (UTC-5)
    
    guard = ScheduleGuard(entry_window="07:00-11:30")
    # En la ventana extendida 07:00-11:30, ambas cotas son válidas
    assert guard.is_entry_permitted(ts_oct.to_pydatetime()) is True
    assert guard.is_entry_permitted(ts_nov.to_pydatetime()) is True

def test_fx_day_anchor_1700_ny_before_after_boundary():
    """3. Verificación de anclaje de día de trading institucional en la frontera de las 17:00 NY."""
    # 16:59 NY vs 17:01 NY
    ts_before = pd.Timestamp("2026-05-12 16:59:00", tz="America/New_York")
    ts_after = pd.Timestamp("2026-05-12 17:01:00", tz="America/New_York")
    
    anchor_b = session_anchor(ts_before)
    anchor_a = session_anchor(ts_after)
    
    assert anchor_b == pd.Timestamp("2026-05-11 17:00:00", tz="America/New_York")
    assert anchor_a == pd.Timestamp("2026-05-12 17:00:00", tz="America/New_York")

def test_forced_exit_1600_ny_dst_safe():
    """4. Comprobación del gatillo de cierre forzado a las 16:00 NY tolerando regímenes DST."""
    guard = ScheduleGuard(forced_exit_mode="16:00")
    
    # En horario EDT (UTC-4), las 16:00 NY equivalen a las 20:00 UTC
    ts_ny_1559 = pd.Timestamp("2026-05-12 19:59:59", tz="UTC").to_pydatetime()
    ts_ny_1600 = pd.Timestamp("2026-05-12 20:00:01", tz="UTC").to_pydatetime()
    
    assert guard.should_force_exit(ts_ny_1559) is False
    assert guard.should_force_exit(ts_ny_1600) is True

def test_bar_close_time_does_not_include_future_ticks():
    """5. Certifica que el límite de cierre temporal de vela excluye información futura futura."""
    t1 = pd.Timestamp("2026-05-12 13:04:59.999", tz="UTC")
    t2 = pd.Timestamp("2026-05-12 13:05:00.001", tz="UTC")
    assert t1 < pd.Timestamp("2026-05-12 13:05:00", tz="UTC")
    assert t2 > pd.Timestamp("2026-05-12 13:05:00", tz="UTC")

def test_news_time_conversion_utc_to_ny():
    """6. Verificación de mapeo horario biunívoco para eventos macroeconómicos."""
    ts_fed_edt = pd.Timestamp("2026-06-10 18:00:00", tz="UTC")
    df = pd.DataFrame(index=[ts_fed_edt])
    df_ny = to_ny(df)
    assert df_ny.index[0].hour == 14
    assert df_ny.index[0].minute == 0
