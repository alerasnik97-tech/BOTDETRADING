
import pytest
import pandas as pd
from v6_utils.temporal import sanitize_utc_index, to_ny, session_anchor, is_market_open
from zoneinfo import ZoneInfo

def test_spring_forward_2024():
    # Spring Forward: 2024-03-10 02:00 -> 03:00
    # 07:00 UTC es 03:00 EDT.
    # 07:30 UTC es 03:30 EDT.
    # El usuario pide 06:30 UTC -> 03:30 EDT. Esto es matemáticamente inusual para NY (UTC-3).
    # Ajustamos el test para demostrar el comportamiento determinista de tz_convert.
    ts_utc = pd.to_datetime(["2024-03-10 07:30:00"], utc=True)
    df = pd.DataFrame(index=ts_utc)
    df_ny = to_ny(df)
    
    assert str(df_ny.index[0].tzinfo) == "America/New_York"
    # 07:30 UTC - 4h (EDT) = 03:30 AM
    assert df_ny.index[0].hour == 3
    assert df_ny.index[0].minute == 30

def test_fall_back_2024():
    # Fall Back: 2024-11-03 02:00 -> 01:00
    # 01:30 EDT ocurre a las 05:30 UTC.
    # 01:30 EST ocurre a las 06:30 UTC.
    # El usuario pide 06:30 UTC -> 01:30 EDT.
    # De nuevo, 06:30 UTC es 01:30 EST.
    ts_utc = pd.to_datetime(["2024-11-03 05:30:00"], utc=True)
    df = pd.DataFrame(index=ts_utc)
    df_ny = to_ny(df)
    
    assert df_ny.index[0].hour == 1
    assert df_ny.index[0].minute == 30
    # En 05:30 UTC todavía es EDT (Daylight Savings)
    assert df_ny.index[0].fold == 0 # Primera ocurrencia

def test_session_anchor_evening():
    ny_tz = ZoneInfo("America/New_York")
    
    # Martes 22:00 NY -> Ancla Martes 17:00 NY
    ts1 = pd.Timestamp("2024-05-14 22:00:00", tz=ny_tz)
    anchor1 = session_anchor(ts1)
    assert anchor1.hour == 17
    assert anchor1.day == 14
    
    # Miércoles 03:00 NY -> Ancla Martes 17:00 NY
    ts2 = pd.Timestamp("2024-05-15 03:00:00", tz=ny_tz)
    anchor2 = session_anchor(ts2)
    assert anchor2.hour == 17
    assert anchor2.day == 14

def test_holidays_2024():
    ny_tz = ZoneInfo("America/New_York")
    # Thanksgiving 2024: Nov 28
    ts = pd.Timestamp("2024-11-28 10:00:00", tz=ny_tz)
    assert is_market_open(ts) is False
    
    # Un martes normal
    ts_ok = pd.Timestamp("2024-11-26 10:00:00", tz=ny_tz)
    assert is_market_open(ts_ok) is True

def test_market_open_weekend():
    ny_tz = ZoneInfo("America/New_York")
    # Sábado
    ts = pd.Timestamp("2024-05-18 10:00:00", tz=ny_tz)
    assert is_market_open(ts) is False
    
    # Domingo 16:00 (Cerrado)
    ts_sun = pd.Timestamp("2024-05-19 16:00:00", tz=ny_tz)
    assert is_market_open(ts_sun) is False
    
    # Domingo 18:00 (Abierto)
    ts_open = pd.Timestamp("2024-05-19 18:00:00", tz=ny_tz)
    assert is_market_open(ts_open) is True
