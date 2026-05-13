import json
import pytest
from datetime import datetime, timedelta
from src.v7_engine.news_filter import (
    NewsEvent,
    NewsCalendar,
    is_blocked_by_news,
    NewsCalendarMissingError,
    parse_forex_factory_json
)

@pytest.fixture
def base_calendar():
    cal = NewsCalendar()
    # Evento de referencia: 2026-05-12 a las 12:30:00 UTC
    ev1 = NewsEvent("1", "US CPI", datetime(2026, 5, 12, 12, 30, 0), "USD", "High")
    cal.add_event(ev1)
    cal.add_covered_period(datetime(2026, 5, 10), datetime(2026, 5, 15))
    return cal

def test_post_news_buffer_5min_blocks_correctly(base_calendar):
    """
    Verifica: test_post_news_buffer_5min_blocks_correctly (Requerido en ANCHOR_CONFIG.md).
    Bloquea estrictamente en [t_news, t_news + 5 min).
    """
    ts_news = datetime(2026, 5, 12, 12, 30, 0)
    
    # En el instante exacto de la noticia
    blocked, ev = is_blocked_by_news(ts_news, base_calendar)
    assert blocked is True
    assert ev.event_id == "1"
    
    # 4 minutos y 59 segundos post-noticia
    blocked, _ = is_blocked_by_news(ts_news + timedelta(minutes=4, seconds=59), base_calendar)
    assert blocked is True

def test_no_pre_news_block(base_calendar):
    """
    Verifica: test_no_pre_news_block (Requerido en ANCHOR_CONFIG.md).
    Garantiza que un segundo antes de la noticia el flujo está libre.
    """
    ts_news = datetime(2026, 5, 12, 12, 30, 0)
    
    blocked, ev = is_blocked_by_news(ts_news - timedelta(seconds=1), base_calendar)
    assert blocked is False
    assert ev is None

def test_buffer_extended_by_subsequent_news():
    """
    Verifica: test_buffer_extended_by_subsequent_news (Requerido en ANCHOR_CONFIG.md).
    Múltiples noticias cercanas solapan y extienden el contador de forma determinista.
    """
    cal = NewsCalendar()
    cal.add_covered_period(datetime(2026, 5, 1), datetime(2026, 5, 30))
    
    ev1 = NewsEvent("1", "News A", datetime(2026, 5, 12, 12, 30, 0), "USD", "High")
    ev2 = NewsEvent("2", "News B", datetime(2026, 5, 12, 12, 32, 0), "USD", "High")
    cal.add_event(ev1)
    cal.add_event(ev2)
    
    # A las 12:36:00, ev1 expiró (12:30 + 5m = 12:35), pero ev2 sigue activo (12:32 + 5m = 12:37)
    blocked, ev = is_blocked_by_news(datetime(2026, 5, 12, 12, 36, 0), cal)
    assert blocked is True
    assert ev.event_id == "2"

def test_buffer_clears_after_threshold(base_calendar):
    """
    Verifica: test_buffer_clears_after_threshold (Requerido en ANCHOR_CONFIG.md).
    A los 5 minutos exactos la ventana se libera.
    """
    ts_clear = datetime(2026, 5, 12, 12, 35, 0)
    blocked, _ = is_blocked_by_news(ts_clear, base_calendar)
    assert blocked is False

def test_blocks_during_window(base_calendar):
    """
    Verifica: Propiedad de bloqueo activa (Test base V7).
    """
    blocked, _ = is_blocked_by_news(datetime(2026, 5, 12, 12, 31, 0), base_calendar)
    assert blocked is True

def test_allows_outside_window(base_calendar):
    """
    Verifica: El mercado fluye inalterado fuera de las ventanas de impacto.
    """
    blocked, _ = is_blocked_by_news(datetime(2026, 5, 12, 14, 0, 0), base_calendar)
    assert blocked is False

def test_partial_overlap_entry_inside(base_calendar):
    """
    Verifica: Comportamiento ante solapamiento límite.
    """
    # Límite superior del buffer
    blocked, _ = is_blocked_by_news(datetime(2026, 5, 12, 12, 34, 59), base_calendar)
    assert blocked is True

def test_dst_transition_correct():
    """
    Verifica: Robustez ante timestamps UTC continuos sin saltos ambiguos de horario de verano.
    """
    cal = NewsCalendar()
    # Fecha típica de cambio DST (Marzo)
    ts_dst = datetime(2026, 3, 8, 7, 0, 0)
    cal.add_covered_period(ts_dst - timedelta(days=1), ts_dst + timedelta(days=1))
    cal.add_event(NewsEvent("dst", "Event", ts_dst, "USD", "High"))
    
    blocked, _ = is_blocked_by_news(ts_dst + timedelta(minutes=2), cal)
    assert blocked is True

def test_unknown_period_raises():
    """
    Verifica: test_unknown_period_raises.
    Falta de disponibilidad lanza NewsCalendarMissingError en lugar de asumir 'sin noticias'.
    """
    cal = NewsCalendar()
    # Calendario vacío sin periodos cubiertos
    with pytest.raises(NewsCalendarMissingError) as exc_info:
        is_blocked_by_news(datetime(2012, 1, 1, 0, 0), cal)
    assert "falta de disponibilidad" in str(exc_info.value).lower()

def test_currency_filter_ignores_jpy():
    """
    Verifica: test_currency_filter_ignores_jpy.
    Noticias de divisas ajenas no alteran el trading de EURUSD.
    """
    cal = NewsCalendar()
    cal.add_covered_period(datetime(2026, 1, 1), datetime(2026, 1, 10))
    cal.add_event(NewsEvent("jpy", "BOJ Rate", datetime(2026, 1, 5, 4, 0), "JPY", "High"))
    
    blocked, _ = is_blocked_by_news(datetime(2026, 1, 5, 4, 1), cal)
    assert blocked is False

def test_impact_filter_ignores_low():
    """
    Verifica: test_impact_filter_ignores_low.
    Eventos de bajo impacto no activan la barrera.
    """
    cal = NewsCalendar()
    cal.add_covered_period(datetime(2026, 1, 1), datetime(2026, 1, 10))
    cal.add_event(NewsEvent("low", "Retail Sales", datetime(2026, 1, 5, 12, 30), "USD", "Low"))
    
    blocked, _ = is_blocked_by_news(datetime(2026, 1, 5, 12, 31), cal)
    assert blocked is False

def test_pre_post_minutes_independent(base_calendar):
    """
    Verifica: test_pre_post_minutes_independent.
    Parámetros pre y post configurables de forma completamente ortogonal.
    """
    ts_news = datetime(2026, 5, 12, 12, 30, 0)
    
    # Activar pre_block 10 min, post_block 0 min
    blocked_pre, _ = is_blocked_by_news(ts_news - timedelta(minutes=5), base_calendar, pre_minutes=10, post_minutes=0)
    assert blocked_pre is True
    
    # Post block debe ser false a los 10 segundos si post_minutes=0
    blocked_post, _ = is_blocked_by_news(ts_news + timedelta(seconds=10), base_calendar, pre_minutes=10, post_minutes=0)
    assert blocked_post is False

def test_parse_forex_factory_json_helper():
    """
    Verifica la extracción limpia y estructurada desde la fuente de ForexFactory.
    """
    raw_json = json.dumps([
        {"id": 101, "title": "FOMC", "country": "USD", "impact": "High", "date": "2026-06-10T18:00:00Z"}
    ])
    cal = parse_forex_factory_json(raw_json)
    assert len(cal.events) == 1
    assert cal.events[0].title == "FOMC"
    assert cal.is_covered(datetime(2026, 6, 10, 18, 0, 0)) is True
