from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

class NewsCalendarMissingError(Exception):
    """Lanzada cuando se consulta un timestamp sin cobertura en el caché de noticias."""
    pass

@dataclass
class NewsEvent:
    event_id: str
    title: str
    timestamp_utc: datetime
    currency: str
    impact: str

@dataclass
class NewsCalendar:
    events: list[NewsEvent] = field(default_factory=list)
    covered_periods: list[tuple[datetime, datetime]] = field(default_factory=list)
    
    def is_covered(self, timestamp_utc: datetime) -> bool:
        """
        Verifica si el timestamp cae dentro de los periodos de tiempo con certidumbre de datos.
        """
        if not self.covered_periods:
            # Si no hay periodos explícitos pero hay eventos, inferir cobertura desde el min/max
            if not self.events:
                return False
            min_ts = min(ev.timestamp_utc for ev in self.events) - timedelta(days=1)
            max_ts = max(ev.timestamp_utc for ev in self.events) + timedelta(days=1)
            return min_ts <= timestamp_utc <= max_ts
            
        for start_p, end_p in self.covered_periods:
            if start_p <= timestamp_utc <= end_p:
                return True
        return False

    def add_event(self, event: NewsEvent) -> None:
        self.events.append(event)
        
    def add_covered_period(self, start_utc: datetime, end_utc: datetime) -> None:
        self.covered_periods.append((start_utc, end_utc))

def is_blocked_by_news(
    timestamp_utc: datetime,
    calendar: NewsCalendar,
    pre_minutes: int = 0,
    post_minutes: int = 5,
    currencies: set[str] | None = None,
    impacts: set[str] | None = None,
) -> tuple[bool, NewsEvent | None]:
    """
    Motor de filtrado de noticias en tiempo real (UTC puro).
    
    Bajo la configuración Anchor (Sección 2.2), implementa un buffer POST-noticia de 5 minutos:
    Una entrada está permitida solo si current_time >= last_news_high_impact_timestamp + 5 minutes.
    NO hay bloqueo pre-noticia por defecto (pre_minutes = 0).
    
    Retorna: (blocked, blocking_event)
    """
    if not calendar.is_covered(timestamp_utc):
        raise NewsCalendarMissingError(f"Falta de disponibilidad del calendario histórico en caché para: {timestamp_utc}")
        
    target_currs = currencies if currencies is not None else {"USD", "EUR"}
    target_imps = impacts if impacts is not None else {"High"}
    
    blocked = False
    blocking_event: NewsEvent | None = None
    
    # Identificar la última noticia relevante ocurrida antes o en el mismo instante
    # para reproducir con fidelidad el contador last_news_high_impact_timestamp
    relevant_events = [
        ev for ev in calendar.events
        if ev.currency in target_currs and ev.impact in target_imps
    ]
    
    for ev in relevant_events:
        # Definición del intervalo de exclusión: [ev.ts - pre_minutes, ev.ts + post_minutes)
        block_start = ev.timestamp_utc - timedelta(minutes=pre_minutes)
        block_end = ev.timestamp_utc + timedelta(minutes=post_minutes)
        
        if block_start <= timestamp_utc < block_end:
            # Conservar el evento de bloqueo más reciente
            if blocking_event is None or ev.timestamp_utc > blocking_event.timestamp_utc:
                blocking_event = ev
                blocked = True
                
    return blocked, blocking_event

def parse_forex_factory_json(json_content: str | dict) -> NewsCalendar:
    """
    Convierte el formato crudo de ForexFactory a una instancia tipada de NewsCalendar.
    """
    data = json.loads(json_content) if isinstance(json_content, str) else json_content
    calendar = NewsCalendar()
    
    # Asumimos que data es una lista de diccionarios de eventos
    # Ejemplo formato FF: {"title": "NFP", "country": "USD", "impact": "High", "date": "2026-05-01T12:30:00Z"}
    events_list = data if isinstance(data, list) else data.get("events", [])
    
    min_ts: datetime | None = None
    max_ts: datetime | None = None
    
    for idx, item in enumerate(events_list):
        ts_str = item.get("date") or item.get("timestamp")
        if not ts_str:
            continue
            
        # Parse ISO string
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).replace(tzinfo=None)
        
        ev = NewsEvent(
            event_id=str(item.get("id", idx)),
            title=item.get("title", "News"),
            timestamp_utc=ts,
            currency=item.get("country") or item.get("currency", "USD"),
            impact=item.get("impact", "High")
        )
        calendar.add_event(ev)
        
        if min_ts is None or ts < min_ts: min_ts = ts
        if max_ts is None or ts > max_ts: max_ts = ts
        
    if min_ts is not None and max_ts is not None:
        # Cubrir la semana o rango de los eventos extraídos con un margen de seguridad
        calendar.add_covered_period(min_ts - timedelta(days=2), max_ts + timedelta(days=2))
        
    return calendar
