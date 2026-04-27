"""
Employment Situation (NFP + tasa de desempleo) — regla de calendario BLS.

Referencia oficial de hora (8:30 a.m. Eastern):
  https://www.bls.gov/schedule/news_release/empsit.htm

Patrón de publicación: habitualmente el primer viernes del mes (BLS lo documenta;
excepciones por feriados deben verificarse en la página oficial).
"""
from __future__ import annotations

from datetime import date, timedelta

from research_lab.official_anchors.connectors.base import ConnectorResult
from research_lab.official_anchors.schema import IntermediateEvent

BLS_EMPSIT_URL = "https://www.bls.gov/schedule/news_release/empsit.htm"


def _first_friday(d0: date) -> date:
    d = date(d0.year, d0.month, 1)
    while d.weekday() != 4:
        d += timedelta(days=1)
    return d


def _iter_month_starts(start: date, end: date) -> list[date]:
    out: list[date] = []
    y, m = start.year, start.month
    while True:
        cur = date(y, m, 1)
        if cur > end:
            break
        if cur >= date(start.year, start.month, 1):
            out.append(cur)
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
    return out


def fetch_bls_employment_situation_events(start: date, end: date) -> ConnectorResult:
    """Genera eventos intermedios para Employment Situation (08:30 NY)."""
    events: list[IntermediateEvent] = []
    note = (
        "Horario 08:30 ET según BLS. Fecha: primer viernes del mes (regla documentada; "
        "verificar excepciones por feriados en empsit.htm)."
    )
    for ms in _iter_month_starts(start, end):
        fd = _first_friday(ms)
        if fd < start or fd > end:
            continue
        ld = fd.isoformat()
        for title, ag in (
            ("non-farm employment change", "NFP"),
            ("unemployment rate", "UNEMPLOYMENT"),
        ):
            events.append(
                IntermediateEvent(
                    title=title,
                    country="United States",
                    currency="USD",
                    local_date_ny=ld,
                    local_time_ny="08:30",
                    source="bls_employment_situation_rule",
                    source_type="official_rule",
                    source_url=BLS_EMPSIT_URL,
                    anchor_group=ag,
                    notes=note,
                )
            )
    return ConnectorResult(
        connector_id="bls_employment_situation",
        events=events,
        status="partial",
        message=(
            "Regla de primer viernes + hora oficial BLS. No sustituye verificación "
            "mensual del calendario BLS ante feriados."
        ),
        meta={"official_reference": BLS_EMPSIT_URL},
    )
