"""Conectores bloqueados: requieren archivo oficial local o automatización futura auditada."""
from __future__ import annotations

from research_lab.official_anchors.connectors.base import ConnectorResult


def stub_fed_fomc() -> ConnectorResult:
    return ConnectorResult(
        connector_id="fed_fomc_official",
        events=[],
        status="blocked",
        message=(
            "La Reserva Federal publica calendarios de FOMC en federalreserve.gov "
            "(PDF/HTML). Sin descarga oficial local ni parser estable aprobado, "
            "este conector permanece bloqueado."
        ),
        meta={"hint": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"},
    )


def stub_ecb() -> ConnectorResult:
    return ConnectorResult(
        connector_id="ecb_official",
        events=[],
        status="blocked",
        message=(
            "ECB publica calendario de reuniones y conferencias. Integración estable "
            "pendiente (ICS/PDF oficial o filas en user_curated_releases.json)."
        ),
        meta={"hint": "https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html"},
    )


def stub_bea() -> ConnectorResult:
    return ConnectorResult(
        connector_id="bea_official",
        events=[],
        status="blocked",
        message="GDP advance/prelim/final y datos BEA: usar manifiesto curado desde bea.gov o conector futuro.",
        meta={"hint": "https://www.bea.gov/news/schedule"},
    )


def stub_ism() -> ConnectorResult:
    return ConnectorResult(
        connector_id="ism_official",
        events=[],
        status="blocked",
        message=(
            "ISM publica fechas en ismworld.org; sin feed estructurado oficial "
            "incluido en el repo, permanece bloqueado."
        ),
        meta={"hint": "https://www.ismworld.org/supply-management-news-and-reports/reports/ism-report-on-business/"},
    )
