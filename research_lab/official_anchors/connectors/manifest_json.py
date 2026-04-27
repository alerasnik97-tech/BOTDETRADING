"""
Manifiesto JSON curado localmente desde calendarios oficiales (BLS, Fed, ECB, etc.).

El usuario rellena `releases` copiando fechas/horas desde páginas oficiales.
Sin entradas no se inventan eventos.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from research_lab.official_anchors.connectors.base import ConnectorResult
from research_lab.official_anchors.schema import IntermediateEvent


def fetch_from_user_manifest(path: Path) -> ConnectorResult:
    if not path.is_file():
        return ConnectorResult(
            connector_id="user_manifest_json",
            events=[],
            status="blocked",
            message=f"Manifiesto no encontrado: {path}",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return ConnectorResult(
            connector_id="user_manifest_json",
            events=[],
            status="blocked",
            message=f"JSON inválido: {exc}",
        )
    rows = payload.get("releases") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return ConnectorResult(
            connector_id="user_manifest_json",
            events=[],
            status="blocked",
            message="Falta clave 'releases' (array) en el manifiesto.",
        )
    events: list[IntermediateEvent] = []
    for i, raw in enumerate(rows):
        if not isinstance(raw, dict):
            continue
        try:
            title = str(raw["title"]).strip()
            ld = str(raw["local_date_ny"]).strip()
            tm = str(raw["local_time_ny"]).strip()
            currency = str(raw.get("currency", "USD")).strip().upper()
            country = str(raw.get("country", "United States")).strip()
            src = str(raw.get("source", "user_curated_official")).strip()
            surl = str(raw.get("source_url", "")).strip()
            ag = str(raw.get("anchor_group", "CUSTOM")).strip()
            note = str(raw.get("notes", "")).strip()
            date.fromisoformat(ld)
        except (KeyError, ValueError):
            continue
        events.append(
            IntermediateEvent(
                title=title,
                country=country,
                currency=currency,
                local_date_ny=ld,
                local_time_ny=tm,
                source=src,
                source_type="user_curated_manifest",
                source_url=surl,
                anchor_group=ag,
                notes=note,
            )
        )
    return ConnectorResult(
        connector_id="user_manifest_json",
        events=events,
        status="ok" if events else "partial",
        message=f"{len(events)} eventos desde manifiesto" if events else "Manifiesto vacío (releases: [])",
        meta={"path": str(path)},
    )
