"""
Ejecuta el Official Anchor Events Pipeline usando solo la biblioteca estándar.
Genera CSV sin dependencias externas (sin pandas).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

# Configuración de paths
PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT / "data" / "official_anchors" / "out" / "canonical_anchor_events.csv"
DEFAULT_AUDIT = PROJECT / "data" / "official_anchors" / "out" / "canonical_anchor_events_audit.csv"
DEFAULT_REPORT = PROJECT / "reports" / "official_anchors" / "pipeline_run_report.json"
MANIFEST_PATH = PROJECT / "data" / "official_anchors" / "manifests" / "user_curated_releases.json"

NY_TZ = "America/New_York"


def ny_local_to_utc_iso(local_date: date, hhmm: str) -> tuple[str, str, str]:
    """Convierte fecha civil + hora NY a ISO UTC y NY usando zoneinfo."""
    try:
        import zoneinfo
        import pandas as pd
    except ImportError:
        # Fallback manual sin pandas
        h, m = (int(x) for x in hhmm.split(":"))
        tz = zoneinfo.ZoneInfo(NY_TZ)
        dt_local = datetime.combine(local_date, time(hour=h, minute=m), tzinfo=tz)
        # Calcular UTC manualmente
        import time as timemod
        # Obtener offset de timezone
        offset = dt_local.utcoffset()
        if offset is None:
            offset = timedelta(hours=-5)  # EST aproximado
        dt_utc = dt_local.replace(tzinfo=None) - offset
        utc_str = dt_utc.isoformat() + "+00:00"
        ny_str = dt_local.isoformat()
        return utc_str, ny_str, f"iana_{NY_TZ.replace('/', '_').lower()}"
    
    h, m = (int(x) for x in hhmm.split(":"))
    tz = zoneinfo.ZoneInfo(NY_TZ)
    dt_local = datetime.combine(local_date, time(hour=h, minute=m), tzinfo=tz)
    ts = pd.Timestamp(dt_local)
    utc = ts.tz_convert("UTC")
    ny = ts.tz_convert(NY_TZ)
    return utc.isoformat(), ny.isoformat(), f"iana_{NY_TZ.replace('/', '_').lower()}"


def stable_hash(*parts: str) -> str:
    """Hash estable para deduplicación."""
    import hashlib
    data = "|".join(parts)
    return hashlib.md5(data.encode("utf-8")).hexdigest()[:16]


def normalize_event_name(name: str) -> str:
    """Normaliza nombre de evento."""
    return name.lower().strip()


def _validate_intermediate(row: dict) -> tuple[str, str]:
    """Valida un evento intermedio. Retorna (status, notes)."""
    if not row.get("local_date_ny") or not row.get("local_time_ny"):
        return "rejected_missing_local_datetime", "missing_date_or_time_ny"
    try:
        date.fromisoformat(row["local_date_ny"])
    except ValueError:
        return "rejected_invalid_local_date", "invalid_local_date_ny"
    parts = row["local_time_ny"].split(":")
    if len(parts) != 2:
        return "rejected_invalid_local_time", "invalid_local_time_ny_format"
    try:
        h, m = int(parts[0]), int(parts[1])
        if not (0 <= h <= 23 and 0 <= m <= 59):
            raise ValueError
    except ValueError:
        return "rejected_invalid_local_time", "invalid_local_time_ny_values"
    if not row.get("title", "").strip():
        return "rejected_missing_title", "missing_title"
    return "approved_technical", ""


def _first_friday(d0: date) -> date:
    """Primer viernes del mes."""
    d = date(d0.year, d0.month, 1)
    while d.weekday() != 4:
        d += timedelta(days=1)
    return d


def _iter_month_starts(start: date, end: date) -> list[date]:
    """Itera sobre el primer día de cada mes en el rango."""
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


def fetch_bls_employment_situation_events(start: date, end: date) -> dict:
    """Genera eventos NFP desde regla BLS (primer viernes del mes, 08:30 ET)."""
    events = []
    note = "Horario 08:30 ET según BLS. Primer viernes del mes. Verificar excepciones por feriados."
    BLS_EMPSIT_URL = "https://www.bls.gov/schedule/news_release/empsit.htm"
    
    for ms in _iter_month_starts(start, end):
        fd = _first_friday(ms)
        if fd < start or fd > end:
            continue
        ld = fd.isoformat()
        for title, ag in (("non-farm employment change", "NFP"), ("unemployment rate", "UNEMPLOYMENT")):
            events.append({
                "title": title,
                "country": "United States",
                "currency": "USD",
                "local_date_ny": ld,
                "local_time_ny": "08:30",
                "source": "bls_employment_situation_rule",
                "source_type": "official_rule",
                "source_url": BLS_EMPSIT_URL,
                "anchor_group": ag,
                "notes": note,
            })
    
    return {
        "connector_id": "bls_employment_situation",
        "events": events,
        "status": "partial",
        "message": "Regla de primer viernes + hora oficial BLS. Verificar feriados.",
        "meta": {"official_reference": BLS_EMPSIT_URL},
    }


def fetch_from_user_manifest(path: Path) -> dict:
    """Carga eventos desde el manifest JSON curado por el usuario."""
    if not path.is_file():
        return {
            "connector_id": "user_manifest_json",
            "events": [],
            "status": "blocked",
            "message": f"Manifiesto no encontrado: {path}",
        }
    
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "connector_id": "user_manifest_json",
            "events": [],
            "status": "blocked",
            "message": f"JSON inválido: {exc}",
        }
    
    rows = payload.get("releases") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return {
            "connector_id": "user_manifest_json",
            "events": [],
            "status": "blocked",
            "message": "Falta clave 'releases' (array) en el manifiesto.",
        }
    
    events = []
    for raw in rows:
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
            date.fromisoformat(ld)  # Validar fecha
        except (KeyError, ValueError):
            continue
        
        events.append({
            "title": title,
            "country": country,
            "currency": currency,
            "local_date_ny": ld,
            "local_time_ny": tm,
            "source": src,
            "source_type": "user_curated_manifest",
            "source_url": surl,
            "anchor_group": ag,
            "notes": note,
        })
    
    return {
        "connector_id": "user_manifest_json",
        "events": events,
        "status": "ok" if events else "partial",
        "message": f"{len(events)} eventos desde manifiesto" if events else "Manifiesto vacío (releases: [])",
        "meta": {"path": str(path)},
    }


def stub_fed_fomc() -> dict:
    return {
        "connector_id": "fed_fomc_official",
        "events": [],
        "status": "blocked",
        "message": "FOMC ya cubierto por user_curated_releases.json. Este stub permanece para compatibilidad.",
        "meta": {"hint": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"},
    }


def stub_ecb() -> dict:
    return {
        "connector_id": "ecb_official",
        "events": [],
        "status": "blocked",
        "message": "ECB ya cubierto por user_curated_releases.json. Este stub permanece para compatibilidad.",
        "meta": {"hint": "https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html"},
    }


def stub_bea() -> dict:
    return {
        "connector_id": "bea_official",
        "events": [],
        "status": "blocked",
        "message": "BEA (Core PCE): requiere manifest curado o conector futuro desde bea.gov/news/schedule",
        "meta": {"hint": "https://www.bea.gov/news/schedule"},
    }


def stub_ism() -> dict:
    return {
        "connector_id": "ism_official",
        "events": [],
        "status": "blocked",
        "message": "ISM Manufacturing/Services: requiere manifest curado o feed estructurado oficial",
        "meta": {"hint": "https://www.ismworld.org/supply-management-news-and-reports/reports/ism-report-on-business/"},
    }


CANONICAL_ANCHOR_COLUMNS = (
    "event_id",
    "source",
    "source_type",
    "title",
    "country",
    "currency",
    "importance",
    "anchor_group",
    "scheduled_at_utc",
    "scheduled_at_ny",
    "timezone_source",
    "is_dst_sensitive",
    "status",
    "source_approved",
    "operational_eligible",
    "source_url",
    "notes",
)


def build_canonical_dataframe(intermediates: list[dict], source_approved: bool = False) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """
    Procesa eventos intermedios y genera filas de audit y clean.
    Retorna: (clean_rows, audit_rows, stats)
    """
    audit_rows = []
    
    for ev in intermediates:
        st, n = _validate_intermediate(ev)
        sched_utc, sched_ny, tz_src = "", "", ""
        is_dst = True
        
        if st == "approved_technical":
            ld = date.fromisoformat(ev["local_date_ny"])
            try:
                sched_utc, sched_ny, tz_src = ny_local_to_utc_iso(ld, ev["local_time_ny"])
            except Exception as exc:
                st = "rejected_invalid_datetime"
                n = f"ny_to_utc_failed:{exc}"
        
        title_norm = normalize_event_name(ev["title"])
        dedupe = stable_hash(ev["source"], title_norm, sched_utc or ev["local_date_ny"], ev["currency"])
        eid = stable_hash("official_anchor", dedupe)
        
        op_elig = st == "approved_technical" and source_approved
        
        audit_rows.append({
            "event_id": eid,
            "source": ev["source"],
            "source_type": ev["source_type"],
            "title": title_norm,
            "country": ev["country"],
            "currency": ev["currency"],
            "importance": ev.get("importance", "HIGH"),
            "anchor_group": ev["anchor_group"],
            "scheduled_at_utc": sched_utc,
            "scheduled_at_ny": sched_ny,
            "timezone_source": tz_src if st == "approved_technical" else "",
            "is_dst_sensitive": str(is_dst).lower(),
            "status": st,
            "source_approved": str(source_approved).lower(),
            "operational_eligible": str(op_elig).lower(),
            "source_url": ev["source_url"],
            "notes": "; ".join(filter(None, [ev.get("notes", ""), n])),
            "dedupe_key": dedupe,
        })
    
    if not audit_rows:
        return [], [], {"raw_intermediate": len(intermediates), "approved": 0, "rejected": 0}
    
    # Ordenar por dedupe_key y status
    audit_rows.sort(key=lambda x: (x["dedupe_key"], x["status"]))
    
    # Marcar duplicados
    seen_keys = set()
    for row in audit_rows:
        if row["dedupe_key"] in seen_keys:
            if row["status"] == "approved_technical":
                row["status"] = "rejected_duplicate"
            if row["status"] == "rejected_duplicate":
                row["operational_eligible"] = "false"
        else:
            seen_keys.add(row["dedupe_key"])
    
    clean_rows = [row for row in audit_rows if row["status"] == "approved_technical"]
    
    stats = {
        "raw_intermediate": len(intermediates),
        "audit_rows": len(audit_rows),
        "technical_approved": len(clean_rows),
        "technical_rejected": len(audit_rows) - len(clean_rows),
        "operational_eligible_rows": sum(1 for r in audit_rows if r["operational_eligible"] == "true"),
        "status_breakdown": {},
    }
    
    # Contar status
    for row in audit_rows:
        st = row["status"]
        stats["status_breakdown"][st] = stats["status_breakdown"].get(st, 0) + 1
    
    return clean_rows, audit_rows, stats


def write_csv(rows: list[dict], path: Path, columns: tuple[str, ...]) -> None:
    """Escribe CSV con columnas específicas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Official Anchor Events Pipeline (stdlib only).")
    parser.add_argument("--start", default="2024-01-01", help="Inicio (YYYY-MM-DD) inclusive")
    parser.add_argument("--end", default="2026-12-31", help="Fin (YYYY-MM-DD) inclusive")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source-approved", action="store_true", help="NO usar sin auditoria; default False")
    args = parser.parse_args()
    
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    
    connector_log = []
    all_intermediate = []
    
    # BLS Employment Situation (NFP)
    cr_bls = fetch_bls_employment_situation_events(start, end)
    connector_log.append({
        "id": cr_bls["connector_id"],
        "status": cr_bls["status"],
        "message": cr_bls["message"],
        "events_emitted": len(cr_bls["events"]),
        "meta": cr_bls["meta"],
    })
    all_intermediate.extend(cr_bls["events"])
    
    # User manifest
    cr_man = fetch_from_user_manifest(MANIFEST_PATH)
    connector_log.append({
        "id": cr_man["connector_id"],
        "status": cr_man["status"],
        "message": cr_man["message"],
        "events_emitted": len(cr_man["events"]),
        "meta": cr_man["meta"],
    })
    all_intermediate.extend(cr_man["events"])
    
    # BLS CPI/PPI Híbrido (multi-capa oficial)
    try:
        sys.path.insert(0, str(PROJECT / "research_lab" / "official_anchors" / "connectors"))
        from bls_cpi_ppi_hybrid import fetch_bls_cpi_ppi_hybrid
        cr_cpi_ppi = fetch_bls_cpi_ppi_hybrid([start.year, end.year])
        connector_log.append({
            "id": cr_cpi_ppi["connector_id"],
            "status": cr_cpi_ppi["status"],
            "message": cr_cpi_ppi["message"],
            "events_emitted": len(cr_cpi_ppi["events"]),
            "meta": cr_cpi_ppi["meta"],
        })
        all_intermediate.extend(cr_cpi_ppi["events"])
    except Exception as e:
        connector_log.append({
            "id": "bls_cpi_ppi_hybrid",
            "status": "error",
            "message": f"Connector error: {type(e).__name__}: {e}",
            "events_emitted": 0,
            "meta": {},
        })
    
    # Stubs (bloqueados, para documentación)
    for stub_fn, sid in (
        (stub_fed_fomc, "fed_fomc"),
        (stub_ecb, "ecb"),
        (stub_bea, "bea"),
        (stub_ism, "ism"),
    ):
        r = stub_fn()
        connector_log.append({
            "id": r["connector_id"],
            "status": r["status"],
            "message": r["message"],
            "events_emitted": 0,
            "meta": r["meta"],
        })
    
    source_approved = bool(args.source_approved)
    clean_rows, audit_rows, stats = build_canonical_dataframe(all_intermediate, source_approved=source_approved)
    
    # Escribir CSVs
    write_csv(clean_rows, args.out, CANONICAL_ANCHOR_COLUMNS)
    audit_path = args.out.with_name(args.out.stem + "_audit.csv")
    write_csv(audit_rows, audit_path, CANONICAL_ANCHOR_COLUMNS + ("dedupe_key",))
    
    # Generar reporte
    report = {
        "pipeline": "official_anchor_events_free",
        "source_approved_config": source_approved,
        "range": {"start": args.start, "end": args.end},
        "output_clean_csv": str(args.out),
        "output_audit_csv": str(audit_path),
        "connectors": connector_log,
        "build_stats": stats,
        "policy": {
            "utc_canonical_field": "scheduled_at_utc",
            "ny_derived_field": "scheduled_at_ny",
            "operational_eligible_requires_source_approved": True,
            "default_source_approved": False,
        },
    }
    
    DEFAULT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    
    print(json.dumps({"clean_rows": len(clean_rows), "audit_rows": len(audit_rows), "report": str(DEFAULT_REPORT)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
