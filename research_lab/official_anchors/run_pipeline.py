"""
Ejecuta el Official Anchor Events Pipeline (free) y escribe dataset + reportes.

No usa red, Forex Factory ni Trading Economics. Conectores: reglas BLS documentadas,
manifiesto JSON local, stubs bloqueados para el resto.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT / "data" / "official_anchors" / "out" / "canonical_anchor_events.csv"
DEFAULT_AUDIT = PROJECT / "data" / "official_anchors" / "out" / "canonical_anchor_events_audit.csv"
DEFAULT_REPORT = PROJECT / "reports" / "official_anchors" / "pipeline_run_report.json"
MANIFEST_PATH = PROJECT / "data" / "official_anchors" / "manifests" / "user_curated_releases.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Official Anchor Events Pipeline (free, official sources).")
    parser.add_argument("--start", default="2024-01-01", help="Inicio (YYYY-MM-DD) inclusive")
    parser.add_argument("--end", default="2026-12-31", help="Fin (YYYY-MM-DD) inclusive")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source-approved", action="store_true", help="NO usar sin auditoria; default False")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    from research_lab.official_anchors.builder import build_canonical_dataframe
    from research_lab.official_anchors.connectors import (
        fetch_bls_employment_situation_events,
        fetch_from_user_manifest,
        stub_bea,
        stub_ecb,
        stub_fed_fomc,
        stub_ism,
    )

    connector_log: list[dict] = []
    all_intermediate: list = []

    cr_bls = fetch_bls_employment_situation_events(start, end)
    connector_log.append(
        {
            "id": cr_bls.connector_id,
            "status": cr_bls.status,
            "message": cr_bls.message,
            "events_emitted": len(cr_bls.events),
            "meta": cr_bls.meta,
        }
    )
    all_intermediate.extend(cr_bls.events)

    cr_man = fetch_from_user_manifest(MANIFEST_PATH)
    connector_log.append(
        {
            "id": cr_man.connector_id,
            "status": cr_man.status,
            "message": cr_man.message,
            "events_emitted": len(cr_man.events),
            "meta": cr_man.meta,
        }
    )
    all_intermediate.extend(cr_man.events)

    for stub_fn, sid in (
        (stub_fed_fomc, "fed_fomc"),
        (stub_ecb, "ecb"),
        (stub_bea, "bea"),
        (stub_ism, "ism"),
    ):
        r = stub_fn()
        connector_log.append(
            {
                "id": r.connector_id,
                "status": r.status,
                "message": r.message,
                "events_emitted": 0,
                "meta": r.meta,
            }
        )

    source_approved = bool(args.source_approved)
    clean_df, audit_df, stats = build_canonical_dataframe(all_intermediate, source_approved=source_approved)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    clean_df.to_csv(args.out, index=False)
    audit_path = args.out.with_name(args.out.stem + "_audit.csv")
    audit_df.to_csv(audit_path, index=False)

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
    DEFAULT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"clean_rows": len(clean_df), "audit_rows": len(audit_df), "report": str(DEFAULT_REPORT)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
