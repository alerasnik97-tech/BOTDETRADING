"""Ensamblado, validación temporal, deduplicación y CSV canónico."""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from research_lab.news_filter import normalize_event_name, stable_hash
from research_lab.official_anchors.schema import CANONICAL_ANCHOR_COLUMNS, IntermediateEvent
from research_lab.official_anchors.time_rules import ny_local_to_utc_iso


def _validate_intermediate(row: IntermediateEvent) -> tuple[str, str]:
    """Retorna (status, notes). status approved_technical o rejected_*."""
    if not row.local_date_ny or not row.local_time_ny:
        return "rejected_missing_local_datetime", "missing_date_or_time_ny"
    try:
        date.fromisoformat(row.local_date_ny)
    except ValueError:
        return "rejected_invalid_local_date", "invalid_local_date_ny"
    parts = row.local_time_ny.split(":")
    if len(parts) != 2:
        return "rejected_invalid_local_time", "invalid_local_time_ny_format"
    try:
        h, m = int(parts[0]), int(parts[1])
        if not (0 <= h <= 23 and 0 <= m <= 59):
            raise ValueError
    except ValueError:
        return "rejected_invalid_local_time", "invalid_local_time_ny_values"
    if not row.title.strip():
        return "rejected_missing_title", "missing_title"
    return "approved_technical", ""


def build_canonical_dataframe(
    intermediates: list[IntermediateEvent],
    *,
    source_approved: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Returns:
        clean_df (solo approved_technical),
        audit_df (todas las filas con status),
        stats dict
    """
    audit_rows: list[dict[str, Any]] = []
    for ev in intermediates:
        st, n = _validate_intermediate(ev)
        sched_utc, sched_ny, tz_src = ("", "", "")
        is_dst = True
        if st == "approved_technical":
            ld = date.fromisoformat(ev.local_date_ny)
            try:
                sched_utc, sched_ny, tz_src = ny_local_to_utc_iso(ld, ev.local_time_ny)
            except Exception as exc:  # noqa: BLE001
                st = "rejected_invalid_datetime"
                n = f"ny_to_utc_failed:{exc}"

        title_norm = normalize_event_name(ev.title)
        dedupe = stable_hash(ev.source, title_norm, sched_utc or ev.local_date_ny, ev.currency)
        eid = stable_hash("official_anchor", dedupe)

        op_elig = st == "approved_technical" and source_approved

        audit_rows.append(
            {
                "event_id": eid,
                "source": ev.source,
                "source_type": ev.source_type,
                "title": title_norm,
                "country": ev.country,
                "currency": ev.currency,
                "importance": ev.importance,
                "anchor_group": ev.anchor_group,
                "scheduled_at_utc": sched_utc,
                "scheduled_at_ny": sched_ny,
                "timezone_source": tz_src if st == "approved_technical" else "",
                "is_dst_sensitive": str(is_dst).lower(),
                "status": st,
                "source_approved": str(source_approved).lower(),
                "operational_eligible": str(op_elig).lower(),
                "source_url": ev.source_url,
                "notes": "; ".join(filter(None, [ev.notes, n])),
                "dedupe_key": dedupe,
            }
        )

    audit_df = pd.DataFrame(audit_rows)
    if audit_df.empty:
        clean_df = pd.DataFrame(columns=list(CANONICAL_ANCHOR_COLUMNS))
        return clean_df, audit_df, {"raw_intermediate": len(intermediates), "approved": 0, "rejected": 0}

    audit_df = audit_df.sort_values(["dedupe_key", "status"]).reset_index(drop=True)
    dup = audit_df.duplicated(subset=["dedupe_key"], keep="first")
    audit_df.loc[dup & (audit_df["status"] == "approved_technical"), "status"] = "rejected_duplicate"
    audit_df.loc[dup & (audit_df["status"] == "rejected_duplicate"), "operational_eligible"] = "false"

    clean_mask = audit_df["status"] == "approved_technical"
    clean_df = audit_df.loc[clean_mask, list(CANONICAL_ANCHOR_COLUMNS)].copy()

    stats = {
        "raw_intermediate": len(intermediates),
        "audit_rows": len(audit_df),
        "technical_approved": int(clean_mask.sum()),
        "technical_rejected": int((~clean_mask).sum()),
        "operational_eligible_rows": int((audit_df["operational_eligible"].astype(str).str.lower() == "true").sum()),
        "status_breakdown": audit_df["status"].value_counts().to_dict(),
    }
    return clean_df, audit_df, stats
