"""
Fase 3: validación masiva del pipeline UTC-first (Trading Economics -> schema V2).

Solo lectura del código de negocio; escribe artefactos bajo reports/news_reliability/.
Requiere un export TE (JSON/CSV) dentro del proyecto, por defecto:
  data/news_imports/tradingeconomics_calendar.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import zoneinfo

from research_lab.config import DEFAULT_TRADING_ECONOMICS_IMPORT_DIR, NewsConfig, NY_TZ
from research_lab.news_filter import SUPPORTED_FIXED_SCHEDULES_NY
from research_lab.news_tradingeconomics import CANONICAL_COLUMNS_V2, import_tradingeconomics_calendar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "news_imports" / "tradingeconomics_calendar.json"
DEFAULT_OUTPUT_STEM = PROJECT_ROOT / "reports" / "news_reliability" / "phase3_run" / "news_te_validated.csv"
REPORT_JSON = PROJECT_ROOT / "reports" / "news_reliability" / "phase3_execution_report.json"
REPORT_FULL_JSON = PROJECT_ROOT / "reports" / "news_reliability" / "phase3_mass_report_full.json"
INPUT_VERIFICATION = PROJECT_ROOT / "reports" / "news_reliability" / "PHASE3_T1_INPUT_VERIFICATION.txt"

REQUIRED_CLEAN_FIELDS = {
    "event_id",
    "source",
    "title",
    "country",
    "currency",
    "importance",
    "scheduled_at_utc",
    "scheduled_at_ny",
    "timezone_source",
    "source_approved",
    "status",
    "operational_eligible",
}

ANCHOR_TITLES = frozenset(SUPPORTED_FIXED_SCHEDULES_NY.keys())


def _ensure_under_project(path: Path) -> Path:
    resolved = path.resolve()
    root = PROJECT_ROOT.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Ruta fuera del proyecto: {path}") from exc
    return resolved


def _dst_bucket(ts_utc: pd.Timestamp) -> str:
    ny = ts_utc.tz_convert(zoneinfo.ZoneInfo(NY_TZ))
    eu = ts_utc.tz_convert(zoneinfo.ZoneInfo("Europe/Berlin"))
    return f"ny{ny.strftime('%z')}_eu{eu.strftime('%z')}"


def _week_key(ts_utc: pd.Timestamp) -> str:
    iso = ts_utc.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _user_rejection_breakdown(audit: pd.DataFrame, raw_n: int, source_approved_config: bool) -> dict[str, Any]:
    """Mapea status internos a etiquetas de informe; porcentajes sobre raw_n."""
    if audit.empty:
        return {"table": [], "note": "audit vacio"}

    empty_utc = audit["scheduled_at_utc"].astype(str).str.strip() == ""
    bad_ts = audit["status"] == "rejected_bad_timestamp"
    missing_utc = int((bad_ts & empty_utc).sum())
    invalid_utc = int((bad_ts & ~empty_utc).sum())

    dup = int((audit["status"] == "rejected_duplicate").sum())
    mismatch = int((audit["status"] == "rejected_time_mismatch").sum())
    bad_imp = int((audit["status"] == "rejected_bad_importance").sum())
    bad_cur = int((audit["status"] == "rejected_irrelevant_currency").sum())

    tech_ok = audit["status"] == "approved"
    op_elig = (
        audit["operational_eligible"].astype(str).str.lower().isin(["true", "1"])
        if "operational_eligible" in audit.columns
        else pd.Series(False, index=audit.index)
    )
    if not source_approved_config:
        source_not_approved_block = int(tech_ok.sum())
    else:
        source_not_approved_block = int((tech_ok & ~op_elig).sum())

    other = int(
        (
            ~audit["status"].isin(
                [
                    "approved",
                    "rejected_bad_timestamp",
                    "rejected_duplicate",
                    "rejected_time_mismatch",
                    "rejected_bad_importance",
                    "rejected_irrelevant_currency",
                ]
            )
        ).sum()
    )

    def pct(x: int) -> float:
        return round(100.0 * x / raw_n, 4) if raw_n else 0.0

    rows = [
        ("missing_scheduled_utc", missing_utc, pct(missing_utc)),
        ("invalid_scheduled_utc", invalid_utc, pct(invalid_utc)),
        ("duplicate_event", dup, pct(dup)),
        ("rejected_time_mismatch", mismatch, pct(mismatch)),
        (
            "source_not_approved_operational_block",
            source_not_approved_block,
            pct(source_not_approved_block),
        ),
        ("rejected_bad_importance", bad_imp, pct(bad_imp)),
        ("rejected_irrelevant_currency", bad_cur, pct(bad_cur)),
        ("other_status", other, pct(other)),
    ]
    return {
        "table": [{"reason": a, "count": b, "pct_of_raw": c} for a, b, c in rows],
        "note_source_not_approved": (
            "Filas tecnicamente approved pero operational_eligible=False por config. "
            "No es status de rechazo en fila; es bloqueo operativo explícito."
        ),
    }


def _sample_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "title",
        "currency",
        "importance",
        "scheduled_at_utc",
        "scheduled_at_ny",
        "status",
        "operational_eligible",
        "notes",
    )
    out = []
    for r in rows:
        out.append({k: r.get(k, "") for k in keys})
    return out


def _build_samples(audit: pd.DataFrame, clean: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ap = clean.head(5).to_dict(orient="records") if len(clean) else []
    mm = audit.loc[audit["status"] == "rejected_time_mismatch"].head(5).to_dict(orient="records")
    other_mask = ~audit["status"].isin(["approved", "rejected_time_mismatch", "rejected_duplicate"])
    others = audit.loc[other_mask].head(5).to_dict(orient="records")
    if len(others) < 5:
        dup = audit.loc[audit["status"] == "rejected_duplicate"].head(5 - len(others))
        others = others + dup.to_dict(orient="records")
    out["five_accepted_clean"] = _sample_fields(ap)
    out["five_rejected_time_mismatch_audit"] = _sample_fields(mm)
    out["five_rejected_other_audit"] = _sample_fields(others[:5])
    return out


def analyze_phase3(result: Any, *, source_approved_config: bool) -> dict[str, Any]:
    audit = result.audit_frame.copy()
    clean = result.clean_frame.copy()
    raw_n = int(result.summary.get("raw_rows", len(audit)))

    if len(audit) == 0:
        return {
            "pipeline": "trading_economics_utc_first_v2",
            "source_approved_config": source_approved_config,
            "metrics_global": {
                "raw_rows": raw_n,
                "audit_rows": 0,
                "technical_approved_rows": 0,
                "operational_eligible_rows": 0,
                "technical_rejected_rows": 0,
                "pct_technical_acceptance_vs_raw": 0.0,
                "pct_operational_eligibility_vs_raw": 0.0,
                "pct_technical_rejection_vs_raw": 0.0,
                "clean_dataset_rows_same_as_technical_approved": 0,
                "clean_rows_operationally_blocked": 0,
            },
            "user_facing_rejection_breakdown": {"table": [], "note": "audit vacio"},
            "integrity_verdict_suggested": "no_confiable",
            "manual_samples": {
                "five_accepted_clean": [],
                "five_rejected_time_mismatch_audit": [],
                "five_rejected_other_audit": [],
            },
        }

    total_audit = len(audit)
    technical_approved = int((audit["status"] == "approved").sum())
    op_eligible_series = (
        audit["operational_eligible"].astype(str).str.lower().isin(["true", "1"])
        if "operational_eligible" in audit.columns
        else pd.Series(False, index=audit.index)
    )
    operational_eligible_n = int(op_eligible_series.sum())
    accepted = technical_approved
    rejected_technical = raw_n - technical_approved if raw_n == total_audit else total_audit - technical_approved
    pct_tech_ok = round(100.0 * technical_approved / raw_n, 4) if raw_n else 0.0
    pct_op_ok = round(100.0 * operational_eligible_n / raw_n, 4) if raw_n else 0.0
    pct_rej_tech = round(100.0 * rejected_technical / raw_n, 4) if raw_n else 0.0

    status_counts = audit["status"].value_counts().to_dict() if len(audit) else {}
    status_pct = {k: round(100.0 * v / raw_n, 4) for k, v in status_counts.items()} if raw_n else {}

    notes_counter = Counter()
    for _, row in audit.iterrows():
        st = str(row.get("status", ""))
        n = str(row.get("notes", "") or "")
        if n:
            notes_counter[f"{st}|{n}"] += 1

    audit["ts_utc"] = pd.to_datetime(audit["scheduled_at_utc"], utc=True, errors="coerce")
    valid_ts = audit["ts_utc"].notna()

    tmp_ts = audit.loc[valid_ts].copy()
    tmp_ts["year"] = tmp_ts["ts_utc"].dt.year
    tmp_ts["ym"] = tmp_ts["ts_utc"].dt.to_period("M").astype(str)
    dist_year = tmp_ts.groupby(["year", "status"]).size().unstack(fill_value=0)
    dist_month = tmp_ts.groupby(["ym", "status"]).size().unstack(fill_value=0)
    dist_currency = audit.groupby(["currency", "status"]).size().unstack(fill_value=0)
    dist_importance = audit.groupby(["importance", "status"]).size().unstack(fill_value=0)

    dst_profile = audit.loc[valid_ts].copy()
    dst_profile["dst_bucket"] = dst_profile["ts_utc"].apply(_dst_bucket)
    dst_by_bucket = dst_profile.groupby(["dst_bucket", "status"]).size().unstack(fill_value=0)

    asymmetric_weeks: dict[str, dict[str, int]] = {}
    sub = audit.loc[valid_ts, ["ts_utc", "status"]].copy()
    sub["week"] = sub["ts_utc"].apply(_week_key)
    sub["bucket"] = sub["ts_utc"].apply(_dst_bucket)
    for wk, grp in sub.groupby("week"):
        buckets = set(grp["bucket"])
        if len(buckets) > 1:
            c_pass = int((grp["status"] == "approved").sum())
            c_fail = int((grp["status"] != "approved").sum())
            asymmetric_weeks[wk] = {"approved": c_pass, "not_approved": c_fail, "dst_buckets_seen": len(buckets)}

    anchor_audit = audit[audit["title"].isin(ANCHOR_TITLES)]
    anchor_summary = anchor_audit.groupby(["title", "status"]).size().unstack(fill_value=0)
    anchor_samples = (
        anchor_audit.sort_values(["title", "scheduled_at_utc"])
        .groupby("title", group_keys=False)
        .head(3)
        .to_dict(orient="records")
    )

    dup_rej = int((audit["status"] == "rejected_duplicate").sum())
    dedupe_keys = audit["dedupe_key"].astype(str)
    dup_keys_total = int(dedupe_keys.duplicated().sum())

    event_ids = clean["event_id"].astype(str)
    collision_ids = int(event_ids.duplicated().sum()) if len(clean) else 0

    title_time = audit.loc[valid_ts, ["title", "scheduled_at_utc", "event_id"]].copy()
    same_title_diff_utc = 0
    for title, g in title_time.groupby("title"):
        if g["scheduled_at_utc"].nunique() > 1:
            same_title_diff_utc += 1

    same_utc_diff_currency = 0
    utc_groups = audit.loc[valid_ts].groupby("scheduled_at_utc")
    for _, g in utc_groups:
        if g["currency"].nunique() > 1 or g["title"].apply(lambda x: str(x)[:40]).nunique() > 3:
            same_utc_diff_currency += 1

    schema_issues: list[str] = []
    missing_cols = REQUIRED_CLEAN_FIELDS.difference(set(clean.columns))
    if missing_cols:
        schema_issues.append(f"clean_missing_columns:{sorted(missing_cols)}")
    extra_expected = {"actual", "forecast", "previous"}
    for col in CANONICAL_COLUMNS_V2:
        if col not in clean.columns:
            schema_issues.append(f"clean_missing_canonical:{col}")
    unapproved_in_clean = 0
    if len(clean) and "operational_eligible" in clean.columns:
        elig = clean["operational_eligible"].astype(str).str.lower().isin(["true", "1"])
        unapproved_in_clean = int((~elig).sum())

    user_rej = _user_rejection_breakdown(audit, raw_n, source_approved_config)

    integrity_verdict = "usable_con_cautela"
    if raw_n == 0:
        integrity_verdict = "no_confiable"
    else:
        mm_pct = 100.0 * int((audit["status"] == "rejected_time_mismatch").sum()) / raw_n
        dup_pct = 100.0 * int((audit["status"] == "rejected_duplicate").sum()) / raw_n
        if mm_pct > 15.0 or dup_pct > 30.0:
            integrity_verdict = "no_confiable"
        elif raw_n >= 500 and technical_approved < 50:
            integrity_verdict = "no_confiable"
        elif raw_n >= 500 and mm_pct < 2.0 and dup_pct < 5.0 and technical_approved > 500:
            integrity_verdict = "robusto_activacion_controlada"

    return {
        "pipeline": "trading_economics_utc_first_v2",
        "source_approved_config": source_approved_config,
        "metrics_global": {
            "raw_rows": raw_n,
            "audit_rows": total_audit,
            "technical_approved_rows": technical_approved,
            "operational_eligible_rows": operational_eligible_n,
            "technical_rejected_rows": rejected_technical,
            "pct_technical_acceptance_vs_raw": pct_tech_ok,
            "pct_operational_eligibility_vs_raw": pct_op_ok,
            "pct_technical_rejection_vs_raw": pct_rej_tech,
            "clean_dataset_rows_same_as_technical_approved": len(clean),
            "clean_rows_operationally_blocked": unapproved_in_clean,
            "note_operational_blocked": (
                "Con source_approved=False, operational_eligible=0 para todas las filas; "
                "el feed no es utilizable operativamente hasta aprobacion explicita."
            ),
        },
        "user_facing_rejection_breakdown": user_rej,
        "rejections_by_status": status_counts,
        "integrity_verdict_suggested": integrity_verdict,
        "integrity_verdict_note": "Heuristica automatica (mismatch/dup/volumen); revisar manualmente antes de activar.",
        "rejections_by_status_pct_of_raw": status_pct,
        "rejections_by_status_and_notes_top": dict(notes_counter.most_common(40)),
        "distribution_year_status": dist_year.to_dict() if len(dist_year) else {},
        "distribution_month_status": dist_month.to_dict() if len(dist_month) else {},
        "distribution_currency_status": dist_currency.to_dict() if len(dist_currency) else {},
        "distribution_importance_status": dist_importance.to_dict() if len(dist_importance) else {},
        "dst_audit": {
            "counts_by_ny_eu_offset_combo_and_status": dst_by_bucket.to_dict() if len(dst_by_bucket) else {},
            "weeks_with_multiple_dst_buckets_count": len(asymmetric_weeks),
            "asymmetric_weeks_sample": dict(list(sorted(asymmetric_weeks.items()))[:12]),
        },
        "anchor_events": {
            "summary_title_status": anchor_summary.to_dict() if len(anchor_summary) else {},
            "sample_rows_per_title_max3": anchor_samples,
        },
        "duplicates_and_collisions": {
            "rejected_duplicate_rows": dup_rej,
            "duplicate_dedupe_key_rows_in_audit": dup_keys_total,
            "clean_duplicate_event_id": collision_ids,
            "titles_with_multiple_distinct_utc_timestamps": same_title_diff_utc,
            "utc_timestamps_with_heterogeneous_metadata_groups_flagged": same_utc_diff_currency,
        },
        "schema_integrity": {
            "canonical_columns_expected": list(CANONICAL_COLUMNS_V2),
            "issues": schema_issues,
        },
        "manual_samples": _build_samples(audit, clean),
        "policy_notes": {
            "scheduled_at_utc_canonical": True,
            "scheduled_at_ny_derived_iana": NY_TZ,
            "no_autocorrect": True,
            "fail_closed_news_enabled": "usa NewsConfig.enabled en runtime (no modificado aqui)",
            "source_approved_flag_in_clean": "Las filas aprobadas pueden tener source_approved=False; es bloqueo operativo, no rechazo de ingesta.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fase 3: validacion masiva pipeline noticias V2 (TE).")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Export TE JSON/CSV dentro del proyecto")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_STEM,
        help="CSV limpio de salida (bajo reports/ por defecto)",
    )
    parser.add_argument("--currencies", nargs="*", default=["USD", "EUR"])
    parser.add_argument("--approved", action="store_true", help="Solo para prueba; NO usar en produccion sin auditoria")
    args = parser.parse_args()

    inp = _ensure_under_project(args.input)
    outp = _ensure_under_project(args.output)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    outp.parent.mkdir(parents=True, exist_ok=True)

    if not inp.is_file():
        alt: list[str] = []
        skip_parts = {".git", "_tmp", "__pycache__", ".venv", ".venv_fixed"}
        for p in PROJECT_ROOT.rglob("*.json"):
            try:
                p.relative_to(PROJECT_ROOT.resolve())
            except ValueError:
                continue
            if any(part in skip_parts for part in p.parts):
                continue
            name = p.name.lower()
            if "calendar" in name or "trading" in name or "economics" in name:
                alt.append(str(p.relative_to(PROJECT_ROOT)))
        lines = [
            "PHASE3 T1 - Verificacion de input TE",
            f"Ruta esperada (default): {DEFAULT_INPUT}",
            f"Existe: False",
            "",
            "Candidatos *.json bajo el proyecto con nombre calendar/trading/economics:",
        ]
        lines.extend(alt if alt else ["(ninguno)"])
        lines.append("")
        lines.append("Contenido actual data/news_imports/:")
        imp = PROJECT_ROOT / "data" / "news_imports"
        if imp.is_dir():
            for c in sorted(imp.iterdir()):
                lines.append(f"  {c.name}  ({c.stat().st_size} bytes)")
        else:
            lines.append("  (directorio inexistente)")
        INPUT_VERIFICATION.parent.mkdir(parents=True, exist_ok=True)
        INPUT_VERIFICATION.write_text("\n".join(lines), encoding="utf-8")

        payload = {
            "phase": "3_mass_validation",
            "status": "BLOCKED_NO_INPUT",
            "message": (
                f"No existe el archivo de entrada TE: {inp}. "
                f"Coloque el export (JSON o CSV) bajo {DEFAULT_TRADING_ECONOMICS_IMPORT_DIR} "
                "o pase --input con ruta dentro del proyecto."
            ),
            "expected_default_path": str(DEFAULT_INPUT),
            "metrics_global": None,
            "verdict_te_pipeline_data": "no_evaluable_sin_dataset_te",
            "input_verification_report": str(INPUT_VERIFICATION),
            "alternate_json_candidates": alt,
            "phase_3_5_status_report": str(PROJECT_ROOT / "reports" / "news_reliability" / "PHASE3_5_STATUS.txt"),
            "te_input_readme": str(PROJECT_ROOT / "data" / "news_imports" / "README_TE_INPUT.txt"),
            "ingestion_operational_gate": "operational_eligible en CSV V2; ver news_tradingeconomics.summary JSON",
        }
        REPORT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 2

    settings = NewsConfig(enabled=False, source_approved=bool(args.approved), currencies=tuple(args.currencies))
    result = import_tradingeconomics_calendar(
        inp,
        clean_output_path=outp,
        settings=settings,
        allowed_currencies=tuple(args.currencies),
    )

    analysis = analyze_phase3(result, source_approved_config=bool(args.approved))
    analysis["phase"] = "3_mass_validation"
    analysis["status"] = "OK"
    analysis["input_path"] = str(inp)
    analysis["output_clean_path"] = str(outp)
    analysis["audit_path"] = str(outp.with_name(outp.stem + "_audit.csv"))
    analysis["summary_path"] = str(outp.with_name(outp.stem + "_summary.json"))

    REPORT_JSON.write_text(json.dumps(analysis, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    REPORT_FULL_JSON.write_text(json.dumps(analysis, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(analysis["metrics_global"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
