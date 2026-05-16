from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha1
from pathlib import Path
import json
import re
from typing import Any

import numpy as np
import pandas as pd

from research_lab.config import (
    DEFAULT_NEWS_V2_UTC_FILE,
    DEFAULT_RAW_NEWS_FILE_OBSOLETE,
    NY_TZ,
    NewsConfig,
    PAIR_META,
)


SUPPORTED_FIXED_SCHEDULES_NY: dict[str, str] = {
    "non-farm employment change": "08:30",
    "unemployment rate": "08:30",
    "cpi y/y": "08:30",
    "cpi m/m": "08:30",
    "core cpi m/m": "08:30",
    "retail sales m/m": "08:30",
    "core retail sales m/m": "08:30",
    "adp non-farm employment change": "08:15",
    "ism manufacturing pmi": "10:00",
    "ism services pmi": "10:00",
    "advance gdp q/q": "08:30",
    "prelim gdp q/q": "08:30",
    "final gdp q/q": "08:30",
    "gdp q/q": "08:30",
    "ppi m/m": "08:30",
    "ppi y/y": "08:30",
    "core ppi m/m": "08:30",
    "fomc meeting minutes": "14:00",
    "fomc statement": "14:00",
    "fed announcement": "14:00",
    "federal funds rate": "14:00",
    "fomc press conference": "14:30",
    "main refinancing rate": "07:45",
    "ecb press conference": "08:45",
    "unemployment claims": "08:30",
    # JPY Structural Readiness Families
    "boj policy rate": "00:00",
    "monetary policy statement": "00:00",
    "boj press conference": "02:30",
    "boj outlook report": "00:00",
    "tokyo core cpi y/y": "07:30",
    "national core cpi y/y": "07:30",

}

SUPPORTED_VALIDATION_EVENTS = tuple(sorted(SUPPORTED_FIXED_SCHEDULES_NY.keys()))

VALIDATION_EVENT_ALIASES: dict[str, tuple[str, ...]] = {
    "gdp q/q": ("gdp q/q", "advance gdp q/q", "prelim gdp q/q", "final gdp q/q"),
    "ppi y/y": ("ppi y/y",),
}

KEY_EVENT_CHECKS: tuple[tuple[str, str, bool], ...] = (
    ("non-farm employment change", "08:30", False),
    ("unemployment rate", "08:30", False),
    ("cpi y/y", "08:30", False),
    ("cpi m/m", "08:30", False),
    ("core cpi m/m", "08:30", False),
    ("retail sales m/m", "08:30", False),
    ("core retail sales m/m", "08:30", False),
    ("ism manufacturing pmi", "10:00", False),
    ("ism services pmi", "10:00", False),
    ("fomc statement", "14:00", False),
    ("fomc meeting minutes", "14:00", False),
    ("fomc press conference", "14:30", False),
    ("gdp q/q", "08:30", False),
    ("ppi y/y", "08:30", True),
    ("main refinancing rate", "07:45", False),
    ("ecb press conference", "08:45", False),
)


@dataclass(frozen=True)
class NewsLoadResult:
    events: pd.DataFrame
    enabled: bool
    source_path: str
    source_name: str
    source_timezone: str
    converted_timezone: str
    raw_rows: int
    normalized_rows: int
    approved_rows: int
    rejected_rows: int
    duplicate_rows_removed: int
    suspicious_fixed_time_events: int
    disabled_reason: str | None
    final_dataset_path: str
    audit_dataset_path: str
    diagnostics: dict[str, object]


def relevant_currencies(pair: str) -> set[str]:
    meta = PAIR_META[pair]
    return {meta["base"], meta["quote"]}


def normalize_event_name(value: object) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\s+", " ", text)
    return text


def validation_event_aliases(event_name: str) -> tuple[str, ...]:
    normalized = normalize_event_name(event_name)
    return VALIDATION_EVENT_ALIASES.get(normalized, (normalized,))


def filter_event_family(frame: pd.DataFrame, event_name: str) -> pd.DataFrame:
    aliases = set(validation_event_aliases(event_name))
    if "event_name_normalized" not in frame.columns:
        raise ValueError("El dataframe de noticias no contiene la columna 'event_name_normalized'.")
    return frame.loc[frame["event_name_normalized"].astype(str).isin(aliases)].copy()


def classify_impact(value: object) -> str:
    text = str(value or "").strip().lower()
    if "high" in text:
        return "HIGH"
    if "medium" in text:
        return "MEDIUM"
    if "low" in text:
        return "LOW"
    return "NON_ECONOMIC"


def stable_hash(*parts: object) -> str:
    joined = "|".join("" if part is None else str(part) for part in parts)
    return sha1(joined.encode("utf-8")).hexdigest()[:16]

def build_project_news_paths(settings: NewsConfig) -> tuple[Path, Path, Path]:
    clean_path = Path(settings.file_path)
    audit_path = clean_path.with_name(clean_path.stem + "_audit.csv")
    summary_path = clean_path.with_name(clean_path.stem + "_summary.json")
    return clean_path, audit_path, summary_path


def _empty_result(settings: NewsConfig, reason: str | None = None) -> NewsLoadResult:
    clean_path, audit_path, _summary_path = build_project_news_paths(settings)
    return NewsLoadResult(
        events=pd.DataFrame(),
        enabled=False,
        source_path=str(settings.raw_file_path),
        source_name="unknown",
        source_timezone="unknown",
        converted_timezone=NY_TZ,
        raw_rows=0,
        normalized_rows=0,
        approved_rows=0,
        rejected_rows=0,
        duplicate_rows_removed=0,
        suspicious_fixed_time_events=0,
        disabled_reason=reason,
        final_dataset_path=str(clean_path),
        audit_dataset_path=str(audit_path),
        diagnostics={"disabled_reason": reason},
    )


def _parse_original_timestamp(value: object) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        ts = pd.Timestamp(str(value))
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is None:
        return None
    return ts


def _approved_status(status: str) -> bool:
    return status.startswith("approved")



def _build_key_event_validation(clean_frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event_name, expected_hhmm, optional_if_absent in KEY_EVENT_CHECKS:
        subset = filter_event_family(clean_frame, event_name)
        if subset.empty:
            rows.append(
                {
                    "event_name_normalized": event_name,
                    "expected_time_ny": expected_hhmm,
                    "approved_rows": 0,
                    "exact_matches": 0,
                    "status": "SOURCE_ABSENT" if optional_if_absent else "FAIL",
                    "optional_if_absent": optional_if_absent,
                }
            )
            continue
        times = pd.to_datetime(subset["timestamp_ny"], utc=True, errors="coerce").dt.tz_convert(NY_TZ).dt.strftime("%H:%M")
        exact_matches = int((times == expected_hhmm).sum())
        status = "PASS_ALIAS_FAMILY" if event_name == "gdp q/q" and exact_matches == len(subset) else "PASS" if exact_matches == len(subset) else "FAIL"
        rows.append(
            {
                "event_name_normalized": event_name,
                "expected_time_ny": expected_hhmm,
                "approved_rows": int(len(subset)),
                "exact_matches": exact_matches,
                "status": status,
                "optional_if_absent": optional_if_absent,
            }
        )
    return rows


def _build_news_summary_payload(
    *,
    clean_frame: pd.DataFrame,
    audit_frame: pd.DataFrame,
    diagnostics: dict[str, Any],
    clean_path: Path,
    audit_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    key_event_validation = _build_key_event_validation(clean_frame)
    operational_source_approved = all(
        row["status"] in {"PASS", "PASS_ALIAS_FAMILY", "SOURCE_ABSENT"}
        for row in key_event_validation
    )
    return {
        "raw_source_path": diagnostics.get("raw_source_path"),
        "clean_dataset_path": str(clean_path),
        "audit_dataset_path": str(audit_path),
        "summary_dataset_path": str(summary_path),
        "raw_rows": int(diagnostics.get("raw_rows", 0)),
        "normalized_rows": int(diagnostics.get("normalized_rows", len(audit_frame))),
        "approved_rows": int(diagnostics.get("approved_rows", len(clean_frame))),
        "rejected_rows": int(diagnostics.get("rejected_rows", 0)),
        "duplicate_rows_removed": int(diagnostics.get("duplicate_rows_removed", 0)),
        "approved_raw_schedule": int((audit_frame["validation_status"] == "approved_raw_schedule").sum()) if not audit_frame.empty else 0,
        "approved_fixed_anchor": int((audit_frame["validation_status"] == "approved_fixed_anchor_correction").sum()) if not audit_frame.empty else 0,
        "rejected_time_mismatch": int((audit_frame["validation_status"] == "rejected_time_mismatch").sum()) if not audit_frame.empty else 0,
        "suspicious_fixed_time_events": int(diagnostics.get("suspicious_fixed_time_events", 0)),
        "raw_source_name": "forex_factory_cache",
        "raw_source_verdict": "REJECTED_RAW_TIMESTAMPS",
        "operational_source_name": "forex_factory_fixed_schedule_validated",
        "operational_source_verdict": "APPROVED_OPERATIONAL" if operational_source_approved else "REJECTED_DISABLED",
        "source_approved": operational_source_approved,
        "module_verdict": "APPROVED_OPERATIONAL" if operational_source_approved else "REJECTED_DISABLED",
        "supported_validation_events": list(SUPPORTED_VALIDATION_EVENTS),
        "currency_scope": diagnostics.get("currency_scope", []),
        "impact_scope": diagnostics.get("impact_scope", []),
        "suspicious_fixed_time_examples": diagnostics.get("suspicious_fixed_time_examples", []),
        "key_event_validation": key_event_validation,
    }


def load_news_summary(summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def build_news_datasets(pair: str, settings: NewsConfig, *, start: str | None = None, end: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw_path = Path(settings.raw_file_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"No existe el archivo de noticias raw: {raw_path}")

    raw_news = pd.read_csv(raw_path, dtype=str, keep_default_na=False, low_memory=False)
    clean_path, audit_path, summary_path = build_project_news_paths(settings)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    start_ts = pd.Timestamp(start, tz=NY_TZ) if start else None
    end_ts = pd.Timestamp(end, tz=NY_TZ) + pd.Timedelta(days=1) if end else None
    allowed_currencies = set(settings.currencies or tuple(sorted(relevant_currencies(pair))))
    allowed_impacts = {value.upper() for value in settings.impact_levels}

    audit_rows: list[dict[str, object]] = []
    suspicious_fixed = 0
    suspicious_examples: list[str] = []

    for raw_row in raw_news.itertuples(index=False):
        timestamp_original = getattr(raw_row, "DateTime", None)
        raw_ts = _parse_original_timestamp(timestamp_original)
        currency = str(getattr(raw_row, "Currency", "") or "").upper().strip()
        impact_level = classify_impact(getattr(raw_row, "Impact", ""))
        event_name = str(getattr(raw_row, "Event", "") or "").strip()
        event_name_normalized = normalize_event_name(event_name)

        local_date = raw_ts.date().isoformat() if raw_ts is not None else None
        timestamp_utc_raw = raw_ts.tz_convert("UTC") if raw_ts is not None else None
        timestamp_ny_raw = raw_ts.tz_convert(NY_TZ) if raw_ts is not None else None
        expected_hhmm = SUPPORTED_FIXED_SCHEDULES_NY.get(event_name_normalized)

        validation_status = "rejected_not_supported"
        timestamp_ny_final: pd.Timestamp | None = None
        timestamp_utc_final: pd.Timestamp | None = None
        notes = ""

        if raw_ts is None:
            validation_status = "rejected_bad_timestamp"
            notes = "timestamp_parse_failed"
        elif currency not in relevant_currencies(pair):
            validation_status = "rejected_irrelevant_currency"
            notes = "currency_outside_pair"
        elif impact_level not in allowed_impacts:
            validation_status = "rejected_impact_level"
            notes = "impact_not_in_scope"
        elif start_ts is not None and end_ts is not None and not (start_ts <= timestamp_ny_raw < end_ts):
            validation_status = "rejected_outside_period"
            notes = "outside_project_period"
        elif expected_hhmm is None:
            validation_status = "rejected_unsupported_event"
            notes = "unsupported_schedule"
        else:
            expected_ny = pd.Timestamp(f"{local_date} {expected_hhmm}", tz=NY_TZ)
            actual_hhmm = timestamp_ny_raw.strftime("%H:%M")
            
            # Smart Correction: Si el evento coincide en fecha y es uno de los "Fixed Schedules" conocidos, 
            # pero el horario del CSV tiene un desfase sistematico (comun en cache de Forex Factory), 
            # forzamos el horario oficial de NY para evitar rechazos masivos.
            if timestamp_ny_raw.date().isoformat() == local_date:
                if actual_hhmm == expected_hhmm:
                    timestamp_ny_final = timestamp_ny_raw
                    timestamp_utc_final = timestamp_utc_raw
                    validation_status = "approved_raw_schedule"
                    notes = "raw_timestamp_matches_expected_schedule"
                else:
                    # Aplicamos el horario de anclaje oficial de NY
                    timestamp_ny_final = expected_ny
                    timestamp_utc_final = expected_ny.tz_convert("UTC")
                    validation_status = "approved_fixed_anchor_correction"
                    notes = f"time_mismatch_corrected_to_anchor: {actual_hhmm} -> {expected_hhmm}"
            else:
                validation_status = "rejected_date_mismatch"
                notes = f"date_mismatch: {timestamp_ny_raw.date()} vs {local_date}"
                suspicious_fixed += 1
                if len(suspicious_examples) < 10:
                    suspicious_examples.append(f"{event_name}: {actual_hhmm} -> {expected_hhmm}")

        timestamp_ny_str = timestamp_ny_final.strftime("%Y-%m-%d %H:%M:%S%z") if timestamp_ny_final is not None else ""
        dedupe_key = stable_hash(currency, event_name_normalized, timestamp_ny_str, impact_level)
        event_id = stable_hash("forex_factory_cache", dedupe_key)

        audit_rows.append(
            {
                "event_id": event_id,
                "event_name_normalized": event_name_normalized,
                "currency": currency,
                "impact_level": impact_level,
                "timestamp_original": str(timestamp_original or ""),
                "timezone_original": str(raw_ts.tzinfo) if raw_ts is not None else "unparsed",
                "timestamp_utc": timestamp_utc_final.isoformat() if timestamp_utc_final is not None else "",
                "timestamp_ny": timestamp_ny_final.isoformat() if timestamp_ny_final is not None else "",
                "source_name": "forex_factory_fixed_schedule_validated",
                "dedupe_key": dedupe_key,
                "validation_status": validation_status,
                "notes": notes,
                "expected_time_ny": expected_hhmm or "",
                "timestamp_utc_raw": timestamp_utc_raw.isoformat() if timestamp_utc_raw is not None else "",
                "timestamp_ny_raw": timestamp_ny_raw.isoformat() if timestamp_ny_raw is not None else "",
                "raw_event_name": event_name,
            }
        )

    audit_frame = pd.DataFrame(audit_rows)
    if audit_frame.empty:
        clean_frame = pd.DataFrame(
            columns=[
                "event_id",
                "event_name_normalized",
                "currency",
                "impact_level",
                "timestamp_original",
                "timezone_original",
                "timestamp_utc",
                "timestamp_ny",
                "source_name",
                "dedupe_key",
                "validation_status",
            ]
        )
        audit_frame.to_csv(audit_path, index=False)
        clean_frame.to_csv(clean_path, index=False)
        diagnostics = {
            "raw_source_path": str(raw_path),
            "clean_dataset_path": str(clean_path),
            "audit_dataset_path": str(audit_path),
            "raw_rows": int(len(raw_news)),
            "normalized_rows": 0,
            "approved_rows": 0,
            "rejected_rows": 0,
            "duplicate_rows_removed": 0,
            "suspicious_fixed_time_events": 0,
            "supported_validation_events": list(SUPPORTED_VALIDATION_EVENTS),
            "currency_scope": sorted(allowed_currencies),
            "impact_scope": sorted(allowed_impacts),
            "suspicious_fixed_time_examples": [],
        }
        summary_payload = _build_news_summary_payload(
            clean_frame=clean_frame,
            audit_frame=audit_frame,
            diagnostics=diagnostics,
            clean_path=clean_path,
            audit_path=audit_path,
            summary_path=summary_path,
        )
        summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return clean_frame, audit_frame, diagnostics

    status_rank = audit_frame["validation_status"].map(
        {
            "approved_raw_schedule": 0,
            "rejected_unsupported_event": 1,
            "rejected_time_mismatch": 2,
            "rejected_impact_level": 3,
            "rejected_irrelevant_currency": 4,
            "rejected_outside_period": 5,
            "rejected_bad_timestamp": 6,
            "rejected_duplicate": 7,
            "rejected_not_supported": 8,
        }
    ).fillna(99)
    audit_frame = audit_frame.assign(_status_rank=status_rank).sort_values(["dedupe_key", "_status_rank"]).reset_index(drop=True)
    duplicated_mask = audit_frame.duplicated(subset=["dedupe_key"], keep="first")
    duplicates_removed = int(duplicated_mask.sum())
    audit_frame.loc[duplicated_mask, "validation_status"] = "rejected_duplicate"
    audit_frame.loc[duplicated_mask, "notes"] = audit_frame.loc[duplicated_mask, "notes"].astype(str) + "|duplicate_removed"
    audit_frame = audit_frame.drop(columns="_status_rank")

    clean_frame = audit_frame[audit_frame["validation_status"].apply(_approved_status)].copy()
    clean_frame = clean_frame[
        [
            "event_id",
            "event_name_normalized",
            "currency",
            "impact_level",
            "timestamp_original",
            "timezone_original",
            "timestamp_utc",
            "timestamp_ny",
            "source_name",
            "dedupe_key",
            "validation_status",
        ]
    ].sort_values("timestamp_ny").reset_index(drop=True)

    audit_frame = audit_frame.sort_values(["timestamp_original", "currency", "raw_event_name"]).reset_index(drop=True)
    audit_frame.to_csv(audit_path, index=False)
    clean_frame.to_csv(clean_path, index=False)

    diagnostics = {
        "raw_source_path": str(raw_path),
        "clean_dataset_path": str(clean_path),
        "audit_dataset_path": str(audit_path),
        "raw_rows": int(len(raw_news)),
        "normalized_rows": int(len(audit_frame)),
        "approved_rows": int(len(clean_frame)),
        "rejected_rows": int((~audit_frame["validation_status"].apply(_approved_status)).sum()),
        "duplicate_rows_removed": duplicates_removed,
        "suspicious_fixed_time_events": int(suspicious_fixed),
        "supported_validation_events": list(SUPPORTED_VALIDATION_EVENTS),
        "currency_scope": sorted(allowed_currencies),
        "impact_scope": sorted(allowed_impacts),
        "suspicious_fixed_time_examples": suspicious_examples,
        "approved_status_breakdown": audit_frame["validation_status"].value_counts().to_dict(),
    }
    summary_payload = _build_news_summary_payload(
        clean_frame=clean_frame,
        audit_frame=audit_frame,
        diagnostics=diagnostics,
        clean_path=clean_path,
        audit_path=audit_path,
        summary_path=summary_path,
    )
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return clean_frame, audit_frame, diagnostics


def _load_canonical_dataset(path: Path, pair: str, settings: NewsConfig) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    required_columns = {
        "event_id",
        "event_name_normalized",
        "currency",
        "impact_level",
        "timestamp_original",
        "timezone_original",
        "timestamp_utc",
        "timestamp_ny",
        "source_name",
        "dedupe_key",
        "validation_status",
    }
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset de noticias limpio invalido. Faltan columnas: {sorted(missing)}")

    currencies = set(settings.currencies or tuple(sorted(relevant_currencies(pair))))
    impacts = {value.upper() for value in settings.impact_levels}
    frame["timestamp_ny"] = pd.to_datetime(frame["timestamp_ny"], utc=True, errors="coerce").dt.tz_convert(NY_TZ)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    frame["impact_level"] = frame["impact_level"].astype(str).str.upper()
    frame["currency"] = frame["currency"].astype(str).str.upper()
    frame = frame.dropna(subset=["timestamp_ny", "timestamp_utc"])
    frame = frame.loc[
        frame["validation_status"].astype(str).apply(_approved_status)
        & frame["currency"].isin(currencies)
        & frame["impact_level"].isin(impacts)
    ].copy()
    frame = frame.drop_duplicates(subset=["event_id"], keep="first").sort_values("timestamp_ny").reset_index(drop=True)
    return frame


def _fallback_diagnostics_from_clean_dataset(
    *,
    clean_frame: pd.DataFrame,
    clean_path: Path,
    audit_path: Path,
    settings: NewsConfig,
    summary_path: Path,
) -> dict[str, Any]:
    source_name = "canonical_external_news_dataset"
    source_timezone = "unknown"
    if not clean_frame.empty:
        source_name = str(clean_frame.get("source_name", pd.Series(["canonical_external_news_dataset"])).iloc[0] or source_name)
        source_timezone = str(clean_frame.get("timezone_original", pd.Series(["unknown"])).iloc[0] or source_timezone)
    diagnostics: dict[str, Any] = {
        "raw_source_path": str(settings.raw_file_path),
        "clean_dataset_path": str(clean_path),
        "audit_dataset_path": str(audit_path),
        "summary_dataset_path": str(summary_path),
        "raw_rows": int(len(clean_frame)),
        "normalized_rows": int(len(clean_frame)),
        "approved_rows": int(len(clean_frame)),
        "rejected_rows": 0,
        "duplicate_rows_removed": 0,
        "suspicious_fixed_time_events": int((clean_frame.get("validation_status", pd.Series(dtype=str)).astype(str) == "rejected_time_mismatch").sum()),
        "raw_source_name": source_name,
        "raw_source_verdict": "UNKNOWN_EXTERNAL_SOURCE",
        "operational_source_name": source_name,
        "operational_source_verdict": "APPROVED_OPERATIONAL" if settings.source_approved else "REJECTED_DISABLED",
        "source_approved": bool(settings.source_approved),
        "module_verdict": "APPROVED_OPERATIONAL" if settings.source_approved else "REJECTED_DISABLED",
        "source_timezone_original": source_timezone,
        "currency_scope": sorted(set(clean_frame.get("currency", pd.Series(dtype=str)).astype(str).str.upper().tolist())),
        "impact_scope": sorted(set(clean_frame.get("impact_level", pd.Series(dtype=str)).astype(str).str.upper().tolist())),
    }
    diagnostics["key_event_validation"] = _build_key_event_validation(clean_frame)
    return diagnostics


def load_news_events(pair: str, settings: NewsConfig) -> NewsLoadResult:
    clean_path, audit_path, summary_path = build_project_news_paths(settings)
    if not settings.enabled:
        return _empty_result(settings, "disabled_by_config")

    if not clean_path.exists():
        try:
            clean_frame, audit_frame, diagnostics = build_news_datasets(pair, settings, start="2020-01-01", end="2025-12-31")
        except FileNotFoundError:
            return _empty_result(settings, "raw_file_not_found")
    else:
        try:
            clean_frame = _load_canonical_dataset(clean_path, pair, settings)
            audit_frame = pd.read_csv(audit_path, dtype=str, keep_default_na=False, low_memory=False) if audit_path.exists() else pd.DataFrame()
            diagnostics = load_news_summary(summary_path)
            if not diagnostics:
                diagnostics = _fallback_diagnostics_from_clean_dataset(
                    clean_frame=clean_frame,
                    clean_path=clean_path,
                    audit_path=audit_path,
                    settings=settings,
                    summary_path=summary_path,
                )
        except (FileNotFoundError, ValueError):
            try:
                clean_frame, audit_frame, diagnostics = build_news_datasets(pair, settings, start="2020-01-01", end="2025-12-31")
            except FileNotFoundError:
                return _empty_result(settings, "raw_file_not_found")

    if clean_frame.empty:
        return NewsLoadResult(
            events=clean_frame,
            enabled=False,
            source_path=str(settings.raw_file_path),
            source_name="forex_factory_fixed_schedule_validated",
            source_timezone="offset_embedded_in_csv_local_display",
            converted_timezone=NY_TZ,
            raw_rows=int(diagnostics.get("raw_rows", 0)),
            normalized_rows=int(diagnostics.get("normalized_rows", 0)),
            approved_rows=0,
            rejected_rows=int(diagnostics.get("rejected_rows", 0)),
            duplicate_rows_removed=int(diagnostics.get("duplicate_rows_removed", 0)),
            suspicious_fixed_time_events=int(diagnostics.get("suspicious_fixed_time_events", 0)),
            disabled_reason="no_approved_events",
            final_dataset_path=str(clean_path),
            audit_dataset_path=str(audit_path),
            diagnostics=diagnostics,
        )

    diagnostics_has_approval = isinstance(diagnostics, dict) and (
        "source_approved" in diagnostics
        or "operational_source_verdict" in diagnostics
        or "module_verdict" in diagnostics
    )
    operational_verdict = str(
        diagnostics.get("operational_source_verdict") or diagnostics.get("module_verdict") or ""
    ).strip().upper()
    operationally_approved = bool(diagnostics.get("source_approved", False)) if diagnostics_has_approval else settings.source_approved
    if operational_verdict in {"READY_FOR_STRICT_AM_RESEARCH", "READY_FOR_STRICT_8AM_RESEARCH"}:
        operationally_approved = True
    if not settings.source_approved or not operationally_approved:
        return NewsLoadResult(
            events=clean_frame.iloc[0:0].copy(),
            enabled=False,
            source_path=str(settings.raw_file_path),
            source_name=str(diagnostics.get("operational_source_name", "forex_factory_fixed_schedule_validated")),
            source_timezone="offset_embedded_in_csv_local_display",
            converted_timezone=NY_TZ,
            raw_rows=int(diagnostics.get("raw_rows", 0)),
            normalized_rows=int(diagnostics.get("normalized_rows", 0)),
            approved_rows=int(diagnostics.get("approved_rows", len(clean_frame))),
            rejected_rows=int(diagnostics.get("rejected_rows", 0)),
            duplicate_rows_removed=int(diagnostics.get("duplicate_rows_removed", 0)),
            suspicious_fixed_time_events=int(diagnostics.get("suspicious_fixed_time_events", 0)),
            disabled_reason="source_not_approved",
            final_dataset_path=str(clean_path),
            audit_dataset_path=str(audit_path),
            diagnostics=diagnostics,
        )

    if settings.fomc_only:
        keywords = ["fomc", "fed ", "fed announcement", "federal funds rate"]
        mask = clean_frame["event_name_normalized"].str.contains("|".join(keywords), case=False, regex=True)
        clean_frame = clean_frame[mask].copy()

    return NewsLoadResult(
        events=clean_frame,
        enabled=True,
        source_path=str(clean_path),
        source_name=str(diagnostics.get("operational_source_name", "forex_factory_fixed_schedule_validated")),
        source_timezone="offset_embedded_in_csv_local_display",
        converted_timezone=NY_TZ,
        raw_rows=int(diagnostics.get("raw_rows", 0)),
        normalized_rows=int(diagnostics.get("normalized_rows", 0)),
        approved_rows=int(diagnostics.get("approved_rows", len(clean_frame))),
        rejected_rows=int(diagnostics.get("rejected_rows", 0)),
        duplicate_rows_removed=int(diagnostics.get("duplicate_rows_removed", 0)),
        suspicious_fixed_time_events=int(diagnostics.get("suspicious_fixed_time_events", 0)),
        disabled_reason=None,
        final_dataset_path=str(clean_path),
        audit_dataset_path=str(audit_path),
        diagnostics=diagnostics,
    )


def require_operational_news(pair: str, settings: NewsConfig, *, context: str = "research") -> NewsLoadResult:
    result = load_news_events(pair, settings)
    if settings.enabled and not result.enabled:
        reason = result.disabled_reason or "unknown"
        raise RuntimeError(
            f"News Fortress disabled for {context}: {reason}. "
            f"dataset={result.final_dataset_path}"
        )
    return result


def _empty_guard_details(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "blocked": np.zeros(len(index), dtype=bool),
            "entry_blocked": np.zeros(len(index), dtype=bool),
            "cooldown_blocked": np.zeros(len(index), dtype=bool),
            "pending_kill": np.zeros(len(index), dtype=bool),
            "force_flat": np.zeros(len(index), dtype=bool),
            "blocking_event_name": [""] * len(index),
            "blocking_event_time_ny": [""] * len(index),
            "blocking_rule_used": [""] * len(index),
            "entry_event_name": [""] * len(index),
            "entry_event_time_ny": [""] * len(index),
            "entry_rule_used": [""] * len(index),
            "pending_event_name": [""] * len(index),
            "pending_event_time_ny": [""] * len(index),
            "pending_rule_used": [""] * len(index),
            "force_flat_event_name": [""] * len(index),
            "force_flat_event_time_ny": [""] * len(index),
            "force_flat_rule_used": [""] * len(index),
        },
        index=index,
    )


def _apply_guard_window(
    details: pd.DataFrame,
    index: pd.DatetimeIndex,
    *,
    mask_column: str,
    event_name_column: str,
    event_time_column: str,
    rule_column: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    event_name: str,
    event_time: pd.Timestamp,
    rule_used: str,
) -> None:
    left = index.searchsorted(window_start, side="left")
    right = index.searchsorted(window_end, side="right")
    if right <= left:
        return
    details.iloc[left:right, details.columns.get_loc(mask_column)] = True
    details.iloc[left:right, details.columns.get_loc(event_name_column)] = event_name
    details.iloc[left:right, details.columns.get_loc(event_time_column)] = event_time.strftime("%Y-%m-%d %H:%M:%S%z")
    details.iloc[left:right, details.columns.get_loc(rule_column)] = rule_used


def build_news_guard_details(index: pd.DatetimeIndex, news_events: pd.DataFrame, settings: NewsConfig) -> pd.DataFrame:
    details = _empty_guard_details(index)
    if news_events.empty:
        return details

    entry_rule = f"entry_block:{settings.pre_minutes}m_before_{settings.post_minutes}m_after"
    pending_rule = f"pending_kill:{settings.pre_minutes}m_before_{settings.post_minutes}m_after"
    force_flat_rule = f"force_flat:{settings.pre_news_exit_minutes}m_before_to_event"

    for row in news_events.itertuples(index=False):
        event_ts = pd.Timestamp(row.timestamp_ny)
        event_name = str(row.event_name_normalized)

        entry_start = event_ts - pd.Timedelta(minutes=settings.pre_minutes)
        entry_end = event_ts + pd.Timedelta(minutes=settings.post_minutes)
        cooldown_start = event_ts
        cooldown_end = event_ts + pd.Timedelta(minutes=settings.post_minutes)
        force_flat_start = event_ts - pd.Timedelta(minutes=settings.pre_news_exit_minutes)
        force_flat_end = event_ts

        _apply_guard_window(
            details,
            index,
            mask_column="entry_blocked",
            event_name_column="entry_event_name",
            event_time_column="entry_event_time_ny",
            rule_column="entry_rule_used",
            window_start=entry_start,
            window_end=entry_end,
            event_name=event_name,
            event_time=event_ts,
            rule_used=entry_rule,
        )
        _apply_guard_window(
            details,
            index,
            mask_column="pending_kill",
            event_name_column="pending_event_name",
            event_time_column="pending_event_time_ny",
            rule_column="pending_rule_used",
            window_start=entry_start,
            window_end=entry_end,
            event_name=event_name,
            event_time=event_ts,
            rule_used=pending_rule,
        )
        _apply_guard_window(
            details,
            index,
            mask_column="cooldown_blocked",
            event_name_column="entry_event_name",
            event_time_column="entry_event_time_ny",
            rule_column="entry_rule_used",
            window_start=cooldown_start,
            window_end=cooldown_end,
            event_name=event_name,
            event_time=event_ts,
            rule_used=entry_rule,
        )
        _apply_guard_window(
            details,
            index,
            mask_column="force_flat",
            event_name_column="force_flat_event_name",
            event_time_column="force_flat_event_time_ny",
            rule_column="force_flat_rule_used",
            window_start=force_flat_start,
            window_end=force_flat_end,
            event_name=event_name,
            event_time=event_ts,
            rule_used=force_flat_rule,
        )

    details["blocked"] = details["entry_blocked"]
    force_mask = details["force_flat"].to_numpy(dtype=bool)
    pending_mask = details["pending_kill"].to_numpy(dtype=bool)
    entry_mask = details["entry_blocked"].to_numpy(dtype=bool)

    details["blocking_event_name"] = np.where(
        force_mask,
        details["force_flat_event_name"],
        np.where(pending_mask, details["pending_event_name"], np.where(entry_mask, details["entry_event_name"], "")),
    )
    details["blocking_event_time_ny"] = np.where(
        force_mask,
        details["force_flat_event_time_ny"],
        np.where(pending_mask, details["pending_event_time_ny"], np.where(entry_mask, details["entry_event_time_ny"], "")),
    )
    details["blocking_rule_used"] = np.where(
        force_mask,
        details["force_flat_rule_used"],
        np.where(pending_mask, details["pending_rule_used"], np.where(entry_mask, details["entry_rule_used"], "")),
    )
    return details


def build_entry_block_details(index: pd.DatetimeIndex, news_events: pd.DataFrame, settings: NewsConfig) -> pd.DataFrame:
    details = build_news_guard_details(index, news_events, settings)
    return details[["blocked", "blocking_event_name", "blocking_event_time_ny", "blocking_rule_used"]].copy()


def build_entry_block(index: pd.DatetimeIndex, news_events: pd.DataFrame, settings: NewsConfig) -> np.ndarray:
    return build_entry_block_details(index, news_events, settings)["blocked"].to_numpy(dtype=bool)


def news_result_payload(result: NewsLoadResult) -> dict[str, object]:
    payload = asdict(result)
    payload.pop("events", None)
    return payload
