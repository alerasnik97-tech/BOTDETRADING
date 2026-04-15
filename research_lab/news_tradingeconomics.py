import argparse
import json
import zoneinfo
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from research_lab.config import NY_TZ, NewsConfig
from research_lab.news_filter import (
    SUPPORTED_FIXED_SCHEDULES_NY,
    SUPPORTED_VALIDATION_EVENTS,
    _build_key_event_validation,
    normalize_event_name,
    stable_hash,
)


CANONICAL_COLUMNS_V2 = [
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
    "actual",
    "forecast",
    "previous",
]

COUNTRY_TO_CURRENCY = {
    "united states": "USD",
    "usa": "USD",
    "us": "USD",
    "euro area": "EUR",
    "euro zone": "EUR",
    "european union": "EUR",
    "germany": "EUR",
    "france": "EUR",
}

TE_EVENT_ALIASES = {
    "non farm payrolls": "non-farm employment change",
    "non-farm employment change": "non-farm employment change",
    "unemployment rate": "unemployment rate",
    "inflation rate yoy": "cpi y/y",
    "inflation rate mom": "cpi m/m",
    "core inflation rate mom": "core cpi m/m",
    "core inflation rate yoy": "core cpi y/y",
    "producer prices change": "ppi y/y",
    "producer prices mom": "ppi m/m",
    "retail sales mom": "retail sales m/m",
    "core retail sales mom": "core retail sales m/m",
    "gdp growth rate qoq adv": "advance gdp q/q",
    "gdp growth rate qoq prelim": "prelim gdp q/q",
    "gdp growth rate qoq final": "final gdp q/q",
    "gdp growth rate qoq": "gdp q/q",
    "ism manufacturing pmi": "ism manufacturing pmi",
    "ism services pmi": "ism services pmi",
    "fed interest rate decision": "federal funds rate",
    "fomc statement": "fomc statement",
    "fomc minutes": "fomc meeting minutes",
    "fomc press conference": "fomc press conference",
    "ecb interest rate decision": "main refinancing rate",
    "ecb press conference": "ecb press conference",
}


@dataclass(frozen=True)
class TradingEconomicsImportResult:
    clean_frame: pd.DataFrame
    audit_frame: pd.DataFrame
    summary: dict[str, Any]


def _load_export(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        frame = pd.DataFrame(payload)
    else:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    if frame.empty:
        return pd.DataFrame(columns=["CalendarId", "Date", "Country", "Category", "Event", "Importance", "Currency", "Source", "SourceURL"])
    return frame


def _infer_currency(row: pd.Series) -> str:
    explicit = str(row.get("Currency", "") or "").upper().strip()
    if explicit in {"USD", "EUR"}:
        return explicit
    country = normalize_event_name(row.get("Country", ""))
    return COUNTRY_TO_CURRENCY.get(country, explicit)


def _normalize_te_event_name(row: pd.Series) -> str:
    candidates = [
        row.get("Event", ""),
        row.get("Category", ""),
        row.get("Ticker", ""),
        row.get("Symbol", ""),
    ]
    for value in candidates:
        normalized = normalize_event_name(value)
        if normalized in TE_EVENT_ALIASES:
            return TE_EVENT_ALIASES[normalized]
    normalized_event = normalize_event_name(row.get("Event", ""))
    normalized_category = normalize_event_name(row.get("Category", ""))
    if normalized_event in SUPPORTED_VALIDATION_EVENTS:
        return normalized_event
    if normalized_category in SUPPORTED_VALIDATION_EVENTS:
        return normalized_category
    return normalized_event or normalized_category


def _map_importance(value: object) -> str:
    text = str(value or "").strip()
    try:
        numeric = int(float(text))
    except ValueError:
        numeric = 0
    if numeric >= 3:
        return "HIGH"
    if numeric == 2:
        return "MEDIUM"
    if numeric == 1:
        return "LOW"
    return "NON_ECONOMIC"


def _parse_te_utc(value: object) -> pd.Timestamp | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        ts = pd.to_datetime(raw, utc=True)
        return ts
    except Exception:
        return None


def import_tradingeconomics_calendar(
    input_path: Path,
    *,
    clean_output_path: Path,
    settings: NewsConfig,
    allowed_currencies: tuple[str, ...] = ("USD", "EUR"),
) -> TradingEconomicsImportResult:
    import json

    raw = _load_export(input_path)
    clean_output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_output_path = clean_output_path.with_name(clean_output_path.stem + "_audit.csv")
    summary_output_path = clean_output_path.with_name(clean_output_path.stem + "_summary.json")

    audit_rows: list[dict[str, Any]] = []
    allowed = {value.upper() for value in allowed_currencies}
    tz_ny = zoneinfo.ZoneInfo(NY_TZ)

    for row in raw.fillna("").astype(str).to_dict(orient="records"):
        series = pd.Series(row)
        title = _normalize_te_event_name(series)
        timestamp_utc = _parse_te_utc(series.get("Date"))
        
        # Ingesta UTC Canónica: Conversión derivada a NY usando zoneinfo
        timestamp_ny = timestamp_utc.astimezone(tz_ny) if timestamp_utc is not None else None
        
        currency = _infer_currency(series)
        importance = _map_importance(series.get("Importance"))
        source = "trading_economics"
        status = "approved"
        notes = ""

        if timestamp_utc is None:
            status = "rejected_bad_timestamp"
            notes = "scheduled_utc_parse_failed"
        elif importance == "NON_ECONOMIC":
            status = "rejected_bad_importance"
            notes = "importance_not_mapped"
        elif currency not in allowed:
            status = "rejected_irrelevant_currency"
            notes = f"currency_{currency}_not_in_scope"

        # Validación estricta de Anchor Time (Sin Autocorrección)
        if status == "approved" and timestamp_ny is not None:
            expected_hhmm = SUPPORTED_FIXED_SCHEDULES_NY.get(title)
            if expected_hhmm:
                actual_hhmm = timestamp_ny.strftime("%H:%M")
                if actual_hhmm != expected_hhmm:
                    status = "rejected_time_mismatch"
                    notes = f"expected_{expected_hhmm}_got_{actual_hhmm}_ny"

        # Generación de event_id determinístico
        calendar_id = str(series.get("CalendarId", "") or "")
        dedupe_key = stable_hash(source, calendar_id, title, currency, timestamp_utc.isoformat() if timestamp_utc else "")
        event_id = stable_hash(source, calendar_id or dedupe_key)

        audit_rows.append(
            {
                "event_id": event_id,
                "source": source,
                "title": title,
                "country": str(series.get("Country", "") or ""),
                "currency": currency,
                "importance": importance,
                "scheduled_at_utc": timestamp_utc.isoformat() if timestamp_utc is not None else "",
                "scheduled_at_ny": timestamp_ny.isoformat() if timestamp_ny is not None else "",
                "timezone_source": "iana_america_new_york",
                "source_approved": settings.source_approved,
                "status": status,
                "actual": str(series.get("Actual", "") or ""),
                "forecast": str(series.get("Forecast", "") or ""),
                "previous": str(series.get("Previous", "") or ""),
                "notes": notes,
                "dedupe_key": dedupe_key,
            }
        )

    audit_frame = pd.DataFrame(audit_rows)
    if audit_frame.empty:
        clean_frame = pd.DataFrame(columns=CANONICAL_COLUMNS_V2)
    else:
        # Lógica de eliminación de duplicados (solo validación técnica)
        audit_frame = audit_frame.sort_values(["dedupe_key", "status"]).reset_index(drop=True)
        duplicated = audit_frame.duplicated(subset=["dedupe_key"], keep="first")
        audit_frame.loc[duplicated & (audit_frame["status"] == "approved"), "status"] = "rejected_duplicate"
        # Distinción explícita: status = resultado técnico; operational_eligible = listo para uso operativo
        audit_frame["operational_eligible"] = (audit_frame["status"] == "approved") & bool(settings.source_approved)
        clean_frame = audit_frame.loc[audit_frame["status"] == "approved", CANONICAL_COLUMNS_V2].copy()

    audit_frame.to_csv(audit_output_path, index=False)
    clean_frame.to_csv(clean_output_path, index=False)

    technical_approved = int((audit_frame["status"] == "approved").sum()) if len(audit_frame) else 0
    operational_eligible_count = (
        int(audit_frame["operational_eligible"].sum()) if len(audit_frame) and "operational_eligible" in audit_frame.columns else 0
    )

    summary = {
        "raw_source_path": str(input_path),
        "clean_dataset_path": str(clean_output_path),
        "audit_dataset_path": str(audit_output_path),
        "summary_dataset_path": str(summary_output_path),
        "raw_rows": int(len(raw)),
        "approved_rows": int(len(clean_frame)),
        "rejected_rows": int(len(audit_frame) - len(clean_frame)),
        "technical_approved_rows": technical_approved,
        "operational_eligible_rows": operational_eligible_count,
        "source_approved": settings.source_approved,
        "utc_canonical": True,
        "status_column_semantics": "technical_validation_only",
        "operational_eligible_rule": "status==approved AND source_approved==True (config)",
        "module_verdict": "V2_CANONICAL_INGESTED" if not clean_frame.empty else "REJECTED_EMPTY",
    }
    summary_output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return TradingEconomicsImportResult(clean_frame=clean_frame, audit_frame=audit_frame, summary=summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Importa una exportacion de Trading Economics al formato canonico de noticias del proyecto.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--currencies", nargs="*", default=["USD", "EUR"])
    parser.add_argument("--approved", action="store_true", help="Marcar la fuente como aprobada en el momento de ingesta.")
    args = parser.parse_args()

    settings = NewsConfig(
        enabled=False,
        file_path=args.output,
        source_approved=args.approved,
        currencies=tuple(args.currencies),
    )

    result = import_tradingeconomics_calendar(
        args.input,
        clean_output_path=args.output,
        settings=settings,
        allowed_currencies=tuple(args.currencies),
    )
    print(json.dumps(result.summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
