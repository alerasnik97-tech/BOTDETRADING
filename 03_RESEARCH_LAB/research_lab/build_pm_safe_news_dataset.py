from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research_lab.config import DEFAULT_NEWS_V2_UTC_FILE, NY_TZ
from research_lab.news_filter import filter_event_family


PM_SAFE_EVENTS: tuple[tuple[str, str], ...] = (
    ("ism manufacturing pmi", "10:00"),
    ("ism services pmi", "10:00"),
    ("fomc statement", "14:00"),
    ("fomc meeting minutes", "14:00"),
    ("federal funds rate", "14:00"),
    ("fomc press conference", "14:30"),
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_VAULT = PROJECT_ROOT / "05_MARKET_DATA_VAULT"
OUTPUT_FILE = DATA_VAULT / "data" / "news_eurusd_pm_research_safe.csv"
OUTPUT_AUDIT_FILE = DATA_VAULT / "data" / "news_eurusd_pm_research_safe_audit.csv"
OUTPUT_SUMMARY_FILE = DATA_VAULT / "data" / "news_eurusd_pm_research_safe_summary.json"


def _load_source_frame() -> pd.DataFrame:
    source = pd.read_csv(DEFAULT_NEWS_V2_UTC_FILE, dtype=str, keep_default_na=False, low_memory=False)
    required_columns = {
        "event_id",
        "event_name_normalized",
        "currency",
        "impact_level",
        "timestamp_utc",
        "timestamp_ny",
        "source_name",
        "dedupe_key",
        "validation_status",
    }
    missing = sorted(required_columns - set(source.columns))
    if missing:
        raise ValueError(f"Dataset canonico de noticias invalido. Faltan columnas: {missing}")
    source["timestamp_ny"] = pd.to_datetime(source["timestamp_ny"], utc=True, errors="raise").dt.tz_convert(NY_TZ)
    source["timestamp_utc"] = pd.to_datetime(source["timestamp_utc"], utc=True, errors="raise")
    source["impact_level"] = source["impact_level"].astype(str).str.upper()
    source["currency"] = source["currency"].astype(str).str.upper()
    source = source.loc[source["validation_status"].astype(str).str.startswith("approved")].copy()
    return source.sort_values("timestamp_ny").reset_index(drop=True)


def build_pm_safe_news_dataset() -> dict[str, object]:
    source = _load_source_frame()
    rows: list[pd.DataFrame] = []
    validation_rows: list[dict[str, object]] = []

    for event_name, expected_hhmm in PM_SAFE_EVENTS:
        subset = filter_event_family(source, event_name)
        subset = subset.loc[(subset["currency"] == "USD") & (subset["impact_level"] == "HIGH")].copy()
        if subset.empty:
            raise RuntimeError(f"No hay filas aprobadas para el evento PM-safe requerido: {event_name}")
        hhmm = subset["timestamp_ny"].dt.strftime("%H:%M")
        exact_matches = int((hhmm == expected_hhmm).sum())
        if exact_matches != len(subset):
            unique_times = sorted(set(hhmm.tolist()))
            raise RuntimeError(
                f"El evento {event_name} no es exacto en NY para el scope PM-safe. "
                f"Esperado={expected_hhmm}, observados={unique_times}"
            )
        rows.append(subset)
        validation_rows.append(
            {
                "event_name_normalized": event_name,
                "expected_time_ny": expected_hhmm,
                "rows": int(len(subset)),
                "exact_matches": exact_matches,
                "status": "PASS",
            }
        )

    pm_safe = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["event_id"]).sort_values("timestamp_ny").reset_index(drop=True)
    output_frame = pm_safe.copy()
    output_frame["timestamp_ny"] = output_frame["timestamp_ny"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    output_frame["timestamp_utc"] = output_frame["timestamp_utc"].dt.strftime("%Y-%m-%d %H:%M:%S%z")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(OUTPUT_FILE, index=False)
    output_frame.to_csv(OUTPUT_AUDIT_FILE, index=False)

    summary = {
        "verdict": "APPROVED_PM_ONLY_NARROW_SCOPE",
        "source_dataset": str(DEFAULT_NEWS_V2_UTC_FILE),
        "derived_dataset": str(OUTPUT_FILE),
        "derived_audit_dataset": str(OUTPUT_AUDIT_FILE),
        "scope": "PM-only strict research",
        "source_approved": True,
        "allowed_event_families": [name for name, _ in PM_SAFE_EVENTS],
        "justification": [
            "Solo incluye familias USD de horario fijo y estable en NY dentro del tramo 10:00-14:30.",
            "Excluye por completo familias 08:30 y eventos EUR con riesgo de desalineacion DST.",
            "No habilita investigacion AM; solo protege investigacion PM bajo condiciones estrechas.",
        ],
        "row_count": int(len(output_frame)),
        "first_timestamp_ny": str(pm_safe["timestamp_ny"].min()) if not pm_safe.empty else "",
        "last_timestamp_ny": str(pm_safe["timestamp_ny"].max()) if not pm_safe.empty else "",
        "exact_time_validation": validation_rows,
    }
    OUTPUT_SUMMARY_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    summary = build_pm_safe_news_dataset()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
