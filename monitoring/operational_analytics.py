"""
Analitica operativa sobre el log diario del sistema automatico.
Lee solo el CSV historico y genera reportes externos sin tocar el core.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Iterable

MONITOR_ROOT = Path(r"C:\Users\alera\Desktop\Bot\operational_logs")
DEFAULT_CSV_PATH = MONITOR_ROOT / "daily_operational_log.csv"
DEFAULT_OUTPUT_DIR = MONITOR_ROOT

REQUIRED_COLUMNS = (
    "date",
    "feeder_status",
    "autopilot_status",
    "promotion_status",
    "chain_status",
    "classification",
    "coverage_ready",
    "daily_operable",
    "blockers",
    "bundle_updated",
)

REAL_ERROR_CLASSIFICATION = "AUTOMATION_BLOCKED_BY_REAL_ERROR"
FAIL_CLOSED_CLASSIFICATION = "FAIL_CLOSED_CORRECT_BEHAVIOR"
CANONICAL_CLASSIFICATION = "DATA_REFRESH_CANONICAL"
FULL_OPERATION_CLASSIFICATION = "DAILY_CHAIN_EXECUTED"
IGNORED_BLOCKERS = {"", "NONE", "MISSING"}
LEGITIMATE_NON_ERROR_CLASSIFICATIONS = {
    CANONICAL_CLASSIFICATION,
    "COVERAGE_RESTORED",
    "BASELINE_PASS_READY_FOR_CHAIN",
    FULL_OPERATION_CLASSIFICATION,
    FAIL_CLOSED_CLASSIFICATION,
}
CLEAR_NON_OPERABLE_CLASSIFICATIONS = {
    REAL_ERROR_CLASSIFICATION,
    FAIL_CLOSED_CLASSIFICATION,
}


@dataclass(frozen=True)
class OperationalRecord:
    date: date
    feeder_status: str
    autopilot_status: str
    promotion_status: str
    chain_status: str
    classification: str
    coverage_ready: bool
    daily_operable: bool
    blockers: str
    bundle_updated: bool
    source_row: int

    @property
    def blocker_tokens(self) -> tuple[str, ...]:
        tokens = []
        for token in self.blockers.split("|"):
            cleaned = token.strip()
            if cleaned and cleaned not in IGNORED_BLOCKERS:
                tokens.append(cleaned)
        return tuple(tokens)

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "feeder_status": self.feeder_status,
            "autopilot_status": self.autopilot_status,
            "promotion_status": self.promotion_status,
            "chain_status": self.chain_status,
            "classification": self.classification,
            "coverage_ready": self.coverage_ready,
            "daily_operable": self.daily_operable,
            "blockers": self.blockers,
            "bundle_updated": self.bundle_updated,
            "source_row": self.source_row,
        }


def parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "y", "si", "sí"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    return default


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def normalize_text(
    raw_row: dict[str, str | None],
    field: str,
    default: str,
    fill_counts: Counter,
) -> str:
    value = str(raw_row.get(field, "") or "").strip()
    if value:
        return value
    fill_counts[field] += 1
    return default


def load_records(csv_path: Path) -> tuple[list[OperationalRecord], dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el CSV operativo: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing_columns:
            raise ValueError(
                "Faltan columnas requeridas en el CSV: "
                + ", ".join(missing_columns)
            )

        fill_counts: Counter[str] = Counter()
        invalid_date_rows: list[dict] = []
        records: list[OperationalRecord] = []

        for row_number, raw_row in enumerate(reader, start=2):
            raw_date = str(raw_row.get("date", "") or "").strip()
            if not raw_date:
                invalid_date_rows.append({"row": row_number, "value": "", "reason": "missing_date"})
                continue

            try:
                parsed_day = parse_date(raw_date)
            except ValueError:
                invalid_date_rows.append(
                    {"row": row_number, "value": raw_date, "reason": "invalid_date_format"}
                )
                continue

            record = OperationalRecord(
                date=parsed_day,
                feeder_status=normalize_text(raw_row, "feeder_status", "MISSING", fill_counts),
                autopilot_status=normalize_text(raw_row, "autopilot_status", "MISSING", fill_counts),
                promotion_status=normalize_text(raw_row, "promotion_status", "MISSING", fill_counts),
                chain_status=normalize_text(raw_row, "chain_status", "MISSING", fill_counts),
                classification=normalize_text(raw_row, "classification", "MISSING", fill_counts),
                coverage_ready=parse_bool(raw_row.get("coverage_ready"), default=False),
                daily_operable=parse_bool(raw_row.get("daily_operable"), default=False),
                blockers=normalize_text(raw_row, "blockers", "NONE", fill_counts),
                bundle_updated=parse_bool(raw_row.get("bundle_updated"), default=False),
                source_row=row_number,
            )
            records.append(record)

    records.sort(key=lambda item: (item.date, item.source_row))
    duplicate_dates = [
        day.isoformat()
        for day, count in Counter(record.date for record in records).items()
        if count > 1
    ]

    cleaning_report = {
        "required_columns": list(REQUIRED_COLUMNS),
        "rows_loaded": len(records) + len(invalid_date_rows),
        "rows_retained": len(records),
        "rows_dropped_invalid_date": len(invalid_date_rows),
        "invalid_date_rows": invalid_date_rows,
        "filled_missing_values": dict(fill_counts),
        "duplicate_dates": sorted(duplicate_dates),
    }
    return records, cleaning_report


def compute_rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(count / total, 4)


def is_operational_success(record: OperationalRecord) -> bool:
    if record.classification == REAL_ERROR_CLASSIFICATION:
        return False
    if record.classification.startswith("UNCLASSIFIED_"):
        return False
    if record.daily_operable:
        return True
    return record.classification in LEGITIMATE_NON_ERROR_CLASSIFICATIONS


def build_metrics(records: list[OperationalRecord]) -> dict:
    total_runs = len(records)
    success_count = sum(1 for record in records if is_operational_success(record))
    error_count = sum(1 for record in records if record.classification == REAL_ERROR_CLASSIFICATION)
    fail_closed_count = sum(1 for record in records if record.classification == FAIL_CLOSED_CLASSIFICATION)
    canonical_overlap_count = sum(
        1 for record in records if record.classification == CANONICAL_CLASSIFICATION
    )
    coverage_ready_count = sum(1 for record in records if record.coverage_ready)
    daily_operable_count = sum(1 for record in records if record.daily_operable)

    return {
        "total_runs": total_runs,
        "unique_dates": len({record.date for record in records}),
        "success_rate_operational": compute_rate(success_count, total_runs),
        "error_rate_real": compute_rate(error_count, total_runs),
        "fail_closed_rate": compute_rate(fail_closed_count, total_runs),
        "canonical_overlap_rate": compute_rate(canonical_overlap_count, total_runs),
        "coverage_ready_rate": compute_rate(coverage_ready_count, total_runs),
        "daily_operable_rate": compute_rate(daily_operable_count, total_runs),
        "counts": {
            "operational_success_days": success_count,
            "real_error_days": error_count,
            "fail_closed_days": fail_closed_count,
            "canonical_overlap_days": canonical_overlap_count,
            "coverage_ready_days": coverage_ready_count,
            "daily_operable_days": daily_operable_count,
        },
        "classification_distribution": dict(
            Counter(record.classification for record in records).most_common()
        ),
        "metric_definitions": {
            "success_rate_operational": (
                "Dias sin error real que terminaron operables o en una proteccion legitima."
            ),
            "error_rate_real": "Dias clasificados como AUTOMATION_BLOCKED_BY_REAL_ERROR.",
            "fail_closed_rate": "Dias clasificados como FAIL_CLOSED_CORRECT_BEHAVIOR.",
            "canonical_overlap_rate": "Dias clasificados como DATA_REFRESH_CANONICAL.",
        },
    }


def calculate_streak(records: list[OperationalRecord], predicate) -> dict:
    longest = 0
    longest_start: date | None = None
    longest_end: date | None = None
    current_run = 0
    current_start: date | None = None

    for record in records:
        if predicate(record):
            if current_run == 0:
                current_start = record.date
            current_run += 1
            if current_run > longest:
                longest = current_run
                longest_start = current_start
                longest_end = record.date
        else:
            current_run = 0
            current_start = None

    trailing_length = 0
    trailing_start: date | None = None
    trailing_end: date | None = None
    for record in reversed(records):
        if predicate(record):
            trailing_length += 1
            trailing_start = record.date
            if trailing_end is None:
                trailing_end = record.date
        else:
            break

    return {
        "longest_length": longest,
        "longest_start": longest_start.isoformat() if longest_start else None,
        "longest_end": longest_end.isoformat() if longest_end else None,
        "current_length": trailing_length,
        "current_start": trailing_start.isoformat() if trailing_start else None,
        "current_end": trailing_end.isoformat() if trailing_end else None,
    }


def build_streaks(records: list[OperationalRecord]) -> dict:
    return {
        "longest_streak_without_real_errors": calculate_streak(
            records,
            lambda record: record.classification != REAL_ERROR_CLASSIFICATION,
        ),
        "current_streak_without_real_errors": calculate_streak(
            records,
            lambda record: record.classification != REAL_ERROR_CLASSIFICATION,
        ),
        "fail_closed_streak": calculate_streak(
            records,
            lambda record: record.classification == FAIL_CLOSED_CLASSIFICATION,
        ),
        "full_operation_streak": calculate_streak(
            records,
            lambda record: record.classification == FULL_OPERATION_CLASSIFICATION,
        ),
    }


def consecutive_runs(records: list[OperationalRecord], predicate) -> list[list[OperationalRecord]]:
    runs: list[list[OperationalRecord]] = []
    current_run: list[OperationalRecord] = []
    for record in records:
        if predicate(record):
            current_run.append(record)
            continue
        if current_run:
            runs.append(current_run)
            current_run = []
    if current_run:
        runs.append(current_run)
    return runs


def detect_repetitive_blockers(records: list[OperationalRecord]) -> list[dict]:
    blocker_dates: defaultdict[str, list[str]] = defaultdict(list)
    blocker_runs: defaultdict[str, list[dict]] = defaultdict(list)
    active_blockers: dict[str, dict] = {}

    for record in records:
        present = set(record.blocker_tokens)
        for blocker in present:
            blocker_dates[blocker].append(record.date.isoformat())
            run = active_blockers.get(blocker)
            if run:
                run["length"] += 1
                run["end"] = record.date.isoformat()
            else:
                active_blockers[blocker] = {
                    "start": record.date.isoformat(),
                    "end": record.date.isoformat(),
                    "length": 1,
                }

        closing_blockers = [blocker for blocker in active_blockers if blocker not in present]
        for blocker in closing_blockers:
            blocker_runs[blocker].append(active_blockers.pop(blocker))

    for blocker, run in active_blockers.items():
        blocker_runs[blocker].append(run)

    repetitive = []
    for blocker, dates in sorted(blocker_dates.items()):
        if len(dates) < 2:
            continue
        longest_run = max(blocker_runs[blocker], key=lambda item: item["length"], default=None)
        repetitive.append(
            {
                "blocker": blocker,
                "occurrences": len(dates),
                "dates": dates,
                "longest_consecutive_run": longest_run,
            }
        )
    return repetitive


def detect_sequence_anomalies(records: list[OperationalRecord], duplicate_dates: Iterable[str]) -> list[dict]:
    anomalies: list[dict] = []
    for duplicate_day in duplicate_dates:
        anomalies.append(
            {
                "type": "duplicate_date",
                "severity": "medium",
                "date": duplicate_day,
                "detail": "Hay mas de un registro para la misma fecha operativa.",
            }
        )

    for record in records:
        if (
            record.classification == FULL_OPERATION_CLASSIFICATION
            and record.chain_status != "SUCCESS"
        ):
            anomalies.append(
                {
                    "type": "inconsistent_chain_success",
                    "severity": "high",
                    "date": record.date.isoformat(),
                    "detail": (
                        "El dia esta marcado como DAILY_CHAIN_EXECUTED "
                        f"pero chain_status={record.chain_status}."
                    ),
                }
            )

        if (
            record.classification == REAL_ERROR_CLASSIFICATION
            and not record.blocker_tokens
        ):
            anomalies.append(
                {
                    "type": "real_error_without_blocker",
                    "severity": "medium",
                    "date": record.date.isoformat(),
                    "detail": "Se marco error real sin blocker explicito en el log.",
                }
            )

        if (
            record.classification in {"COVERAGE_RESTORED", "BASELINE_PASS_READY_FOR_CHAIN"}
            and not record.coverage_ready
        ):
            anomalies.append(
                {
                    "type": "coverage_state_inconsistent",
                    "severity": "medium",
                    "date": record.date.isoformat(),
                    "detail": (
                        f"La clasificacion {record.classification} requiere coverage_ready=True."
                    ),
                }
            )

        if (
            record.daily_operable is False
            and record.classification == FULL_OPERATION_CLASSIFICATION
        ):
            anomalies.append(
                {
                    "type": "operability_state_inconsistent",
                    "severity": "high",
                    "date": record.date.isoformat(),
                    "detail": "El log declara operacion completa pero daily_operable=False.",
                }
            )
    return anomalies


def detect_behavior_changes(records: list[OperationalRecord]) -> list[dict]:
    changes: list[dict] = []
    for previous, current in zip(records, records[1:]):
        field_changes = {}
        for field in (
            "classification",
            "coverage_ready",
            "daily_operable",
            "chain_status",
            "autopilot_status",
            "bundle_updated",
            "blockers",
        ):
            old_value = getattr(previous, field)
            new_value = getattr(current, field)
            if old_value != new_value:
                field_changes[field] = {"from": old_value, "to": new_value}

        if field_changes and (
            "classification" in field_changes or len(field_changes) >= 2
        ):
            changes.append(
                {
                    "date": current.date.isoformat(),
                    "previous_date": previous.date.isoformat(),
                    "changed_fields": field_changes,
                    "change_magnitude": len(field_changes),
                }
            )
    return changes


def build_patterns(records: list[OperationalRecord], cleaning_report: dict) -> dict:
    return {
        "repetitive_blockers": detect_repetitive_blockers(records),
        "anomalous_sequences": detect_sequence_anomalies(
            records,
            cleaning_report.get("duplicate_dates", []),
        ),
        "behavior_changes": detect_behavior_changes(records),
    }


def build_alerts(records: list[OperationalRecord]) -> list[dict]:
    alerts: list[dict] = []

    real_error_runs = consecutive_runs(
        records,
        lambda record: record.classification == REAL_ERROR_CLASSIFICATION,
    )
    for run in real_error_runs:
        if len(run) >= 2:
            alerts.append(
                {
                    "code": "REAL_ERROR_STREAK",
                    "severity": "high",
                    "start_date": run[0].date.isoformat(),
                    "end_date": run[-1].date.isoformat(),
                    "length": len(run),
                    "message": (
                        "Se detectaron 2 o mas AUTOMATION_BLOCKED_BY_REAL_ERROR consecutivos."
                    ),
                }
            )

    for previous, current in zip(records, records[1:]):
        if previous.coverage_ready and not current.coverage_ready:
            alerts.append(
                {
                    "code": "COVERAGE_READY_DROP",
                    "severity": "medium",
                    "start_date": previous.date.isoformat(),
                    "end_date": current.date.isoformat(),
                    "length": 2,
                    "message": (
                        "coverage_ready cayo frente al dia anterior y requiere seguimiento."
                    ),
                }
            )

    stale_bundle_runs = consecutive_runs(
        records,
        lambda record: record.bundle_updated is False,
    )
    for run in stale_bundle_runs:
        if len(run) >= 2:
            alerts.append(
                {
                    "code": "BUNDLE_NOT_UPDATED_STREAK",
                    "severity": "medium",
                    "start_date": run[0].date.isoformat(),
                    "end_date": run[-1].date.isoformat(),
                    "length": len(run),
                    "message": "bundle_updated=false en dias consecutivos.",
                }
            )

    for record in records:
        clear_reason = (
            record.classification in CLEAR_NON_OPERABLE_CLASSIFICATIONS
            or bool(record.blocker_tokens)
        )
        if record.daily_operable is False and not clear_reason:
            alerts.append(
                {
                    "code": "DAILY_OPERABLE_FALSE_UNCLEAR_REASON",
                    "severity": "medium",
                    "start_date": record.date.isoformat(),
                    "end_date": record.date.isoformat(),
                    "length": 1,
                    "message": "daily_operable=false sin blocker ni clasificacion explicita.",
                }
            )
    return alerts


def analyze_log(csv_path: Path) -> dict:
    records, cleaning_report = load_records(csv_path)
    metrics = build_metrics(records)
    streaks = build_streaks(records)
    patterns = build_patterns(records, cleaning_report)
    alerts = build_alerts(records)

    report = {
        "generated_at_utc": (
            datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        ),
        "source_csv": str(csv_path),
        "output_scope": "read_only_input_write_only_reports",
        "records": [record.to_dict() for record in records],
        "cleaning": cleaning_report,
        "metrics": metrics,
        "streaks": streaks,
        "patterns": patterns,
        "alerts": alerts,
    }
    return report


def format_rate(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_console_summary(report: dict) -> str:
    metrics = report["metrics"]
    alerts = report["alerts"]
    streaks = report["streaks"]
    lines = [
        "=== OPERATIONAL ANALYTICS ===",
        f"Fuente: {report['source_csv']}",
        f"Total runs: {metrics['total_runs']}",
        f"Fechas unicas: {metrics['unique_dates']}",
        f"Success rate operational: {format_rate(metrics['success_rate_operational'])}",
        f"Error rate real: {format_rate(metrics['error_rate_real'])}",
        f"Fail-closed rate: {format_rate(metrics['fail_closed_rate'])}",
        f"Coverage ready rate: {format_rate(metrics['coverage_ready_rate'])}",
        f"Daily operable rate: {format_rate(metrics['daily_operable_rate'])}",
        (
            "Racha actual sin errores reales: "
            f"{streaks['current_streak_without_real_errors']['current_length']}"
        ),
        f"Alertas activas: {len(alerts)}",
    ]
    return "\n".join(lines)


def build_text_report(report: dict) -> str:
    metrics = report["metrics"]
    streaks = report["streaks"]
    patterns = report["patterns"]
    alerts = report["alerts"]
    cleaning = report["cleaning"]

    lines = [
        "REPORTE DE ANALITICA OPERATIVA",
        f"Generado UTC: {report['generated_at_utc']}",
        f"Fuente: {report['source_csv']}",
        "",
        "1. CARGA Y LIMPIEZA",
        f"- Rows loaded: {cleaning['rows_loaded']}",
        f"- Rows retained: {cleaning['rows_retained']}",
        f"- Rows dropped invalid date: {cleaning['rows_dropped_invalid_date']}",
        f"- Duplicate dates: {', '.join(cleaning['duplicate_dates']) or 'NONE'}",
        f"- Missing values filled: {json.dumps(cleaning['filled_missing_values'], ensure_ascii=False)}",
        "",
        "2. METRICAS AVANZADAS",
        f"- total_runs: {metrics['total_runs']}",
        f"- unique_dates: {metrics['unique_dates']}",
        f"- success_rate_operational: {format_rate(metrics['success_rate_operational'])}",
        f"- error_rate_real: {format_rate(metrics['error_rate_real'])}",
        f"- fail_closed_rate: {format_rate(metrics['fail_closed_rate'])}",
        f"- canonical_overlap_rate: {format_rate(metrics['canonical_overlap_rate'])}",
        f"- coverage_ready_rate: {format_rate(metrics['coverage_ready_rate'])}",
        f"- daily_operable_rate: {format_rate(metrics['daily_operable_rate'])}",
        "",
        "3. STREAKS",
        (
            "- longest_streak_without_real_errors: "
            f"{streaks['longest_streak_without_real_errors']['longest_length']}"
        ),
        (
            "- current_streak_without_real_errors: "
            f"{streaks['current_streak_without_real_errors']['current_length']}"
        ),
        f"- fail_closed_streak: {streaks['fail_closed_streak']['longest_length']}",
        f"- full_operation_streak: {streaks['full_operation_streak']['longest_length']}",
        "",
        "4. PATRONES",
        (
            "- repetitive_blockers: "
            f"{len(patterns['repetitive_blockers'])}"
        ),
        (
            "- anomalous_sequences: "
            f"{len(patterns['anomalous_sequences'])}"
        ),
        f"- behavior_changes: {len(patterns['behavior_changes'])}",
        "",
        "5. ALERTAS",
    ]

    if alerts:
        for alert in alerts:
            lines.append(
                (
                    f"- [{alert['severity'].upper()}] {alert['code']} "
                    f"{alert['start_date']}->{alert['end_date']} | {alert['message']}"
                )
            )
    else:
        lines.append("- NONE")

    return "\n".join(lines)


def write_reports(report: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "operational_report.json"
    txt_path = output_dir / "operational_report.txt"

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    txt_path.write_text(build_text_report(report), encoding="utf-8")
    return json_path, txt_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analitica operativa sobre daily_operational_log.csv",
    )
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV_PATH),
        help="Ruta al CSV operativo (default: operational_logs/daily_operational_log.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directorio de salida para operational_report.json y operational_report.txt",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)

    try:
        report = analyze_log(csv_path)
        json_path, txt_path = write_reports(report, output_dir)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    print(build_console_summary(report))
    print(f"JSON: {json_path}")
    print(f"TXT: {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
