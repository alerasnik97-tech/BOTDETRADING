from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
NY_TZ = ZoneInfo("America/New_York")

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]
NEWS_REQUIRED_COLUMNS = [
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
    "news_source_tier",
    "source_url",
    "notes",
    "source_priority",
    "selection_status",
]

CANONICAL_PRICE_FILES = {
    "H1": ROOT / "data_candidates_2022_2025" / "prepared" / "EURUSD_H1.csv",
    "M5": ROOT / "data_candidates_2022_2025" / "prepared" / "EURUSD_M5.csv",
}
CANONICAL_NEWS_FILE = ROOT / "data" / "news_eurusd_am_fortress_v3.csv"

PIPELINE_ROOT = ROOT / "data" / "coverage_pipeline"
INTAKE_ROOT = PIPELINE_ROOT / "intake"
STAGING_ROOT = PIPELINE_ROOT / "staging"
BACKUP_ROOT = PIPELINE_ROOT / "backups"
RESULTS_DIR = ROOT / "results" / "data_coverage_pipeline"
STATUS_PATH = ROOT / "DATA_COVERAGE_PIPELINE_STATUS.json"


@dataclass(frozen=True)
class DatasetSpec:
    kind: str
    canonical_path: Path
    intake_path: Path
    required_close_hour: int | None = None
    required_close_minute: int | None = None


SPECS = {
    "H1": DatasetSpec(
        kind="H1",
        canonical_path=CANONICAL_PRICE_FILES["H1"],
        intake_path=INTAKE_ROOT / "EURUSD_H1.csv",
        required_close_hour=20,
        required_close_minute=0,
    ),
    "M5": DatasetSpec(
        kind="M5",
        canonical_path=CANONICAL_PRICE_FILES["M5"],
        intake_path=INTAKE_ROOT / "EURUSD_M5.csv",
        required_close_hour=19,
        required_close_minute=55,
    ),
    "NEWS": DatasetSpec(
        kind="NEWS",
        canonical_path=CANONICAL_NEWS_FILE,
        intake_path=INTAKE_ROOT / "news_eurusd_am_fortress_v3.csv",
    ),
}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_within_project(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    root = ROOT.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"FAIL-CLOSED: path fuera del proyecto: {resolved}")
    return resolved


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return ensure_within_project(path)


def ensure_pipeline_dirs() -> None:
    for path in (INTAKE_ROOT, STAGING_ROOT, BACKUP_ROOT, RESULTS_DIR):
        ensure_within_project(path)
        path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def parse_timestamp(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp sin timezone explicito")
    return parsed.astimezone(NY_TZ)


def target_close(target_date: str, hour: int, minute: int) -> datetime:
    return datetime.fromisoformat(f"{target_date}T{hour:02d}:{minute:02d}:00").replace(tzinfo=NY_TZ)


def read_csv_dicts(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return header, rows


def timestamp_column(header: list[str]) -> str | None:
    if "timestamp" in header:
        return "timestamp"
    if header and header[0] == "":
        return ""
    return None


def validate_price_csv(path: Path, kind: str, *, target_date: str | None = None) -> dict[str, Any]:
    path = project_path(path)
    result: dict[str, Any] = {
        "kind": kind,
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "schema_ok": False,
        "timezone_ok": False,
        "duplicates": 0,
        "sorted": False,
        "invalid_timestamp_rows": 0,
        "invalid_ohlc_rows": 0,
        "timeframe_alignment_errors": 0,
        "rows": 0,
        "first_timestamp_ny": "",
        "last_timestamp_ny": "",
        "target_date": target_date,
        "target_rows": 0,
        "target_last_ny": "",
        "target_coverage_ok": False,
        "blocking_reasons": [],
    }
    if not path.exists():
        result["blocking_reasons"].append(f"{kind}_FILE_MISSING")
        return result

    header, rows = read_csv_dicts(path)
    ts_col = timestamp_column(header)
    missing = [column for column in PRICE_COLUMNS if column not in header]
    if ts_col is None:
        result["blocking_reasons"].append("TIMESTAMP_COLUMN_MISSING")
    if missing:
        result["blocking_reasons"].append(f"PRICE_COLUMNS_MISSING:{','.join(missing)}")
    if ts_col is None or missing:
        return result
    result["schema_ok"] = True
    result["rows"] = len(rows)

    timestamps: list[datetime] = []
    seen: set[datetime] = set()
    duplicate_count = 0
    invalid_ts = 0
    invalid_ohlc = 0
    alignment_errors = 0
    for row in rows:
        try:
            ts = parse_timestamp(row.get(ts_col, ""))
        except Exception:
            invalid_ts += 1
            continue
        if ts in seen:
            duplicate_count += 1
        seen.add(ts)
        timestamps.append(ts)

        if kind == "H1" and ts.minute != 0:
            alignment_errors += 1
        if kind == "M5" and ts.minute % 5 != 0:
            alignment_errors += 1

        try:
            open_ = float(row["open"])
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            float(row["volume"])
        except Exception:
            invalid_ohlc += 1
            continue
        if high < max(open_, low, close) or low > min(open_, high, close):
            invalid_ohlc += 1

    result["invalid_timestamp_rows"] = invalid_ts
    result["invalid_ohlc_rows"] = invalid_ohlc
    result["duplicates"] = duplicate_count
    result["timeframe_alignment_errors"] = alignment_errors
    result["timezone_ok"] = invalid_ts == 0
    result["sorted"] = timestamps == sorted(timestamps)
    if timestamps:
        result["first_timestamp_ny"] = timestamps[0].isoformat()
        result["last_timestamp_ny"] = timestamps[-1].isoformat()

    if invalid_ts:
        result["blocking_reasons"].append("TIMESTAMP_INVALID_OR_NAIVE")
    if duplicate_count:
        result["blocking_reasons"].append("DUPLICATE_TIMESTAMPS")
    if not result["sorted"]:
        result["blocking_reasons"].append("TIMESTAMPS_NOT_SORTED")
    if invalid_ohlc:
        result["blocking_reasons"].append("INVALID_OHLC")
    if alignment_errors:
        result["blocking_reasons"].append("TIMEFRAME_ALIGNMENT_ERROR")

    if target_date and timestamps:
        target_values = [ts for ts in timestamps if ts.date().isoformat() == target_date]
        result["target_rows"] = len(target_values)
        if target_values:
            result["target_last_ny"] = max(target_values).isoformat()
            spec = SPECS[kind]
            if spec.required_close_hour is not None and spec.required_close_minute is not None:
                result["target_coverage_ok"] = max(target_values) >= target_close(
                    target_date,
                    spec.required_close_hour,
                    spec.required_close_minute,
                )
        if not result["target_rows"]:
            result["blocking_reasons"].append(f"{kind}_NO_TARGET_ROWS")
        elif not result["target_coverage_ok"]:
            result["blocking_reasons"].append(f"{kind}_TARGET_INCOMPLETE")

    return result


def validate_news_csv(path: Path, *, target_date: str | None = None, exact_header: list[str] | None = None) -> dict[str, Any]:
    path = project_path(path)
    result: dict[str, Any] = {
        "kind": "NEWS",
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "schema_ok": False,
        "timezone_ok": False,
        "duplicates": 0,
        "invalid_timestamp_rows": 0,
        "non_approved_rows": 0,
        "rows": 0,
        "first_date_ny": "",
        "last_date_ny": "",
        "target_date": target_date,
        "target_in_horizon": False,
        "blocking_reasons": [],
    }
    if not path.exists():
        result["blocking_reasons"].append("FILE_MISSING")
        return result

    header, rows = read_csv_dicts(path)
    missing = [column for column in NEWS_REQUIRED_COLUMNS if column not in header]
    if exact_header is not None and header != exact_header:
        result["blocking_reasons"].append("NEWS_HEADER_NOT_CANONICAL")
    if missing:
        result["blocking_reasons"].append(f"NEWS_COLUMNS_MISSING:{','.join(missing)}")
    if missing or "NEWS_HEADER_NOT_CANONICAL" in result["blocking_reasons"]:
        return result
    result["schema_ok"] = True
    result["rows"] = len(rows)

    timestamps: list[datetime] = []
    invalid_ts = 0
    non_approved = 0
    dedupe_seen: set[str] = set()
    duplicate_dedupe = 0
    for row in rows:
        try:
            ts_ny = parse_timestamp(row.get("timestamp_ny", ""))
            parse_timestamp(row.get("timestamp_utc", ""))
            timestamps.append(ts_ny)
        except Exception:
            invalid_ts += 1
        if not str(row.get("validation_status", "")).startswith("approved"):
            non_approved += 1
        dedupe_key = str(row.get("dedupe_key", "")).strip()
        if dedupe_key:
            if dedupe_key in dedupe_seen:
                duplicate_dedupe += 1
            dedupe_seen.add(dedupe_key)

    result["invalid_timestamp_rows"] = invalid_ts
    result["non_approved_rows"] = non_approved
    result["duplicates"] = duplicate_dedupe
    result["timezone_ok"] = invalid_ts == 0
    if timestamps:
        result["first_date_ny"] = min(timestamps).date().isoformat()
        result["last_date_ny"] = max(timestamps).date().isoformat()
        if target_date:
            result["target_in_horizon"] = result["first_date_ny"] <= target_date <= result["last_date_ny"]

    if invalid_ts:
        result["blocking_reasons"].append("NEWS_TIMESTAMP_INVALID_OR_NAIVE")
    if non_approved:
        result["blocking_reasons"].append("NEWS_NON_APPROVED_ROWS")
    if duplicate_dedupe:
        result["blocking_reasons"].append("NEWS_DUPLICATE_DEDUPE_KEY")
    if target_date and not result["target_in_horizon"]:
        result["blocking_reasons"].append("NEWS_HORIZON_INSUFFICIENT")
    return result


def coverage_report(
    *,
    target_date: str | None,
    prepared_root: Path | None = None,
    news_path: Path | None = None,
    validator_integration: bool = True,
    enforce_rerun_check: bool = True,
) -> dict[str, Any]:
    ensure_pipeline_dirs()
    prepared = project_path(prepared_root) if prepared_root else None
    h1_path = prepared / "EURUSD_H1.csv" if prepared else CANONICAL_PRICE_FILES["H1"]
    m5_path = prepared / "EURUSD_M5.csv" if prepared else CANONICAL_PRICE_FILES["M5"]
    news = project_path(news_path) if news_path else CANONICAL_NEWS_FILE

    h1 = validate_price_csv(h1_path, "H1", target_date=target_date)
    m5 = validate_price_csv(m5_path, "M5", target_date=target_date)
    news_status = validate_news_csv(news, target_date=target_date)
    coverage_blockers = (
        list(h1["blocking_reasons"])
        + list(m5["blocking_reasons"])
        + list(news_status["blocking_reasons"])
    )
    coverage_ready = not coverage_blockers

    validator_report: dict[str, Any] | None = None
    validator_decision = "SKIPPED"
    validator_blockers: list[str] = []
    if validator_integration and prepared_root is None and news_path is None:
        try:
            from validate_scbi_phase1_baseline import build_validation_report

            validator_report = build_validation_report(
                readiness_date=target_date,
                enforce_rerun_check=enforce_rerun_check,
            )
            validator_decision = str(validator_report["decision"])
            validator_blockers = [str(item["code"]) for item in validator_report["blocking_issues"]]
        except Exception as exc:
            validator_decision = "ERROR"
            validator_blockers = [f"VALIDATOR_INTEGRATION_ERROR:{exc}"]

    daily_operable = coverage_ready and validator_decision in {"PASS", "SKIPPED"}
    decision = "PASS" if daily_operable else "BLOCK"
    taxonomy_outcome = "COVERAGE_STATUS_CLEAR"
    if coverage_ready and target_date and news_status.get("target_in_horizon"):
        taxonomy_outcome = "NEWS_HORIZON_SUFFICIENT"
    if coverage_ready and validator_decision == "PASS":
        taxonomy_outcome = "DATA_REFRESH_CANONICAL"
    if coverage_blockers:
        taxonomy_outcome = "COVERAGE_GAP_REAL"
    if validator_decision == "ERROR":
        taxonomy_outcome = "PIPELINE_NOT_READY"

    report = {
        "generated_at_utc": now_utc_iso(),
        "program": "CANONICAL_DATA_COVERAGE_REFRESH_AND_PROMOTION_PIPELINE",
        "target_date": target_date,
        "decision": decision,
        "taxonomy_outcome": taxonomy_outcome,
        "coverage_ready": coverage_ready,
        "daily_operable": daily_operable,
        "coverage_blockers": coverage_blockers,
        "validator_integration": {
            "enabled": validator_integration and prepared_root is None and news_path is None,
            "decision": validator_decision,
            "blockers": validator_blockers,
            "enforce_rerun_check": enforce_rerun_check,
        },
        "datasets": {
            "H1": h1,
            "M5": m5,
            "NEWS": news_status,
        },
        "intake_paths": {
            "H1": str(SPECS["H1"].intake_path.relative_to(ROOT)),
            "M5": str(SPECS["M5"].intake_path.relative_to(ROOT)),
            "NEWS": str(SPECS["NEWS"].intake_path.relative_to(ROOT)),
        },
    }
    return report


def write_status(report: dict[str, Any], path: Path = STATUS_PATH) -> None:
    path = project_path(path)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")


def canonical_header(path: Path) -> list[str]:
    header, _ = read_csv_dicts(project_path(path))
    return header


def copy_to_stage(source: Path, stage_kind: str, run_id: str) -> Path:
    source = project_path(source)
    target_dir = STAGING_ROOT / run_id / stage_kind
    ensure_within_project(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    shutil.copy2(source, target)
    return target


def _rows_after_last_price(intake: Path, canonical: Path, kind: str) -> tuple[list[str], list[dict[str, str]], list[datetime], dict[str, Any]]:
    canonical_status = validate_price_csv(canonical, kind)
    intake_status = validate_price_csv(intake, kind)
    if canonical_status["blocking_reasons"]:
        return [], [], [], {"status": "BLOCK", "reason": "CANONICAL_INVALID", "details": canonical_status}
    if intake_status["blocking_reasons"]:
        return [], [], [], {"status": "BLOCK", "reason": "INTAKE_INVALID", "details": intake_status}

    canonical_last = parse_timestamp(str(canonical_status["last_timestamp_ny"]))
    header, rows = read_csv_dicts(project_path(intake))
    ts_col = timestamp_column(header)
    parsed = [parse_timestamp(row.get(ts_col or "", "")) for row in rows]
    if any(ts <= canonical_last for ts in parsed):
        return header, rows, parsed, {"status": "STAGING", "reason": "INTAKE_OVERLAPS_CANONICAL"}
    return header, rows, parsed, {"status": "PROMOTABLE", "reason": "STRICTLY_APPEND_ONLY"}


def _rows_after_last_news(intake: Path, canonical: Path) -> tuple[list[str], list[dict[str, str]], list[datetime], dict[str, Any]]:
    header = canonical_header(canonical)
    canonical_status = validate_news_csv(canonical)
    intake_status = validate_news_csv(intake, exact_header=header)
    if canonical_status["blocking_reasons"]:
        return [], [], [], {"status": "BLOCK", "reason": "CANONICAL_INVALID", "details": canonical_status}
    if intake_status["blocking_reasons"]:
        return [], [], [], {"status": "BLOCK", "reason": "INTAKE_INVALID", "details": intake_status}

    canonical_last = datetime.fromisoformat(str(canonical_status["last_date_ny"]) + "T23:59:59").replace(tzinfo=NY_TZ)
    _, rows = read_csv_dicts(project_path(intake))
    parsed = [parse_timestamp(row["timestamp_ny"]) for row in rows]
    if any(ts <= canonical_last for ts in parsed):
        return header, rows, parsed, {"status": "STAGING", "reason": "INTAKE_OVERLAPS_CANONICAL"}
    return header, rows, parsed, {"status": "PROMOTABLE", "reason": "STRICTLY_APPEND_ONLY"}


def append_rows_preserving_prefix(
    *,
    canonical_path: Path,
    canonical_header_fields: list[str],
    intake_rows: list[dict[str, str]],
    intake_timestamp_column: str | None = None,
) -> int:
    canonical_path = project_path(canonical_path)
    rendered = StringIO()
    writer = csv.writer(rendered, lineterminator="\n")
    count = 0
    for row in intake_rows:
        values: list[str] = []
        for field in canonical_header_fields:
            if field == "" and intake_timestamp_column:
                values.append(row.get(intake_timestamp_column, ""))
            else:
                values.append(row.get(field, ""))
        writer.writerow(values)
        count += 1

    existing = canonical_path.read_bytes()
    prefix = b"" if existing.endswith((b"\n", b"\r\n")) else b"\n"
    with canonical_path.open("ab") as handle:
        handle.write(prefix + rendered.getvalue().encode("utf-8"))
    return count


def promote_dataset(
    *,
    kind: str,
    intake_path: Path | None = None,
    canonical_path: Path | None = None,
    promote: bool = False,
    target_date: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    ensure_pipeline_dirs()
    kind = kind.upper()
    if kind not in SPECS:
        raise ValueError(f"Dataset no soportado: {kind}")
    spec = SPECS[kind]
    intake = project_path(intake_path or spec.intake_path)
    canonical = project_path(canonical_path or spec.canonical_path)
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    result: dict[str, Any] = {
        "kind": kind,
        "intake_path": str(intake.relative_to(ROOT)),
        "canonical_path": str(canonical.relative_to(ROOT)),
        "promote_requested": promote,
        "status": "BLOCK",
        "reason": "",
        "rows_appended": 0,
        "backup_path": "",
        "staging_path": "",
        "sha256_before": sha256_file(canonical) if canonical.exists() else "",
        "sha256_after": "",
    }
    if not intake.exists():
        result["reason"] = "INTAKE_FILE_MISSING"
        return result

    if kind in {"H1", "M5"}:
        header, rows, _, gate = _rows_after_last_price(intake, canonical, kind)
        canonical_fields = canonical_header(canonical)
        ts_col = timestamp_column(header)
    else:
        header, rows, _, gate = _rows_after_last_news(intake, canonical)
        canonical_fields = canonical_header(canonical)
        ts_col = None

    result["status"] = gate["status"]
    result["reason"] = gate["reason"]
    if gate["status"] == "BLOCK":
        result["details"] = gate.get("details", {})
        return result
    if gate["status"] == "STAGING":
        result["staging_path"] = str(copy_to_stage(intake, "needs_manual_review", run_id).relative_to(ROOT))
        result["sha256_after"] = result["sha256_before"]
        return result
    if not promote:
        result["status"] = "PROMOTABLE_DRY_RUN"
        result["reason"] = "PROMOTION_NOT_REQUESTED"
        result["rows_appended"] = len(rows)
        result["sha256_after"] = result["sha256_before"]
        return result

    backup_dir = BACKUP_ROOT / run_id
    ensure_within_project(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / canonical.name
    shutil.copy2(canonical, backup_path)
    result["backup_path"] = str(backup_path.relative_to(ROOT))
    result["staging_path"] = str(copy_to_stage(intake, "promoted", run_id).relative_to(ROOT))
    result["rows_appended"] = append_rows_preserving_prefix(
        canonical_path=canonical,
        canonical_header_fields=canonical_fields,
        intake_rows=rows,
        intake_timestamp_column=ts_col,
    )

    if kind in {"H1", "M5"}:
        post_status = validate_price_csv(canonical, kind, target_date=target_date)
    else:
        post_status = validate_news_csv(canonical, target_date=target_date)
    if post_status["blocking_reasons"]:
        result["status"] = "BLOCK"
        result["reason"] = "POST_PROMOTION_VALIDATION_FAILED"
        result["post_validation"] = post_status
    else:
        result["status"] = "PROMOTED"
        result["reason"] = "APPEND_ONLY_PROMOTION_CONFIRMED"
        result["post_validation"] = post_status
    result["sha256_after"] = sha256_file(canonical)
    return result


def write_pipeline_heartbeat(path: Path = ROOT / "DATA_COVERAGE_PIPELINE_HEARTBEAT.md") -> None:
    report = coverage_report(target_date=None, validator_integration=False)
    h1 = report["datasets"]["H1"]
    m5 = report["datasets"]["M5"]
    news = report["datasets"]["NEWS"]
    text = (
        "# Data Coverage Pipeline Heartbeat\n\n"
        f"Updated: `{report['generated_at_utc']}`\n\n"
        "## Canonical Coverage\n\n"
        f"- H1 last NY: `{h1['last_timestamp_ny']}`\n"
        f"- M5 last NY: `{m5['last_timestamp_ny']}`\n"
        f"- News horizon NY: `{news['first_date_ny']} -> {news['last_date_ny']}`\n\n"
        "## Intake Paths\n\n"
        f"- H1: `{SPECS['H1'].intake_path.relative_to(ROOT)}`\n"
        f"- M5: `{SPECS['M5'].intake_path.relative_to(ROOT)}`\n"
        f"- News: `{SPECS['NEWS'].intake_path.relative_to(ROOT)}`\n"
    )
    project_path(path).write_text(text, encoding="utf-8")
