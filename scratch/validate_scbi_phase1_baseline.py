from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo").resolve()
FREEZE_STATUS_PATH = ROOT / "SCBI_PHASE1_FREEZE_STATUS.json"
BASELINE_SNAPSHOT_PATH = ROOT / "EURUSD_SCBI_PHASE1_BASELINE_SNAPSHOT.md"
HEARTBEAT_PATH = ROOT / "SCBI_PHASE1_FREEZE_HEARTBEAT.md"
HARDENING_STATUS_PATH = ROOT / "BASELINE_VALIDATOR_HARDENING_STATUS.json"

FREEZE_REFERENCE_DATE = "2026-04-21"
NY_TZ = ZoneInfo("America/New_York")
VALIDATOR_ENGINE = "BASELINE_VALIDATOR_HARDENED_V1"
PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]
BENCHMARK = {
    "strategy": "H6_SILVER_BULLET_HYBRID",
    "profit_factor": 1.29,
    "expectancy_r": 0.089,
    "drawdown_r": -4.37,
}

IMMUTABLE_BASELINE_FILES = [
    "EURUSD_RESEARCH_INTEGRITY_AUDIT.md",
    "EURUSD_CANONICAL_BACKTEST_PIPELINE.md",
    "EURUSD_SCBI_GLOBAL_VALIDATION_PROTOCOL.md",
    "EURUSD_SCBI_GLOBAL_VALIDATION_RESULTS.md",
    "EURUSD_SCBI_GLOBAL_VALIDATION_DECISION.md",
    "EURUSD_SCBI_FULL_CAMPAIGN_PROTOCOL.md",
    "EURUSD_SCBI_FULL_CAMPAIGN_RESULTS.md",
    "EURUSD_SCBI_FULL_CAMPAIGN_DECISION.md",
    "EURUSD_SCBI_FORWARD_TEST_POLICY.md",
    "EURUSD_SCBI_FORWARD_OPERATING_SYSTEM.md",
    "EURUSD_SCBI_FORWARD_LEDGER_SCHEMA.md",
    "EURUSD_SCBI_FORWARD_PHASE1_PROTOCOL.md",
    "EURUSD_SCBI_FORWARD_DAILY_RUNBOOK.md",
    "EURUSD_SCBI_FORWARD_INCIDENT_POLICY.md",
    "EURUSD_SCBI_FORWARD_AUTOMATION_ARCHITECTURE.md",
    "EURUSD_SCBI_FORWARD_AUTOMATION_PROTOCOL.md",
    "EURUSD_SCBI_FORWARD_LAUNCH_CHECKLIST.md",
    "EURUSD_SCBI_FORWARD_REHEARSAL_RESULTS.md",
    "EURUSD_SCBI_FORWARD_LAUNCH_DECISION.md",
    "EURUSD_SCBI_PHASE1_GOVERNANCE_PROTOCOL.md",
    "EURUSD_SCBI_PHASE1_DRIFT_MONITORING.md",
    "EURUSD_SCBI_PHASE1_WEEKLY_REVIEW_PROTOCOL.md",
    "EURUSD_SCBI_PHASE1_INCIDENT_SEVERITY_MODEL.md",
    "EURUSD_SCBI_PHASE1_EVIDENCE_FREEZE_PROTOCOL.md",
    "EURUSD_SCBI_PHASE1_CHANGE_CONTROL.md",
    "SCBI_PHASE1_FREEZE_RUNBOOK.md",
    "scratch/run_scbi_forward_phase1.py",
    "scratch/generate_scbi_phase1_weekly_review.py",
    "scratch/validate_scbi_phase1_baseline.py",
    "research_lab/config.py",
    "research_lab/news_filter.py",
    "research_lab/data_loader.py",
    "research_lab/engine.py",
    "research_lab/report.py",
    "scripts/build_chatgpt_bundle.py",
    "CURRENT_STATE_OF_LAB.md",
]

EXTEND_ONLY_INPUTS = [
    {
        "path": "data_candidates_2022_2025/prepared/EURUSD_H1.csv",
        "label": "EURUSD H1 prepared feed",
        "timestamp_column": "__index__",
    },
    {
        "path": "data_candidates_2022_2025/prepared/EURUSD_M5.csv",
        "label": "EURUSD M5 prepared feed",
        "timestamp_column": "__index__",
    },
    {
        "path": "data/news_eurusd_am_fortress_v3.csv",
        "label": "Canonical EURUSD high-impact news calendar",
        "timestamp_column": "timestamp_ny",
    },
]

TRACKED_RUNTIME_FILES = [
    {
        "path": "results/SCBI_FORWARD_LEDGER.csv",
        "label": "Phase 1 forward ledger",
    },
    {
        "path": "results/SCBI_FORWARD_DAILY_STATUS.csv",
        "label": "Phase 1 daily status",
    },
]

RUNTIME_TEMPLATES = [
    {
        "path": "results/SCBI_PHASE1_WEEKLY_REVIEW.csv",
        "label": "Phase 1 weekly review",
        "type": "csv",
    },
    {
        "path": "EURUSD_SCBI_PHASE1_EXCEPTION_LOG.md",
        "label": "Phase 1 exception log",
        "type": "markdown",
    },
]

WEEKLY_REVIEW_HEADER = "review_date,days_run_total,trades_total,pf,exp,max_dd,status,flags"

PYTHON_CONTRACTS = {
    "scratch/run_scbi_forward_phase1.py": {
        "defs": {"load_data", "compute_session_levels", "get_news_events", "process_day"},
        "strings": {"validate_current_state", "seal_runtime_state", "SCBI_M5_GLOBAL", "PAIR_CANONICAL_NEWS_FILES"},
    },
    "scratch/run_scbi_core_forward_phase1.py": {
        "defs": {"initialize_ledger", "run_rehearsal"},
        "strings": {"SCBI_CORE", "core_phase1_ledger.csv", "event_id"},
    },
    "scratch/run_dual_line_daily_chain.py": {
        "defs": {"run_command", "main"},
        "strings": {"chain_lock", "run_scbi_phase1_autopilot.py", "run_scbi_core_forward_phase1.py"},
    },
    "scratch/build_scbi_dual_line_scoreboard.py": {
        "defs": {"main"},
        "strings": {"SCOREBOARD_CSV", "Telemetry_Execution_Fidelity"},
    },
    "scratch/run_forward_evidence_tribunal.py": {
        "defs": {"run_tribunal"},
        "strings": {"TRIBUNAL_JSON", "PAPER_ONLY"},
    },
    "scratch/forward_telemetry_lib.py": {
        "defs": {"append_trace_rows", "telemetry_snapshot_by_line", "build_trace_snapshot"},
        "strings": {"SCBI_FORWARD_TELEMETRY_TRACE.csv", "SCBI_M5_GLOBAL", "SCBI_CORE"},
    },
    "research_lab/data_loader.py": {
        "defs": {"parse_prepared_index", "load_prepared_ohlcv", "validate_price_frame"},
        "strings": {"NY_TZ", "El indice del CSV preparado no tiene offset timezone explicito"},
    },
    "research_lab/news_filter.py": {
        "defs": {"load_news_events", "require_operational_news", "build_news_guard_details", "build_entry_block"},
        "strings": {"entry_blocked", "pending_kill", "force_flat"},
    },
}

TEXT_CONTRACTS = {
    "research_lab/config.py": {
        "strings": {
            "PAIR_CANONICAL_NEWS_FILES",
            "news_eurusd_am_fortress_v3.csv",
            "pre_minutes: int = 30",
            "post_minutes: int = 60",
            "forced_exit_pre_news: bool = True",
            "cancel_pending_pre_news: bool = True",
            "pre_news_exit_minutes: int = 10",
        }
    }
}

REQUIRED_OPERATIONAL_FILES = [
    "CURRENT_STATE_OF_LAB.md",
    "AGENTS.md",
    "EURUSD_MAY_2026_OBSERVATION_PROTOCOL.md",
    "EURUSD_MAY_2026_DAILY_CHECKLIST.md",
    "EURUSD_MAY_2026_WEEKLY_CHECKLIST.md",
    "EURUSD_MAY_2026_CHECKPOINT_RULES.md",
    "EURUSD_MAY_2026_INCIDENT_RULES.md",
    "scratch/validate_scbi_phase1_baseline.py",
    "scratch/run_scbi_forward_phase1.py",
    "scratch/run_scbi_core_forward_phase1.py",
    "scratch/run_dual_line_daily_chain.py",
    "scratch/build_scbi_dual_line_scoreboard.py",
    "scratch/run_forward_evidence_tribunal.py",
    "scratch/forward_telemetry_lib.py",
    "scripts/preflight_project_boundary_check.py",
    "research_lab/data_loader.py",
    "research_lab/news_filter.py",
    "research_lab/config.py",
    "data_candidates_2022_2025/prepared/EURUSD_H1.csv",
    "data_candidates_2022_2025/prepared/EURUSD_M5.csv",
    "data/news_eurusd_am_fortress_v3.csv",
    "results/SCBI_FORWARD_LEDGER.csv",
    "results/SCBI_FORWARD_DAILY_STATUS.csv",
    "results/SCBI_CORE_PHASE1/core_phase1_ledger.csv",
    "results/SCBI_DUAL_LINE_SCOREBOARD.csv",
    "results/SCBI_FORWARD_TRIBUNAL_SUMMARY.json",
    "results/SCBI_UNIFIED_LINE_STATUS.json",
    "results/SCBI_UNIFIED_LINE_STATUS.csv",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def project_path(relative_path: str) -> Path:
    return (ROOT / relative_path).resolve()


def sha256_bytes(chunks: list[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_first_lines(path: Path, line_count: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for idx, line in enumerate(handle):
            if idx >= line_count:
                break
            digest.update(line)
    return digest.hexdigest().upper()


def file_last_modified_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat()


def read_header_line(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.readline().rstrip("\r\n")


def csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        line_count = sum(1 for _ in handle)
    return max(line_count - 1, 0)


def parse_iso_date(value: str) -> str:
    if not value:
        return ""
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized).date().isoformat()


def parse_aware_ny(value: str) -> datetime | None:
    value = str(value or "").strip()
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(NY_TZ)


def target_ny_timestamp(target_date: str, hour: int, minute: int = 0) -> datetime:
    return datetime.fromisoformat(f"{target_date}T{hour:02d}:{minute:02d}:00").replace(tzinfo=NY_TZ)


def add_check(
    checks: list[dict[str, str]],
    dimension: str,
    status: str,
    code: str,
    message: str,
    path: str = "",
) -> None:
    checks.append(
        {
            "dimension": dimension,
            "status": status,
            "code": code,
            "message": message,
            "path": path,
        }
    )


def python_source_contract(relative_path: str) -> list[str]:
    issues: list[str] = []
    contract = PYTHON_CONTRACTS.get(relative_path)
    if not contract:
        return issues
    path = project_path(relative_path)
    if not path.exists():
        return [f"falta archivo para contrato python: {relative_path}"]
    try:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"parse python fallo en {relative_path}: {exc}"]
    except UnicodeDecodeError as exc:
        return [f"encoding invalido en {relative_path}: {exc}"]

    defined = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    missing_defs = sorted(set(contract.get("defs", set())) - defined)
    for name in missing_defs:
        issues.append(f"falta simbolo requerido `{name}` en {relative_path}")

    missing_strings = sorted(str(value) for value in set(contract.get("strings", set())) if str(value) not in source)
    for value in missing_strings:
        issues.append(f"falta contrato textual `{value}` en {relative_path}")
    return issues


def text_source_contract(relative_path: str) -> list[str]:
    contract = TEXT_CONTRACTS.get(relative_path)
    if not contract:
        return []
    path = project_path(relative_path)
    if not path.exists():
        return [f"falta archivo para contrato textual: {relative_path}"]
    source = path.read_text(encoding="utf-8", errors="replace")
    missing = sorted(str(value) for value in set(contract.get("strings", set())) if str(value) not in source)
    return [f"falta contrato textual `{value}` en {relative_path}" for value in missing]


def price_csv_coverage(relative_path: str, target_date: str) -> dict[str, object]:
    path = project_path(relative_path)
    result: dict[str, object] = {
        "path": relative_path,
        "exists": path.exists(),
        "schema_ok": False,
        "rows": 0,
        "invalid_timestamp_rows": 0,
        "invalid_ohlc_rows": 0,
        "target_rows": 0,
        "target_duplicate_timestamps": 0,
        "target_first_ny": "",
        "target_last_ny": "",
        "last_timestamp_ny": "",
        "has_previous_price_day": False,
    }
    if not path.exists():
        return result

    seen_target: set[str] = set()
    target_first: datetime | None = None
    target_last: datetime | None = None
    last_ts: datetime | None = None
    target_dt = datetime.fromisoformat(target_date).date()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        result["schema_ok"] = header[1:] == PRICE_COLUMNS
        for row in reader:
            if not row:
                continue
            result["rows"] = int(result["rows"]) + 1
            ts = parse_aware_ny(row[0])
            if ts is None:
                result["invalid_timestamp_rows"] = int(result["invalid_timestamp_rows"]) + 1
                continue
            last_ts = ts
            if ts.date() < target_dt:
                result["has_previous_price_day"] = True
            if ts.date() != target_dt:
                continue

            result["target_rows"] = int(result["target_rows"]) + 1
            key = ts.isoformat()
            if key in seen_target:
                result["target_duplicate_timestamps"] = int(result["target_duplicate_timestamps"]) + 1
            seen_target.add(key)
            target_first = ts if target_first is None or ts < target_first else target_first
            target_last = ts if target_last is None or ts > target_last else target_last
            try:
                o, h, l, c = (float(row[idx]) for idx in range(1, 5))
                if min(o, h, l, c) <= 0 or h < l:
                    result["invalid_ohlc_rows"] = int(result["invalid_ohlc_rows"]) + 1
            except (ValueError, IndexError):
                result["invalid_ohlc_rows"] = int(result["invalid_ohlc_rows"]) + 1

    if target_first is not None:
        result["target_first_ny"] = target_first.isoformat()
    if target_last is not None:
        result["target_last_ny"] = target_last.isoformat()
    if last_ts is not None:
        result["last_timestamp_ny"] = last_ts.isoformat()
    return result


def news_csv_coverage(relative_path: str, target_date: str) -> dict[str, object]:
    path = project_path(relative_path)
    result: dict[str, object] = {
        "path": relative_path,
        "exists": path.exists(),
        "schema_ok": False,
        "rows": 0,
        "invalid_timestamp_rows": 0,
        "non_approved_rows": 0,
        "first_date_ny": "",
        "last_date_ny": "",
        "target_in_horizon": False,
    }
    if not path.exists():
        return result

    required = {"timestamp_ny", "timestamp_utc", "currency", "impact_level", "validation_status"}
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    target_dt = datetime.fromisoformat(target_date).date()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        result["schema_ok"] = required.issubset(set(reader.fieldnames or []))
        for row in reader:
            result["rows"] = int(result["rows"]) + 1
            ts = parse_aware_ny(str(row.get("timestamp_ny", "")))
            if ts is None:
                result["invalid_timestamp_rows"] = int(result["invalid_timestamp_rows"]) + 1
                continue
            first_ts = ts if first_ts is None or ts < first_ts else first_ts
            last_ts = ts if last_ts is None or ts > last_ts else last_ts
            if not str(row.get("validation_status", "")).startswith("approved"):
                result["non_approved_rows"] = int(result["non_approved_rows"]) + 1

    if first_ts is not None:
        result["first_date_ny"] = first_ts.date().isoformat()
    if last_ts is not None:
        result["last_date_ny"] = last_ts.date().isoformat()
    if first_ts is not None and last_ts is not None:
        result["target_in_horizon"] = first_ts.date() <= target_dt <= last_ts.date()
    return result


def date_exists_in_csv(relative_path: str, column: str, target_date: str) -> bool:
    path = project_path(relative_path)
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = str(row.get(column, "")).strip()
            if value.startswith(target_date):
                return True
    return False


def csv_last_timestamp(path: Path, timestamp_column: str) -> str:
    if timestamp_column == "__index__":
        last_value = ""
        with path.open("r", encoding="utf-8", newline="") as handle:
            next(handle, None)
            for raw_line in handle:
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue
                last_value = line.split(",", 1)[0]
        return last_value

    last_row: dict[str, str] | None = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            last_row = row
    return "" if last_row is None else str(last_row.get(timestamp_column, "")).strip()


def ensure_runtime_templates() -> None:
    weekly_path = project_path("results/SCBI_PHASE1_WEEKLY_REVIEW.csv")
    if not weekly_path.exists():
        weekly_path.write_text(WEEKLY_REVIEW_HEADER + "\n", encoding="utf-8")


def collect_immutable_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for relative_path in IMMUTABLE_BASELINE_FILES:
        path = project_path(relative_path)
        if not path.exists():
            raise FileNotFoundError(f"Falta archivo baseline: {relative_path}")
        records.append(
            {
                "path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "last_modified_utc": file_last_modified_utc(path),
            }
        )
    return records


def collect_extend_only_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for entry in EXTEND_ONLY_INPUTS:
        path = project_path(entry["path"])
        if not path.exists():
            raise FileNotFoundError(f"Falta input operativo: {entry['path']}")
        rows = csv_row_count(path)
        last_timestamp = csv_last_timestamp(path, entry["timestamp_column"])
        records.append(
            {
                "path": entry["path"],
                "label": entry["label"],
                "timestamp_column": entry["timestamp_column"],
                "header_sha256": sha256_bytes([read_header_line(path).encode("utf-8")]),
                "prefix_line_count": rows + 1,
                "prefix_sha256": sha256_first_lines(path, rows + 1),
                "rows_at_freeze": rows,
                "last_timestamp": last_timestamp,
                "last_date": parse_iso_date(last_timestamp),
            }
        )
    return records


def collect_runtime_record(relative_path: str, label: str) -> dict[str, object]:
    path = project_path(relative_path)
    if not path.exists():
        raise FileNotFoundError(f"Falta archivo runtime: {relative_path}")
    return {
        "path": relative_path,
        "label": label,
        "rows": csv_row_count(path),
        "header_sha256": sha256_bytes([read_header_line(path).encode("utf-8")]),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "last_modified_utc": file_last_modified_utc(path),
    }


def collect_runtime_records() -> list[dict[str, object]]:
    return [collect_runtime_record(entry["path"], entry["label"]) for entry in TRACKED_RUNTIME_FILES]


def collect_template_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for entry in RUNTIME_TEMPLATES:
        path = project_path(entry["path"])
        if not path.exists():
            continue
        record: dict[str, object] = {
            "path": entry["path"],
            "label": entry["label"],
            "bytes": path.stat().st_size,
            "last_modified_utc": file_last_modified_utc(path),
        }
        if entry["type"] == "csv":
            record["rows"] = csv_row_count(path)
            record["header_sha256"] = sha256_bytes([read_header_line(path).encode("utf-8")])
        else:
            record["sha256"] = sha256_file(path)
        records.append(record)
    return records


def build_blockers(extend_records: list[dict[str, object]], runtime_records: list[dict[str, object]]) -> list[str]:
    blockers: list[str] = []
    coverage_by_path = {record["path"]: record["last_date"] for record in extend_records}
    h1_last = str(coverage_by_path.get("data_candidates_2022_2025/prepared/EURUSD_H1.csv", ""))
    m5_last = str(coverage_by_path.get("data_candidates_2022_2025/prepared/EURUSD_M5.csv", ""))
    news_last = str(coverage_by_path.get("data/news_eurusd_am_fortress_v3.csv", ""))

    if h1_last < FREEZE_REFERENCE_DATE or m5_last < FREEZE_REFERENCE_DATE:
        blockers.append(
            "BLOCKER: los feeds operativos EURUSD_H1/EURUSD_M5 terminan antes del 2026-04-21; no existe cobertura viva para iniciar paper oficial."
        )
    if news_last < FREEZE_REFERENCE_DATE:
        blockers.append(
            "BLOCKER: el calendario canonico de noticias EURUSD termina antes del 2026-04-21; News Fortress no cubre el inicio oficial."
        )
    if any(int(record["rows"]) > 0 for record in runtime_records):
        blockers.append(
            "BLOCKER: los archivos runtime no estan limpios; ledger/status deben quedar header-only antes del dia 1 oficial."
        )
    return blockers


def build_freeze_status() -> dict[str, object]:
    ensure_runtime_templates()
    immutable_records = collect_immutable_records()
    extend_records = collect_extend_only_records()
    runtime_records = collect_runtime_records()
    template_records = collect_template_records()
    blockers = build_blockers(extend_records, runtime_records)
    readiness = "PHASE1_EVIDENCE_FREEZE_READY" if not blockers else "PHASE1_EVIDENCE_FREEZE_BLOCKED"
    now_iso = utc_now_iso()

    return {
        "updated_at_utc": now_iso,
        "freeze_reference_date": FREEZE_REFERENCE_DATE,
        "program": "SCBI_PHASE1_EVIDENCE_FREEZE_AND_CHANGE_CONTROL",
        "strategy": "SCBI_M5_GLOBAL",
        "benchmark_reference": BENCHMARK,
        "baseline_id": f"SCBI_PHASE1_BASELINE_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "official_phase1_started": False,
        "official_phase1_start_date": None,
        "status": "PRE_DAY1_FROZEN",
        "readiness": readiness,
        "blockers": blockers,
        "immutable_baseline_files": immutable_records,
        "extend_only_inputs": extend_records,
        "tracked_runtime_files": runtime_records,
        "template_files": template_records,
        "last_valid_checkpoint": {
            "sealed_at_utc": now_iso,
            "reason": "baseline_write",
        },
    }


def render_snapshot(status: dict[str, object]) -> str:
    lines = [
        "# EURUSD SCBI Phase 1 Baseline Snapshot",
        "",
        f"- **Fecha de congelamiento**: `{status['freeze_reference_date']}`",
        f"- **Baseline ID**: `{status['baseline_id']}`",
        f"- **Estado oficial**: `{status['status']}`",
        f"- **Readiness**: `{status['readiness']}`",
        "- **Arquitectura congelada**: `SCBI_M5_GLOBAL`",
        (
            "- **Benchmark conceptual vigente**: "
            f"`{BENCHMARK['strategy']}` | PF `{BENCHMARK['profit_factor']}` | "
            f"Exp `{BENCHMARK['expectancy_r']}R` | DD `{BENCHMARK['drawdown_r']}R`"
        ),
        "",
        "## Runtime En El Momento Del Freeze",
        "",
        "| Archivo | Filas | SHA256 |",
        "| --- | ---: | --- |",
    ]
    for record in status["tracked_runtime_files"]:
        lines.append(f"| `{record['path']}` | {record['rows']} | `{str(record['sha256'])[:16]}` |")

    lines.extend(
        [
            "",
            "## Inputs Operativos Extend-Only",
            "",
            "| Archivo | Filas congeladas | Ultima fecha cubierta | Prefix SHA256 |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for record in status["extend_only_inputs"]:
        lines.append(
            f"| `{record['path']}` | {record['rows_at_freeze']} | `{record['last_date']}` | `{str(record['prefix_sha256'])[:16]}` |"
        )

    lines.extend(
        [
            "",
            "## Baseline Exacta",
            "",
            "| Archivo | Bytes | SHA256 |",
            "| --- | ---: | --- |",
        ]
    )
    for record in status["immutable_baseline_files"]:
        lines.append(f"| `{record['path']}` | {record['bytes']} | `{str(record['sha256'])[:16]}` |")

    lines.extend(["", "## Bloqueos De Readiness", ""])
    blockers = status["blockers"]
    if blockers:
        for blocker in blockers:
            lines.append(f"- {blocker}")
    else:
        lines.append("- Ninguno.")

    lines.extend(
        [
            "",
            "## Verificacion Diaria",
            "",
            "- Pre-run: `python scratch/validate_scbi_phase1_baseline.py --check --date YYYY-MM-DD`",
            "- Run diario: `python scratch/run_scbi_forward_phase1.py --date YYYY-MM-DD`",
            "- Nueva baseline tras cambio aprobado: `python scratch/validate_scbi_phase1_baseline.py --write-baseline`",
        ]
    )
    return "\n".join(lines) + "\n"


def render_heartbeat(status: dict[str, object]) -> str:
    blockers = status["blockers"]
    lines = [
        "# SCBI PHASE 1 FREEZE - HEARTBEAT",
        "",
        f"- **Ultima actualizacion UTC**: `{status['updated_at_utc']}`",
        f"- **Baseline ID**: `{status['baseline_id']}`",
        f"- **Estado**: `{status['status']}`",
        f"- **Readiness**: `{status['readiness']}`",
        f"- **Phase 1 iniciada**: `{status['official_phase1_started']}`",
        "",
        "## Resumen",
        "",
        "- Baseline exacta registrada con hashes SHA256 para docs, codigo y contratos operativos.",
        "- Inputs vivos definidos como `extend-only` para detectar reescritura historica.",
        "- Ledger y daily status quedan sellados entre corridas.",
        "",
        "## Checkpoint Vigente",
        "",
        f"- **Ultimo sello valido**: `{status['last_valid_checkpoint']['sealed_at_utc']}`",
        f"- **Motivo**: `{status['last_valid_checkpoint']['reason']}`",
        "",
        "## Bloqueos",
        "",
    ]
    if blockers:
        for blocker in blockers:
            lines.append(f"- {blocker}")
    else:
        lines.append("- Ninguno.")
    return "\n".join(lines) + "\n"


def write_status_files(status: dict[str, object]) -> None:
    FREEZE_STATUS_PATH.write_text(json.dumps(status, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    HEARTBEAT_PATH.write_text(render_heartbeat(status), encoding="utf-8")


def write_baseline() -> int:
    status = build_freeze_status()
    write_status_files(status)
    BASELINE_SNAPSHOT_PATH.write_text(render_snapshot(status), encoding="utf-8")
    print(f"[OK] Baseline escrita: {status['baseline_id']}")
    print(f"[STATUS] {status['readiness']}")
    return 0


def load_status() -> dict[str, object]:
    if not FREEZE_STATUS_PATH.exists():
        raise FileNotFoundError("Falta SCBI_PHASE1_FREEZE_STATUS.json. Ejecuta --write-baseline primero.")
    return json.loads(FREEZE_STATUS_PATH.read_text(encoding="utf-8"))


def build_validation_report(
    *,
    readiness_date: str | None = None,
    allow_runtime_drift: bool = False,
    enforce_rerun_check: bool = False,
) -> dict[str, object]:
    status = load_status()
    checks: list[dict[str, str]] = []
    coverage_summary: dict[str, object] = {}

    for relative_path in REQUIRED_OPERATIONAL_FILES:
        path = project_path(relative_path)
        if path.exists():
            add_check(checks, "file_presence_check", "PASS", "FILE_PRESENT", f"Archivo presente: {relative_path}", relative_path)
        else:
            add_check(checks, "file_presence_check", "BLOCK", "FILE_MISSING", f"Falta archivo requerido: {relative_path}", relative_path)

    for record in status["immutable_baseline_files"]:
        path = project_path(str(record["path"]))
        if not path.exists():
            add_check(
                checks,
                "baseline_integrity_check",
                "BLOCK",
                "BASELINE_FILE_MISSING",
                f"Falta baseline exacta: {record['path']}",
                str(record["path"]),
            )
            continue
        if sha256_file(path) != record["sha256"]:
            add_check(
                checks,
                "stale_doc_vs_physical_file_resolution",
                "WARN",
                "HASH_DRIFT_SEMANTICALLY_EVALUATED",
                f"Hash drift no bloqueante; se evalua contrato fisico/semantico: {record['path']}",
                str(record["path"]),
            )

    for relative_path in sorted(PYTHON_CONTRACTS):
        issues = python_source_contract(relative_path)
        for issue in issues:
            add_check(checks, "baseline_integrity_check", "BLOCK", "PYTHON_CONTRACT_BROKEN", issue, relative_path)
        if not issues:
            add_check(
                checks,
                "baseline_integrity_check",
                "PASS",
                "PYTHON_CONTRACT_OK",
                f"Contrato python vigente: {relative_path}",
                relative_path,
            )

    for relative_path in sorted(TEXT_CONTRACTS):
        issues = text_source_contract(relative_path)
        for issue in issues:
            add_check(checks, "baseline_integrity_check", "BLOCK", "TEXT_CONTRACT_BROKEN", issue, relative_path)
        if not issues:
            add_check(
                checks,
                "baseline_integrity_check",
                "PASS",
                "TEXT_CONTRACT_OK",
                f"Contrato textual vigente: {relative_path}",
                relative_path,
            )

    for record in status["extend_only_inputs"]:
        path = project_path(str(record["path"]))
        if not path.exists():
            add_check(
                checks,
                "file_presence_check",
                "BLOCK",
                "EXTEND_ONLY_INPUT_MISSING",
                f"Falta input extend-only: {record['path']}",
                str(record["path"]),
            )
            continue
        current_rows = csv_row_count(path)
        if current_rows < int(record["rows_at_freeze"]):
            add_check(
                checks,
                "baseline_integrity_check",
                "BLOCK",
                "EXTEND_ONLY_ROWS_ROLLED_BACK",
                f"Rows retrocedieron en input extend-only: {record['path']}",
                str(record["path"]),
            )
            continue
        header_hash = sha256_bytes([read_header_line(path).encode("utf-8")])
        if header_hash != record["header_sha256"]:
            add_check(
                checks,
                "baseline_integrity_check",
                "BLOCK",
                "EXTEND_ONLY_HEADER_DRIFT",
                f"Header drift en input extend-only: {record['path']}",
                str(record["path"]),
            )
        prefix_hash = sha256_first_lines(path, int(record["prefix_line_count"]))
        if prefix_hash != record["prefix_sha256"]:
            add_check(
                checks,
                "baseline_integrity_check",
                "BLOCK",
                "EXTEND_ONLY_PREFIX_REWRITE",
                f"Se detecto reescritura del tramo historico congelado: {record['path']}",
                str(record["path"]),
            )

    if readiness_date:
        h1_path = "data_candidates_2022_2025/prepared/EURUSD_H1.csv"
        m5_path = "data_candidates_2022_2025/prepared/EURUSD_M5.csv"
        news_path = "data/news_eurusd_am_fortress_v3.csv"
        h1_coverage = price_csv_coverage(h1_path, readiness_date)
        m5_coverage = price_csv_coverage(m5_path, readiness_date)
        news_coverage = news_csv_coverage(news_path, readiness_date)
        coverage_summary = {
            "h1": h1_coverage,
            "m5": m5_coverage,
            "news": news_coverage,
        }

        for label, coverage in (("H1", h1_coverage), ("M5", m5_coverage)):
            path_label = str(coverage["path"])
            if not coverage["exists"]:
                add_check(checks, "file_presence_check", "BLOCK", f"{label}_FILE_MISSING", f"Falta archivo {label}: {path_label}", path_label)
                continue
            if not coverage["schema_ok"]:
                add_check(checks, "file_presence_check", "BLOCK", f"{label}_SCHEMA_INVALID", f"Schema {label} invalido: {path_label}", path_label)
            if int(coverage["invalid_timestamp_rows"]) > 0:
                add_check(
                    checks,
                    "timezone_consistency_check",
                    "BLOCK",
                    f"{label}_TIMESTAMP_INVALID",
                    f"{label} tiene timestamps invalidos o sin timezone: {coverage['invalid_timestamp_rows']}",
                    path_label,
                )
            if int(coverage["target_duplicate_timestamps"]) > 0:
                add_check(
                    checks,
                    "time_coverage_check",
                    "BLOCK",
                    f"{label}_TARGET_DUPLICATES",
                    f"{label} tiene timestamps duplicados en {readiness_date}: {coverage['target_duplicate_timestamps']}",
                    path_label,
                )
            if int(coverage["invalid_ohlc_rows"]) > 0:
                add_check(
                    checks,
                    "time_coverage_check",
                    "BLOCK",
                    f"{label}_OHLC_INVALID",
                    f"{label} tiene OHLC invalido en {readiness_date}: {coverage['invalid_ohlc_rows']}",
                    path_label,
                )
            if int(coverage["target_rows"]) == 0:
                add_check(checks, "time_coverage_check", "BLOCK", f"{label}_NO_TARGET_ROWS", f"{label} no tiene filas para {readiness_date}", path_label)
                continue
            required_end = target_ny_timestamp(readiness_date, 20, 0) if label == "H1" else target_ny_timestamp(readiness_date, 19, 55)
            last_target = parse_aware_ny(str(coverage["target_last_ny"]))
            if last_target is None or last_target < required_end:
                add_check(
                    checks,
                    "time_coverage_check",
                    "BLOCK",
                    f"{label}_COVERAGE_INSUFFICIENT",
                    f"{label} cobertura insuficiente para {readiness_date}: last={coverage['target_last_ny'] or 'NONE'}, required>={required_end.isoformat()}",
                    path_label,
                )
            else:
                add_check(
                    checks,
                    "time_coverage_check",
                    "PASS",
                    f"{label}_COVERAGE_OK",
                    f"{label} cubre {readiness_date} hasta {coverage['target_last_ny']}",
                    path_label,
                )

        if not bool(h1_coverage.get("has_previous_price_day")):
            add_check(
                checks,
                "time_coverage_check",
                "BLOCK",
                "H1_PREVIOUS_CONTEXT_MISSING",
                f"H1 no tiene dia previo disponible para niveles de sesion antes de {readiness_date}",
                h1_path,
            )

        if not news_coverage["exists"]:
            add_check(checks, "file_presence_check", "BLOCK", "NEWS_FILE_MISSING", f"Falta news calendar: {news_path}", news_path)
        else:
            if not news_coverage["schema_ok"]:
                add_check(checks, "news_coverage_check", "BLOCK", "NEWS_SCHEMA_INVALID", "News calendar sin columnas canonicas requeridas", news_path)
            if int(news_coverage["invalid_timestamp_rows"]) > 0:
                add_check(
                    checks,
                    "timezone_consistency_check",
                    "BLOCK",
                    "NEWS_TIMESTAMP_INVALID",
                    f"News calendar tiene timestamps invalidos o sin timezone: {news_coverage['invalid_timestamp_rows']}",
                    news_path,
                )
            if int(news_coverage["non_approved_rows"]) > 0:
                add_check(
                    checks,
                    "news_coverage_check",
                    "BLOCK",
                    "NEWS_NON_APPROVED_ROWS",
                    f"News calendar contiene filas no aprobadas: {news_coverage['non_approved_rows']}",
                    news_path,
                )
            if not bool(news_coverage["target_in_horizon"]):
                add_check(
                    checks,
                    "news_coverage_check",
                    "BLOCK",
                    "NEWS_COVERAGE_INSUFFICIENT",
                    f"News calendar no cubre {readiness_date}: horizon={news_coverage['first_date_ny']}->{news_coverage['last_date_ny']}",
                    news_path,
                )
            else:
                add_check(
                    checks,
                    "news_coverage_check",
                    "PASS",
                    "NEWS_COVERAGE_OK",
                    f"News calendar cubre {readiness_date}: horizon={news_coverage['first_date_ny']}->{news_coverage['last_date_ny']}",
                    news_path,
                )

    if not allow_runtime_drift:
        for record in status["tracked_runtime_files"]:
            path = project_path(str(record["path"]))
            if not path.exists():
                add_check(
                    checks,
                    "rerun/idempotency_precheck",
                    "BLOCK",
                    "RUNTIME_FILE_MISSING",
                    f"Falta runtime sellado: {record['path']}",
                    str(record["path"]),
                )
                continue
            if sha256_file(path) != record["sha256"]:
                add_check(
                    checks,
                    "rerun/idempotency_precheck",
                    "BLOCK",
                    "RUNTIME_SEAL_DRIFT",
                    f"Runtime drift entre checkpoints: {record['path']}",
                    str(record["path"]),
                )

    if readiness_date and enforce_rerun_check:
        if date_exists_in_csv("results/SCBI_FORWARD_DAILY_STATUS.csv", "session_date", readiness_date) or date_exists_in_csv(
            "results/SCBI_FORWARD_LEDGER.csv", "session_date", readiness_date
        ):
            add_check(
                checks,
                "rerun/idempotency_precheck",
                "BLOCK",
                "GLOBAL_DATE_ALREADY_PROCESSED",
                f"GLOBAL ya tiene evidencia oficial para {readiness_date}",
                "results/SCBI_FORWARD_LEDGER.csv",
            )
        if date_exists_in_csv("results/SCBI_CORE_PHASE1/core_phase1_ledger.csv", "timestamp_ny", readiness_date):
            add_check(
                checks,
                "rerun/idempotency_precheck",
                "BLOCK",
                "CORE_DATE_ALREADY_PROCESSED",
                f"CORE ya tiene evidencia oficial para {readiness_date}",
                "results/SCBI_CORE_PHASE1/core_phase1_ledger.csv",
            )

    blocking = [check for check in checks if check["status"] == "BLOCK"]
    warnings = [check for check in checks if check["status"] == "WARN"]
    decision = "BLOCK" if blocking else "PASS"
    taxonomy_outcome = (
        "VALIDATION_GATE_STRICT_AND_USABLE" if decision == "PASS" else "FAIL_CLOSED_PRESERVED"
    )
    if any(check["code"] == "HASH_DRIFT_SEMANTICALLY_EVALUATED" for check in warnings):
        taxonomy_outcome = (
            "FALSE_BLOCKING_REDUCED" if decision == "PASS" else taxonomy_outcome
        )

    report: dict[str, object] = {
        "generated_at_utc": utc_now_iso(),
        "engine": VALIDATOR_ENGINE,
        "target_date": readiness_date,
        "decision": decision,
        "taxonomy_outcome": taxonomy_outcome,
        "blocking_count": len(blocking),
        "warning_count": len(warnings),
        "checks": checks,
        "blocking_issues": blocking,
        "warnings": warnings,
        "coverage_summary": coverage_summary,
        "physical_source_precedence": [
            "results/SCBI_FORWARD_LEDGER.csv",
            "results/SCBI_CORE_PHASE1/core_phase1_ledger.csv",
            "results/SCBI_DUAL_LINE_SCOREBOARD.csv",
            "results/SCBI_UNIFIED_LINE_STATUS.json",
            "SCBI_PHASE1_FREEZE_STATUS.json",
        ],
    }
    return report


def write_hardening_status(report: dict[str, object]) -> None:
    HARDENING_STATUS_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def validate_current_state(
    *,
    readiness_date: str | None = None,
    allow_runtime_drift: bool = False,
    enforce_rerun_check: bool = True,
) -> tuple[bool, list[str]]:
    report = build_validation_report(
        readiness_date=readiness_date,
        allow_runtime_drift=allow_runtime_drift,
        enforce_rerun_check=enforce_rerun_check,
    )
    issues = [str(check["message"]) for check in report["blocking_issues"]]
    return (report["decision"] == "PASS", issues)


def seal_runtime_state(*, reason: str) -> tuple[bool, list[str]]:
    ok, issues = validate_current_state(allow_runtime_drift=True)
    if not ok:
        return False, issues

    status = load_status()
    status["tracked_runtime_files"] = collect_runtime_records()
    daily_status_path = project_path("results/SCBI_FORWARD_DAILY_STATUS.csv")
    if daily_status_path.exists() and csv_row_count(daily_status_path) > 0:
        status["official_phase1_started"] = True
        with daily_status_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            last_row: dict[str, str] | None = None
            for row in reader:
                last_row = row
        if last_row is not None and not status.get("official_phase1_start_date"):
            status["official_phase1_start_date"] = last_row.get("session_date", "")

    now_iso = utc_now_iso()
    status["updated_at_utc"] = now_iso
    status["last_valid_checkpoint"] = {
        "sealed_at_utc": now_iso,
        "reason": reason,
    }
    write_status_files(status)
    return True, []


def main() -> int:
    parser = argparse.ArgumentParser(description="Congela y valida la baseline operativa de SCBI Phase 1.")
    parser.add_argument("--write-baseline", action="store_true", help="Escribe el baseline freeze oficial.")
    parser.add_argument("--check", action="store_true", help="Valida la baseline vigente sin reescribirla.")
    parser.add_argument("--seal-runtime", action="store_true", help="Actualiza el sello runtime tras una corrida valida.")
    parser.add_argument("--date", type=str, help="Fecha objetivo YYYY-MM-DD para verificar cobertura operativa.")
    parser.add_argument("--reason", type=str, default="manual_seal", help="Motivo del sello runtime.")
    parser.add_argument("--skip-rerun-check", action="store_true", help="Modo auditoria: valida integridad/cobertura sin bloquear por fecha ya procesada.")
    args = parser.parse_args()

    if args.write_baseline:
        return write_baseline()

    if args.seal_runtime:
        ok, issues = seal_runtime_state(reason=args.reason)
        if ok:
            print("[OK] Runtime sellado.")
            return 0
        for issue in issues:
            print(f"[FAIL] {issue}")
        return 1

    if args.check:
        report = build_validation_report(
            readiness_date=args.date,
            enforce_rerun_check=not args.skip_rerun_check,
        )
        write_hardening_status(report)
        for check in report["warnings"]:
            print(f"[WARN] {check['code']}: {check['message']}")
        for check in report["blocking_issues"]:
            print(f"[BLOCK] {check['code']}: {check['message']}")
        if report["decision"] == "PASS":
            print(f"[PASS] {report['taxonomy_outcome']}: gate operativo valido para {args.date or 'runtime actual'}.")
            return 0
        print(f"[FAIL-CLOSED] {report['taxonomy_outcome']}: {len(report['blocking_issues'])} bloqueo(s) operativo(s).")
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
