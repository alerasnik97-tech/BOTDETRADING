from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data_certification.m3_bid_ask_builder import (
    CERTIFIED,
    audit_m1_source,
    build_m3_from_m1,
    file_sha256,
    gap_report,
    ohlc_integrity,
    read_price_csv,
    reject_source_for_m3,
    validate_m3_files,
)
from news_guard_strict import save_news_audit
from phase19_repaired_preflight import run_preflight


ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
LAB = ROOT / "BOT_V2_DAYTIME_LAB"
OUT = LAB / "outputs" / "m3_bid_ask_certification"
REPORTS = LAB / "reports"
MANIFEST = LAB / "data" / "certified_data_paths.json"
M1_BID = ROOT / "data_intake_2020_2026_bidask" / "raw" / "EURUSD_M1_BID_FULL_2020_2026.csv"
M1_ASK = ROOT / "data_intake_2020_2026_bidask" / "raw" / "EURUSD_M1_ASK_FULL_2020_2026.csv"
NEWS_2020_2026 = ROOT / "data_intake_2020_2026_bidask" / "news" / "EURUSD_NEWS_2020_2026.csv"
M3_DIR = LAB / "data" / "certified_m3"
NOW = datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dirs() -> None:
    for name in [
        "diagnosis",
        "inventory",
        "source_audit",
        "m3_validation",
        "manifest",
        "news_guard",
        "tests",
        "phase19_preflight",
    ]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def count_csv_rows(path: Path) -> int | None:
    if not path.exists() or path.suffix.lower() != ".csv":
        return None
    rows = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            rows += chunk.count(b"\n")
    return max(rows - 1, 0)


def first_last_timestamp(path: Path) -> tuple[str | None, str | None, str]:
    if not path.exists() or path.suffix.lower() != ".csv":
        return None, None, "unknown"
    try:
        first = pd.read_csv(path, nrows=1)
        columns = list(first.columns)
        ts_col = "timestamp_utc" if "timestamp_utc" in columns else ("timestamp" if "timestamp" in columns else None)
        if ts_col is None:
            return None, None, "unknown"
        start = pd.to_datetime(first[ts_col].iloc[0], utc=True, errors="coerce")
        last_line = ""
        with path.open("rb") as f:
            f.seek(0, 2)
            pos = f.tell() - 1
            while pos > 0:
                f.seek(pos)
                char = f.read(1)
                if char == b"\n" and last_line:
                    break
                if char not in {b"\n", b"\r"}:
                    last_line = char.decode("utf-8", errors="ignore") + last_line
                pos -= 1
        parsed = next(csv.reader([last_line]))
        idx = columns.index(ts_col)
        end = pd.to_datetime(parsed[idx], utc=True, errors="coerce") if idx < len(parsed) else pd.NaT
        return (
            start.isoformat() if pd.notna(start) else None,
            end.isoformat() if pd.notna(end) else None,
            "UTC" if pd.notna(start) else "unknown",
        )
    except Exception:
        return None, None, "unknown"


def timeframe_from_name(path: Path) -> str:
    name = path.name.upper()
    for token in ["TICK", "M1", "M3", "M5", "H1"]:
        if token in name:
            return token
    if "NEWS" in name:
        return "NEWS"
    if path.name == "certified_data_paths.json":
        return "MANIFEST"
    return "UNKNOWN"


def source_type_from_name(path: Path) -> str:
    name = path.name.upper()
    if path.suffix.lower() == ".bi5" or "TICK" in name:
        return "tick_or_raw_cache"
    if "M1" in name:
        return "m1"
    if "M3" in name:
        return "m3"
    if "M5" in name:
        return "m5"
    if "H1" in name:
        return "h1"
    if "NEWS" in name:
        return "news"
    if path.name == "certified_data_paths.json":
        return "manifest"
    return "unknown"


def inventory_row(path: Path, paired_bid: bool | None = None, paired_ask: bool | None = None) -> dict:
    source_type = source_type_from_name(path)
    name = path.name.upper()
    bid_present = ("BID" in name) if paired_bid is None else paired_bid
    ask_present = ("ASK" in name) if paired_ask is None else paired_ask
    status = reject_source_for_m3(source_type, bid_present, ask_present)
    if source_type in {"news", "manifest"}:
        status = "SOURCE_REQUIRES_REPAIR" if path.exists() else "SOURCE_MISSING"
    start, end, tz = first_last_timestamp(path)
    return {
        "file_path": str(path),
        "timeframe": timeframe_from_name(path),
        "bid_present": bool(bid_present),
        "ask_present": bool(ask_present),
        "start_date": start,
        "end_date": end,
        "rows": count_csv_rows(path),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 3) if path.exists() else None,
        "timezone": tz,
        "source_type": source_type,
        "certification_status": status,
        "notes": "individual_file_inventory",
    }


def build_inventory() -> tuple[pd.DataFrame, dict]:
    paths: set[Path] = {
        MANIFEST,
        M1_BID,
        M1_ASK,
        NEWS_2020_2026,
        ROOT / "research_lab" / "data" / "news" / "news_events.csv",
    }
    for folder in [ROOT / "data_intake_2020_2026_bidask" / "prepared", ROOT / "data_intake_2015_2019" / "prepared", LAB / "data"]:
        if folder.exists():
            paths.update(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".json"})
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for period in manifest.values():
            if isinstance(period, dict):
                for entry in period.values():
                    path = Path(entry["path"] if isinstance(entry, dict) else entry)
                    paths.add(path)
    rows = [inventory_row(p) for p in sorted(paths, key=lambda x: str(x).lower())]
    if M1_BID.exists() and M1_ASK.exists():
        rows.append(
            {
                **inventory_row(M1_BID, paired_bid=True, paired_ask=True),
                "file_path": f"{M1_BID} | {M1_ASK}",
                "certification_status": "SOURCE_VALID_FOR_M3_CERTIFICATION",
                "notes": "paired_m1_bid_ask_real_source_selected",
            }
        )
    bi5_root = ROOT / "data_intake_2015_2019" / "cache"
    bi5_count = len(list(bi5_root.rglob("*.bi5"))) if bi5_root.exists() else 0
    if bi5_count:
        rows.append(
            {
                "file_path": str(bi5_root / "*.bi5"),
                "timeframe": "TICK",
                "bid_present": False,
                "ask_present": False,
                "start_date": None,
                "end_date": None,
                "rows": None,
                "size_mb": None,
                "timezone": "unknown",
                "source_type": "tick_or_raw_cache",
                "certification_status": "SOURCE_REQUIRES_REPAIR",
                "notes": f"{bi5_count} raw cache files detected; not decoded/certified for Phase19 M3",
            }
        )
    df = pd.DataFrame(rows)
    summary = {
        "generated_at": NOW,
        "verdict": "SOURCE_VALID_FOR_M3_CERTIFICATION" if M1_BID.exists() and M1_ASK.exists() else "SOURCE_MISSING",
        "selected_source": "M1 BID/ASK real 2020-2026" if M1_BID.exists() and M1_ASK.exists() else None,
        "m1_bid": str(M1_BID),
        "m1_ask": str(M1_ASK),
        "tick_cache_files_detected": bi5_count,
        "phase19_not_run": True,
    }
    return df, summary


def write_starting_point() -> None:
    payload = {
        "generated_at": NOW,
        "phase19_legacy": "PHASE19_INVALIDATED",
        "phase19_repaired": "BLOCKED_PENDING_M3_BID_ASK_AND_STRICT_NEWS",
        "phase18": "PROTECTED_UNCHANGED",
        "scbi": "PROTECTED_UNTOUCHED",
        "strategy_backtests_run": False,
        "objective": "certify M3 BID/ASK from valid granular data and strict News Guard only",
    }
    write_json(OUT / "diagnosis" / "m3_certification_starting_point.json", payload)
    write_text(
        OUT / "diagnosis" / "m3_certification_starting_point.md",
        "# M3 BID/ASK Certification Starting Point\n\n"
        "- Phase19 sigue invalidada.\n"
        "- Phase19 repaired sigue bloqueada hasta certificar M3 BID/ASK y News Guard.\n"
        "- No se correran backtests de estrategia en esta fase.\n"
        "- Phase18 sigue protegida.\n"
        "- SCBI no se toca.\n",
    )


def write_source_audit(bid: pd.DataFrame, ask: pd.DataFrame, summary: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = pd.merge(bid[["timestamp", "close"]], ask[["timestamp", "close"]], on="timestamp", suffixes=("_bid", "_ask"))
    merged["spread_close_pips"] = (merged["close_ask"] - merged["close_bid"]) / 0.0001
    merged["year"] = merged["timestamp"].dt.year
    coverage = merged.groupby("year").size().reset_index(name="rows")
    coverage["start"] = merged.groupby("year")["timestamp"].min().values
    coverage["end"] = merged.groupby("year")["timestamp"].max().values
    spread = (
        merged.groupby("year")["spread_close_pips"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
        .rename(columns={"count": "rows", "mean": "mean_pips", "median": "median_pips", "min": "min_pips", "max": "max_pips"})
    )
    gaps = gap_report(merged["timestamp"], 1)
    ohlc = pd.DataFrame(
        [
            {"side": "bid", **ohlc_integrity(bid)},
            {"side": "ask", **ohlc_integrity(ask)},
        ]
    )
    coverage.to_csv(OUT / "source_audit" / "source_coverage_by_year.csv", index=False)
    gaps.to_csv(OUT / "source_audit" / "source_gap_report.csv", index=False)
    spread.to_csv(OUT / "source_audit" / "source_spread_report.csv", index=False)
    ohlc.to_csv(OUT / "source_audit" / "source_ohlc_integrity_report.csv", index=False)
    write_json(OUT / "source_audit" / "source_audit_summary.json", summary)
    write_text(
        OUT / "source_audit" / "source_audit_summary.md",
        "# Source Audit Summary\n\n"
        f"Verdicto: {summary['verdict']}\n\n"
        f"Fuente: M1 BID/ASK real.\n\n"
        f"Rows merged: {summary.get('merged_rows')}\n\n"
        f"Critical gaps: {summary.get('critical_gap_count')}\n\n"
        f"Spread negativo: {summary.get('spread_negative_rows')}\n",
    )
    return coverage, gaps


def write_m3_validation(metadata: dict, validation: dict) -> None:
    bid = read_price_csv(metadata["bid_path"])
    ask = read_price_csv(metadata["ask_path"])
    merged = pd.merge(bid[["timestamp", "close"]], ask[["timestamp", "close"]], on="timestamp", suffixes=("_bid", "_ask"))
    merged["spread_close_pips"] = (merged["close_ask"] - merged["close_bid"]) / 0.0001
    merged["year"] = merged["timestamp"].dt.year
    coverage = merged.groupby("year").size().reset_index(name="rows")
    coverage["start"] = merged.groupby("year")["timestamp"].min().values
    coverage["end"] = merged.groupby("year")["timestamp"].max().values
    gaps = gap_report(merged["timestamp"], 3)
    spread = (
        merged.groupby("year")["spread_close_pips"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
        .rename(columns={"count": "rows", "mean": "mean_pips", "median": "median_pips", "min": "min_pips", "max": "max_pips"})
    )
    ohlc = pd.DataFrame(
        [
            {"side": "bid", **ohlc_integrity(bid)},
            {"side": "ask", **ohlc_integrity(ask)},
        ]
    )
    ny = merged["timestamp"].dt.tz_convert("America/New_York")
    hour = ny.dt.hour + ny.dt.minute / 60.0
    sessions = pd.DataFrame(
        [
            {"window_ny": "07:00-20:00", "rows": int(((hour >= 7.0) & (hour < 20.0)).sum())},
            {"window_ny": "08:00-16:30", "rows": int(((hour >= 8.0) & (hour < 16.5)).sum())},
            {"window_ny": "08:00-11:00", "rows": int(((hour >= 8.0) & (hour < 11.0)).sum())},
        ]
    )
    coverage.to_csv(OUT / "m3_validation" / "m3_coverage_by_year.csv", index=False)
    gaps.to_csv(OUT / "m3_validation" / "m3_gap_report.csv", index=False)
    spread.to_csv(OUT / "m3_validation" / "m3_spread_report.csv", index=False)
    ohlc.to_csv(OUT / "m3_validation" / "m3_ohlc_integrity_report.csv", index=False)
    sessions.to_csv(OUT / "m3_validation" / "m3_session_coverage_report.csv", index=False)
    write_json(OUT / "m3_validation" / "m3_validation_summary.json", validation)
    write_text(
        OUT / "m3_validation" / "m3_validation_summary.md",
        "# M3 Validation Summary\n\n"
        f"Verdicto: {validation['verdict']}\n\n"
        f"Tipo: {CERTIFIED}\n\n"
        f"Rows: {validation.get('merged_rows')}\n\n"
        f"Critical gaps: {validation.get('m3_critical_gap_count')}\n",
    )


def update_manifest(metadata: dict, validation: dict) -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    period = manifest.setdefault("period_2020_2026", {})
    if validation["verdict"] == "M3_BID_ASK_CERTIFIED":
        common = {
            "rows": metadata["rows"],
            "start": metadata["start"],
            "end": metadata["end"],
            "timezone": "UTC",
            "certification_status": validation["verdict"],
            "certification_name": CERTIFIED,
            "source_type": "m1_derived",
            "generated_at": NOW,
        }
        period["m3_bid"] = {
            "path": metadata["bid_path"],
            "source": metadata["source_bid"],
            "sha256": metadata["bid_sha256"],
            **common,
        }
        period["m3_ask"] = {
            "path": metadata["ask_path"],
            "source": metadata["source_ask"],
            "sha256": metadata["ask_sha256"],
            **common,
        }
        period["m3_spread"] = {
            "path": metadata["spread_path"],
            "source": f"{metadata['source_bid']} | {metadata['source_ask']}",
            "sha256": metadata["spread_sha256"],
            **common,
        }
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        result = {"verdict": "MANIFEST_UPDATED_WITH_CERTIFIED_M3", "m3_declared": True, "manifest": str(MANIFEST)}
    else:
        result = {"verdict": "MANIFEST_NOT_UPDATED_M3_NOT_CERTIFIED", "m3_declared": False, "manifest": str(MANIFEST)}
    write_json(OUT / "manifest" / "m3_manifest_update_summary.json", result)
    write_text(
        OUT / "manifest" / "m3_manifest_update_summary.md",
        "# M3 Manifest Update Summary\n\n"
        f"Verdicto: {result['verdict']}\n\n"
        f"M3 declarado: {result['m3_declared']}\n",
    )
    return result


def run_unit_tests() -> dict:
    tests = [
        "BOT_V2_DAYTIME_LAB.tests.engine_safety.test_m3_bid_ask_certification",
        "BOT_V2_DAYTIME_LAB.tests.engine_safety.test_news_guard_strict",
        "BOT_V2_DAYTIME_LAB.tests.engine_safety.test_phase19_repaired_requires_m3",
    ]
    cmd = [sys.executable, "-m", "unittest", *tests]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    payload = {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "verdict": "TESTS_PASSED" if proc.returncode == 0 else "TESTS_FAILED",
    }
    write_json(OUT / "tests" / "m3_news_certification_test_results.json", payload)
    write_text(
        OUT / "tests" / "m3_news_certification_test_results.md",
        "# M3/News Certification Test Results\n\n"
        f"Verdicto: {payload['verdict']}\n\n"
        "```text\n" + (proc.stdout + proc.stderr).strip() + "\n```\n",
    )
    return payload


def write_preflight(preflight: dict) -> None:
    write_json(OUT / "phase19_preflight" / "phase19_repaired_preflight_report.json", preflight)
    write_text(
        OUT / "phase19_preflight" / "phase19_repaired_preflight_report.md",
        "# Phase19 Repaired Preflight\n\n"
        f"Verdicto: {preflight['verdict']}\n\n"
        f"Blockers: {', '.join(preflight.get('blockers', [])) or 'none'}\n\n"
        "Phase19 repaired no fue ejecutada.\n",
    )


def write_final_report(state: dict) -> None:
    final = {
        "generated_at": NOW,
        "objective": "M3 BID/ASK certified data and strict news guard foundation",
        "final_verdict": state["final_verdict"],
        "source": state.get("source_summary"),
        "m3_metadata": state.get("m3_metadata"),
        "m3_validation": state.get("m3_validation"),
        "manifest_update": state.get("manifest_update"),
        "news_guard": state.get("news_guard"),
        "tests": state.get("tests"),
        "phase19_repaired_preflight": state.get("preflight"),
        "phase19_not_run": True,
        "phase18_status": "PROTECTED_UNCHANGED",
        "phase19_legacy_status": "PHASE19_INVALIDATED",
        "scbi_touched": False,
        "mt5_touched": False,
        "real_trading_enabled": False,
        "allowed_next_step": "run Phase19 repaired retest only in a separate authorized phase if preflight passed",
        "prohibited": ["Phase19 legacy authority", "M3 from M5", "MT5", "real trading", "SCBI edits", "Phase18 edits"],
    }
    write_json(REPORTS / "M3_BID_ASK_AND_NEWS_CERTIFICATION_REPORT.json", final)
    md = [
        "# M3 BID/ASK and News Certification Report",
        "",
        f"Verdicto final: {state['final_verdict']}",
        "",
        "## Objetivo",
        "Certificar M3 BID/ASK desde fuente granular valida y blindar News Guard estricto. No se corrio Phase19.",
        "",
        "## Fuente encontrada",
        state.get("source_text", "No valid source."),
        "",
        "## M3",
        f"Tipo: {state.get('m3_type', 'M3_CERTIFICATION_BLOCKED')}",
        f"Validacion: {state.get('m3_validation', {}).get('verdict', 'M3_CERTIFICATION_BLOCKED')}",
        "",
        "## News Guard estricto",
        f"Verdicto: {state.get('news_guard', {}).get('verdict', 'NEWS_GUARD_INVALIDATED')}",
        "",
        "## Tests",
        f"Verdicto: {state.get('tests', {}).get('verdict', 'TESTS_NOT_RUN')}",
        "",
        "## Phase19 repaired preflight",
        f"Verdicto: {state.get('preflight', {}).get('verdict', 'PHASE19_REPAIRED_PREFLIGHT_BLOCKED')}",
        "",
        "## Permitido",
        "Solo reabrir un retest Phase19 repaired en una fase posterior autorizada si el preflight paso.",
        "",
        "## Prohibido",
        "No usar Phase19 legacy como autoridad; no usar M3 desde M5; no tocar MT5, real, SCBI ni Phase18.",
        "",
        "## Siguiente paso unico",
        state["next_step"],
        "",
    ]
    write_text(REPORTS / "M3_BID_ASK_AND_NEWS_CERTIFICATION_REPORT.md", "\n".join(md))


def update_master_status(final_verdict: str, preflight: dict) -> None:
    current = {
        "project_status": {
            "date": "2026-04-28",
            "root_status": "M3_BID_ASK_NEWS_FOUNDATION_UPDATED",
            "lab": "BOT_V2_DAYTIME_LAB",
            "strategies": {
                "SCBI_M5_GLOBAL": "protected_unchanged",
                "Phase18_Fractal_Sweep": "daytime_baseline_protected",
                "Phase19_Expanded_Sweep": "PHASE19_INVALIDATED",
                "Phase19_Repaired": preflight.get("verdict", "PHASE19_REPAIRED_PREFLIGHT_BLOCKED"),
            },
            "critical_note": "M3 BID/ASK y News Guard fueron tratados como capa de datos. No se corrio Phase19 ni se promovio ninguna estrategia.",
            "final_data_verdict": final_verdict,
            "mt5_touched": False,
            "real_trading_enabled": False,
        }
    }
    write_json(ROOT / "01_CURRENT_PROJECT_STATUS.json", current)
    write_text(
        ROOT / "01_CURRENT_PROJECT_STATUS.md",
        "# CURRENT PROJECT STATUS\n\n"
        "Fecha de actualizacion: 2026-04-28\n"
        f"Estado data foundation: {final_verdict}\n\n"
        "## Estado de estrategias\n"
        "- SCBI_M5_GLOBAL: PROTEGIDA / SIN CAMBIOS.\n"
        "- Phase18 Fractal Sweep: baseline diurna protegida; no fue reemplazada.\n"
        "- Phase19 legacy: PHASE19_INVALIDATED; no es autoridad positiva.\n"
        f"- Phase19 repaired: {preflight.get('verdict', 'PHASE19_REPAIRED_PREFLIGHT_BLOCKED')}; no fue ejecutada en esta fase.\n\n"
        "## Nota critica\n"
        "Esta fase certifica datos M3 BID/ASK y News Guard estricto. No optimiza parametros ni corre backtests Phase19.\n\n"
        "## Siguiente paso unico\n"
        "Si el preflight paso, autorizar en una fase separada un retest Phase19 repaired; si no, reparar la capa bloqueante.\n",
    )
    authority = {
        "authority_hierarchy": {
            "primary": {
                "id": "SCBI_M5_GLOBAL",
                "role": "overnight_authority",
                "status": "protected_unchanged",
                "window": "00:00-05:00 UTC",
            },
            "daytime_baseline": {
                "id": "Phase18_H1_Fractal_Sweep_First_M3_CHOCH",
                "pf": 1.63,
                "sample": 1040,
                "status": "validated_for_forward_demo_baseline_protected",
            },
            "data_foundation": {
                "id": "M3_BID_ASK_AND_STRICT_NEWS_GUARD",
                "status": final_verdict,
                "phase19_repaired_preflight": preflight.get("verdict"),
            },
            "quarantined_or_rejected": [
                {
                    "id": "Phase19_Expanded_Sweep",
                    "reported_pf": 3.18,
                    "reported_sample": 3177,
                    "status": "PHASE19_INVALIDATED",
                    "reason": "legacy_invalidated; repaired line may only be retested after data/news preflight",
                },
                {"id": "Phase12", "status": "invalidated_not_authority"},
            ],
        },
        "lab_status": {"id": "BOT_V2_DAYTIME_LAB", "role": "research_only", "authority": "none_production"},
    }
    write_json(ROOT / "02_STRATEGY_AUTHORITY_MAP.json", authority)
    write_text(
        ROOT / "02_STRATEGY_AUTHORITY_MAP.md",
        "# STRATEGY AUTHORITY MAP\n\n"
        "## 1. Autoridad maxima: SCBI_M5_GLOBAL\n"
        "- Estado: PROTEGIDA / SIN CAMBIOS.\n"
        "- No fue modificada por la certificacion M3/news.\n\n"
        "## 2. Baseline diurna protegida\n"
        "- Phase18 H1 Fractal Sweep + First M3 CHOCH: VALIDATED_FOR_FORWARD_DEMO, PF 1.63, sample 1.040.\n"
        "- Phase19 no reemplaza Phase18.\n\n"
        "## 3. Data foundation Phase19 repaired\n"
        f"- Estado: {final_verdict}.\n"
        f"- Preflight: {preflight.get('verdict', 'PHASE19_REPAIRED_PREFLIGHT_BLOCKED')}.\n"
        "- Phase19 repaired no fue ejecutada en esta fase.\n\n"
        "## 4. Phase19 legacy\n"
        "- Estado: PHASE19_INVALIDATED.\n"
        "- No habilita forward demo, real, VPS ni reemplazo operativo.\n\n"
        "## 5. Regla de autoridad\n"
        "Si hay contradiccion entre narrativa previa y archivos reales actuales, mandan estos archivos de estado y el reporte M3_BID_ASK_AND_NEWS_CERTIFICATION_REPORT.\n",
    )


def main() -> int:
    ensure_dirs()
    write_starting_point()
    inventory, inventory_summary = build_inventory()
    inventory.to_csv(OUT / "inventory" / "data_inventory.csv", index=False)
    write_json(OUT / "inventory" / "data_inventory_summary.json", inventory_summary)
    write_text(
        OUT / "inventory" / "data_inventory_summary.md",
        "# Data Inventory Summary\n\n"
        f"Verdicto: {inventory_summary['verdict']}\n\n"
        f"Fuente seleccionada: {inventory_summary.get('selected_source')}\n\n"
        "Phase19 no fue ejecutada.\n",
    )
    state: dict = {"source_summary": inventory_summary}
    if not (M1_BID.exists() and M1_ASK.exists()):
        state.update(
            {
                "final_verdict": "NEWS_CERTIFIED_M3_BLOCKED",
                "source_text": "No se encontro M1 BID/ASK real completo.",
                "m3_type": "M3_CERTIFICATION_BLOCKED",
                "next_step": "Ingerir fuente M1/tick BID/ASK real certificable antes de reabrir Phase19 repaired.",
            }
        )
    else:
        bid = read_price_csv(M1_BID)
        ask = read_price_csv(M1_ASK)
        source_summary = audit_m1_source(bid, ask)
        state["source_summary"] = source_summary
        write_source_audit(bid, ask, source_summary)
        if source_summary["verdict"] != "SOURCE_VALID_FOR_M3_CERTIFICATION":
            state.update(
                {
                    "final_verdict": "DATA_REPAIR_REQUIRED",
                    "source_text": "M1 BID/ASK encontrado pero fallo auditoria critica.",
                    "m3_type": "M3_CERTIFICATION_BLOCKED",
                    "next_step": "Reparar fuente M1 BID/ASK antes de generar M3 certificado.",
                }
            )
        else:
            metadata = build_m3_from_m1(M1_BID, M1_ASK, M3_DIR)
            validation = validate_m3_files(metadata["bid_path"], metadata["ask_path"])
            metadata["validation_status"] = validation["verdict"]
            metadata["manifest_declared"] = validation["verdict"] == "M3_BID_ASK_CERTIFIED"
            metadata["usable_for_phase19_repaired"] = validation["verdict"] == "M3_BID_ASK_CERTIFIED"
            metadata["blocked_reason"] = None if validation["verdict"] == "M3_BID_ASK_CERTIFIED" else "M3 validation did not pass; generated files remain local diagnostics only"
            write_json(M3_DIR / "M3_CERTIFICATION_METADATA.json", metadata)
            write_m3_validation(metadata, validation)
            manifest_update = update_manifest(metadata, validation)
            state.update(
                {
                    "m3_metadata": metadata,
                    "m3_validation": validation,
                    "manifest_update": manifest_update,
                    "source_text": f"M1 BID/ASK real certificado: {M1_BID} | {M1_ASK}",
                    "m3_type": CERTIFIED if validation["verdict"] == "M3_BID_ASK_CERTIFIED" else "M3_CERTIFICATION_BLOCKED",
                }
            )
    news_summary = save_news_audit(NEWS_2020_2026, OUT / "news_guard")
    state["news_guard"] = news_summary
    tests = run_unit_tests()
    state["tests"] = tests
    preflight = run_preflight()
    state["preflight"] = preflight
    write_preflight(preflight)
    if state.get("m3_validation", {}).get("verdict") == "M3_BID_ASK_CERTIFIED" and news_summary.get("verdict") == "NEWS_GUARD_STRICT_CERTIFIED" and tests.get("verdict") == "TESTS_PASSED" and preflight.get("verdict") == "PHASE19_REPAIRED_PREFLIGHT_PASSED":
        state["final_verdict"] = "M3_AND_NEWS_CERTIFIED_PHASE19_READY_FOR_RETEST"
        state["next_step"] = "Autorizar en una fase separada el retest Phase19 repaired; no usar legacy."
    elif state.get("m3_validation", {}).get("verdict") == "M3_BID_ASK_CERTIFIED" and news_summary.get("verdict") != "NEWS_GUARD_STRICT_CERTIFIED":
        state["final_verdict"] = "M3_CERTIFIED_NEWS_REQUIRES_REPAIR"
        state["next_step"] = "Reparar News Guard/feed antes de retest Phase19 repaired."
    elif state.get("m3_validation", {}).get("verdict") != "M3_BID_ASK_CERTIFIED" and news_summary.get("verdict") == "NEWS_GUARD_STRICT_CERTIFIED":
        state["final_verdict"] = state.get("final_verdict", "NEWS_CERTIFIED_M3_BLOCKED")
        state["next_step"] = state.get("next_step", "Reparar/certificar M3 BID/ASK antes de retest Phase19 repaired.")
    else:
        state["final_verdict"] = state.get("final_verdict", "M3_AND_NEWS_BLOCKED")
        state["next_step"] = state.get("next_step", "Reparar data M3 y News Guard antes de cualquier retest Phase19.")
    update_master_status(state["final_verdict"], preflight)
    write_final_report(state)
    print(json.dumps({"final_verdict": state["final_verdict"], "preflight": preflight["verdict"], "phase19_run": False}, indent=2))
    return 0 if tests.get("verdict") == "TESTS_PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
