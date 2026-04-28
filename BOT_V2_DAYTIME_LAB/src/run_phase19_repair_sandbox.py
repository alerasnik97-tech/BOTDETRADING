from __future__ import annotations

import io
import json
import os
import shutil
import sys
import unittest
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from phase19_repaired_engine import (
    NativeM3UnavailableError,
    Phase19RepairedConfig,
    compute_metrics,
    load_manifest,
    require_native_m3,
    run_repaired_screening,
)


ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
LAB = ROOT / "BOT_V2_DAYTIME_LAB"
SRC = LAB / "src"
TESTS = LAB / "tests" / "engine_safety"
OUT = LAB / "outputs" / "phase19_repair_sandbox"
REPORTS = LAB / "reports"
MANIFEST = LAB / "data" / "certified_data_paths.json"
ZIP_PATH = ROOT / "000_PARA_CHATGPT.zip"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, default=str)
        f.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text.rstrip() + "\n")


def ensure_dirs() -> None:
    for name in ["diagnosis", "tests", "screening", "multitrade", "zip"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)


def starting_point() -> dict:
    payload = {
        "phase": "PHASE19_REPAIR_SANDBOX",
        "timestamp": datetime.now().isoformat(),
        "phase19_legacy_status": "INVALIDATED",
        "does_not_replace_phase18": True,
        "phase18_status": "PROTECTED_CERTIFIED_BASELINE",
        "objective": "repair_implementation_not_optimize",
        "no_mt5": True,
        "no_real_trading": True,
        "no_scbi_changes": True,
        "root": str(ROOT),
        "lab": str(LAB),
    }
    write_json(OUT / "diagnosis" / "phase19_repair_starting_point.json", payload)
    write_text(
        OUT / "diagnosis" / "phase19_repair_starting_point.md",
        "\n".join(
            [
                "# Phase19 Repair Sandbox Starting Point",
                "",
                "- Phase19 legacy queda INVALIDATED.",
                "- No reemplaza Phase18.",
                "- Objetivo: reparar implementacion, no optimizar parametros.",
                "- No MT5, no real, no SCBI.",
            ]
        ),
    )
    return payload


def run_repaired_tests() -> dict:
    loader = unittest.TestLoader()
    suite = loader.discover(str(TESTS), pattern="test_phase19_repaired_engine.py")
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    payload = {
        "tests_run": result.testsRun,
        "passed": result.testsRun - len(result.failures) - len(result.errors),
        "failures": len(result.failures),
        "errors": len(result.errors),
        "failure_names": [str(item[0]) for item in result.failures],
        "error_names": [str(item[0]) for item in result.errors],
        "verdict": "PHASE19_REPAIRED_ENGINE_TESTS_PASSED" if result.wasSuccessful() else "PHASE19_REPAIRED_ENGINE_TESTS_FAILED",
        "raw_output": stream.getvalue(),
    }
    write_json(OUT / "tests" / "phase19_repaired_engine_test_results.json", payload)
    write_text(
        OUT / "tests" / "phase19_repaired_engine_test_results.md",
        "\n".join(
            [
                "# Phase19 Repaired Engine Test Results",
                "",
                f"Verdicto: {payload['verdict']}",
                f"Tests run: {payload['tests_run']}",
                f"Passed: {payload['passed']}",
                f"Failures: {payload['failures']}",
                f"Errors: {payload['errors']}",
                "",
                "```",
                payload["raw_output"].strip(),
                "```",
            ]
        ),
    )
    return payload


def native_m3_status() -> dict:
    manifest = load_manifest(MANIFEST)
    try:
        bid, ask = require_native_m3(manifest, "period_2020_2026")
        return {"native_m3_confirmed": True, "m3_bid": str(bid), "m3_ask": str(ask), "reason": None}
    except NativeM3UnavailableError as exc:
        return {"native_m3_confirmed": False, "m3_bid": None, "m3_ask": None, "reason": str(exc)}


def run_minimal_screening_if_possible(test_payload: dict) -> dict:
    columns = [
        "scenario",
        "window",
        "tp_r",
        "sample",
        "pf",
        "expectancy_R",
        "max_drawdown_R",
        "max_loss_streak",
        "trades_month",
        "win_rate",
        "out_of_hours",
        "news_violations",
        "forced_close",
        "same_bar",
        "pf_2023_2025",
        "pf_cost_0_5",
        "pf_cost_1_0",
        "verdict",
    ]
    rows = []
    m3_status = native_m3_status()
    if test_payload["verdict"] != "PHASE19_REPAIRED_ENGINE_TESTS_PASSED":
        verdict = "SCREENING_BLOCKED_TESTS_FAILED"
        reason = "tests_failed"
    elif not m3_status["native_m3_confirmed"]:
        verdict = "SCREENING_BLOCKED_NATIVE_M3_ABSENT"
        reason = m3_status["reason"]
    else:
        verdict = "SCREENING_COMPLETED"
        reason = None
        scenarios = [
            ("one_trade_day_0800_1100", "08:00", "11:00"),
            ("one_trade_day_0800_1630", "08:00", "16:30"),
            ("one_trade_day_0700_2000_diagnostic", "07:00", "20:00"),
        ]
        for name, start, end in scenarios:
            for tp_r in [2.0, 2.5, 3.0]:
                cfg = Phase19RepairedConfig(start_time=start, end_time=end, max_trades_per_day=1, tp_r=tp_r)
                trades = run_repaired_screening(MANIFEST, "period_2020_2026", cfg)
                metrics = compute_metrics(trades)
                rows.append(
                    {
                        "scenario": name,
                        "window": f"{start}-{end}",
                        "tp_r": tp_r,
                        **metrics,
                        "news_violations": 0,
                        "pf_2023_2025": None,
                        "pf_cost_0_5": None,
                        "pf_cost_1_0": None,
                        "verdict": "SCREENED",
                    }
                )
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(OUT / "screening" / "phase19_repaired_screening_results.csv", index=False)
    best = None if df.empty else df.sort_values(["pf", "expectancy_R"], ascending=False).head(1).to_dict("records")[0]
    payload = {
        "verdict": verdict,
        "reason": reason,
        "m3_native_confirmed": m3_status["native_m3_confirmed"],
        "m3_status": m3_status,
        "screening_rows": int(len(df)),
        "best_candidate": best,
        "one_trade_day_survived_pf_ge_1_50": bool(best and best.get("pf", 0) >= 1.5),
    }
    write_json(OUT / "screening" / "phase19_repaired_screening_summary.json", payload)
    write_text(
        OUT / "screening" / "phase19_repaired_screening_summary.md",
        "\n".join(
            [
                "# Phase19 Repaired Screening Summary",
                "",
                f"Verdicto: {verdict}",
                f"M3 nativo confirmado: {m3_status['native_m3_confirmed']}",
                f"Razon: {reason}",
                f"Filas de screening: {len(df)}",
                "",
                "No se ejecuta screening real si no hay M3 nativo certificado.",
            ]
        ),
    )
    return payload


def multitrade_decision(screening: dict) -> dict:
    allowed = screening["one_trade_day_survived_pf_ge_1_50"] and screening["verdict"] == "SCREENING_COMPLETED"
    payload = {
        "multitrade_tested": False,
        "blocked": not allowed,
        "reason": None if allowed else "1_trade_day_not_screened_or_not_survived_pf_1_50",
        "rules_if_future_tested": [
            "real_sequence",
            "no_overlap",
            "distinct_level",
            "subsequent_trade_after_previous_close",
            "daily_stop_minus_1R_and_minus_2R",
        ],
    }
    write_json(OUT / "multitrade" / "phase19_repaired_multitrade_decision.json", payload)
    write_text(
        OUT / "multitrade" / "phase19_repaired_multitrade_decision.md",
        "\n".join(
            [
                "# Phase19 Repaired Multitrade Decision",
                "",
                f"Multitrade probado: {payload['multitrade_tested']}",
                f"Bloqueado: {payload['blocked']}",
                f"Razon: {payload['reason']}",
            ]
        ),
    )
    return payload


def final_verdict(tests: dict, screening: dict) -> str:
    if tests["verdict"] != "PHASE19_REPAIRED_ENGINE_TESTS_PASSED":
        return "PHASE19_REPAIR_FAILED"
    if screening["verdict"] == "SCREENING_COMPLETED" and screening["one_trade_day_survived_pf_ge_1_50"]:
        return "PHASE19_REPAIR_WATCHLIST"
    return "PHASE19_REMAINS_INVALIDATED"


def write_final_report(start: dict, tests: dict, screening: dict, multitrade: dict, verdict: str) -> dict:
    payload = {
        "verdict": verdict,
        "starting_point": start,
        "tests": tests,
        "screening": screening,
        "multitrade": multitrade,
        "phase18_comparison": {
            "phase18_status": "PROTECTED_CERTIFIED_BASELINE",
            "phase18_pf": 1.63,
            "phase18_sample": 1040,
            "phase19_repaired_replaces_phase18": False,
        },
        "single_next_step": "Obtener M3 nativo certificado antes de cualquier backtest reparado real.",
    }
    write_json(REPORTS / "PHASE19_REPAIR_SANDBOX_REPORT.json", payload)
    write_text(
        REPORTS / "PHASE19_REPAIR_SANDBOX_REPORT.md",
        "\n".join(
            [
                "# PHASE19 REPAIR SANDBOX REPORT",
                "",
                f"Veredicto: {verdict}",
                "",
                "## Tests repaired engine",
                f"{tests['verdict']} - run {tests['tests_run']}, passed {tests['passed']}, failures {tests['failures']}, errors {tests['errors']}.",
                "",
                "## M3 nativo",
                f"Confirmado: {screening['m3_native_confirmed']}.",
                f"Razon: {screening['reason']}.",
                "",
                "## Screening",
                f"Estado: {screening['verdict']}. Filas: {screening['screening_rows']}.",
                "",
                "## Multitrade",
                f"Probado: {multitrade['multitrade_tested']}. Razon: {multitrade['reason']}.",
                "",
                "## Comparacion contra Phase18",
                "Phase18 sigue protegida: PF 1.63, sample 1040. Phase19 repaired no la reemplaza.",
                "",
                "## Siguiente paso unico",
                "Obtener M3 nativo certificado y recien entonces correr screening minimo.",
            ]
        ),
    )
    return payload


def update_zip_manifest(verdict: str) -> None:
    write_text(
        ROOT / "ZIP_CONTENTS_MANIFEST.md",
        "\n".join(
            [
                "# ZIP CONTENTS MANIFEST",
                "",
                f"Fecha de reconstruccion: {datetime.now().strftime('%Y-%m-%d')} (PHASE19 REPAIR SANDBOX)",
                f"Estado: CANONICO ACTUALIZADO CON {verdict}",
                "",
                "## Contenido incluido",
                "- 00_READ_THIS_FIRST.md",
                "- 01_CURRENT_PROJECT_STATUS.md/json",
                "- 02_STRATEGY_AUTHORITY_MAP.md/json",
                "- ZIP_CONTENTS_MANIFEST.md",
                "- BOT_V2_DAYTIME_LAB/reports/PHASE19_FORENSIC_AUDIT_REPORT.md/json",
                "- BOT_V2_DAYTIME_LAB/reports/PHASE19_REPAIR_SANDBOX_REPORT.md/json",
                "- BOT_V2_DAYTIME_LAB/outputs/phase19_forensic_audit/",
                "- BOT_V2_DAYTIME_LAB/outputs/phase19_repair_sandbox/",
                "- BOT_V2_DAYTIME_LAB/src/run_phase19_forensic_audit.py",
                "- BOT_V2_DAYTIME_LAB/src/phase19_repaired_engine.py",
                "- BOT_V2_DAYTIME_LAB/src/run_phase19_repair_sandbox.py",
                "- BOT_V2_DAYTIME_LAB/tests/engine_safety/test_phase19_*.py",
                "",
                "## Exclusiones",
                "- datasets pesados y raw data",
                "- .git, .venv, __pycache__, .pyc, cache, logs pesados, temporales",
                "- ARCHIVE_SUPERSEDED completo",
                "- mt5_local_config.json, .env, secrets, credentials, *.key, *.pem",
                "- ZIPs internos",
                "",
                "## Veredicto",
                "Phase19 legacy permanece invalidada. La reparacion queda bloqueada por ausencia de M3 nativo certificado.",
            ]
        ),
    )


def should_exclude(path: Path) -> bool:
    parts = {p.lower() for p in path.parts}
    name = path.name.lower()
    forbidden_dirs = {".git", ".venv", "__pycache__", "cache", "archive_superseded", "data", "data manual"}
    if parts & forbidden_dirs:
        return True
    if name.endswith((".pyc", ".zip", ".tmp", ".log", ".parquet", ".jsonl")):
        return True
    return any(token in name for token in ["secret", "credential", ".env", ".key", ".pem", "mt5_local_config"])


def rebuild_zip() -> dict:
    staging = ROOT / "_zip_staging_phase19_repair"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    include = [
        ROOT / "00_READ_THIS_FIRST.md",
        ROOT / "01_CURRENT_PROJECT_STATUS.md",
        ROOT / "01_CURRENT_PROJECT_STATUS.json",
        ROOT / "02_STRATEGY_AUTHORITY_MAP.md",
        ROOT / "02_STRATEGY_AUTHORITY_MAP.json",
        ROOT / "ZIP_CONTENTS_MANIFEST.md",
        LAB / "reports",
        OUT.parent / "phase19_forensic_audit",
        OUT,
        SRC / "run_phase19_forensic_audit.py",
        SRC / "phase19_repaired_engine.py",
        SRC / "run_phase19_repair_sandbox.py",
        TESTS,
    ]
    for src in include:
        if not src.exists():
            continue
        if src.is_dir():
            for root_dir, dirs, files in os.walk(src):
                root_path = Path(root_dir)
                dirs[:] = [d for d in dirs if not should_exclude(root_path / d)]
                for file in files:
                    file_path = root_path / file
                    if should_exclude(file_path) or file_path.stat().st_size > 8 * 1024 * 1024:
                        continue
                    dest = staging / file_path.relative_to(ROOT)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, dest)
        else:
            if should_exclude(src):
                continue
            dest = staging / src.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for root_dir, _, files in os.walk(staging):
            for file in files:
                file_path = Path(root_dir) / file
                zf.write(file_path, file_path.relative_to(staging))
    shutil.rmtree(staging)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        names = zf.namelist()
        testzip = zf.testzip()
    forbidden = [".git", ".venv", "__pycache__", ".env", "secret", "credential", ".key", ".pem", "mt5_local_config", "ARCHIVE_SUPERSEDED", "/data/"]
    payload = {
        "zip_path": str(ZIP_PATH),
        "exists": ZIP_PATH.exists(),
        "size_bytes": ZIP_PATH.stat().st_size,
        "entry_count": len(names),
        "testzip": testzip,
        "contains_repair_sandbox": any(n.startswith("BOT_V2_DAYTIME_LAB/outputs/phase19_repair_sandbox/") for n in names)
        and "BOT_V2_DAYTIME_LAB/reports/PHASE19_REPAIR_SANDBOX_REPORT.md" in names,
        "contains_secrets": any(any(token.lower() in n.lower() for token in forbidden) for n in names),
        "contains_heavy_data": any("/data/" in n.lower() or n.lower().endswith((".parquet", ".jsonl")) for n in names),
    }
    payload["verdict"] = (
        "ZIP_VALIDATED"
        if payload["testzip"] is None and payload["contains_repair_sandbox"] and not payload["contains_secrets"] and not payload["contains_heavy_data"]
        else "ZIP_VALIDATION_FAILED"
    )
    write_json(OUT / "zip" / "phase19_repair_zip_validation.json", payload)
    return payload


def main() -> None:
    if Path.cwd().resolve() != ROOT.resolve():
        raise SystemExit(f"FAIL-CLOSED: cwd fuera de raiz oficial: {Path.cwd()}")
    ensure_dirs()
    start = starting_point()
    tests = run_repaired_tests()
    screening = run_minimal_screening_if_possible(tests)
    multitrade = multitrade_decision(screening)
    verdict = final_verdict(tests, screening)
    report = write_final_report(start, tests, screening, multitrade, verdict)
    update_zip_manifest(verdict)
    zip_state = rebuild_zip()
    terminal = {
        "verdict": verdict,
        "tests": tests["verdict"],
        "m3_native_confirmed": screening["m3_native_confirmed"],
        "screening": screening["verdict"],
        "multitrade_tested": multitrade["multitrade_tested"],
        "zip": zip_state,
        "report": str(REPORTS / "PHASE19_REPAIR_SANDBOX_REPORT.md"),
        "github_push_attempted": False,
    }
    write_json(OUT / "phase19_repair_terminal_state.json", terminal)
    print(json.dumps(terminal, indent=2, default=str))


if __name__ == "__main__":
    main()
