from __future__ import annotations

import argparse
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo").resolve()
RUNNER_PATH = ROOT / "scratch" / "run_scbi_forward_phase1.py"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "scbi_phase1_dry_run"


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("scbi_forward_phase1_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar el runner oficial: {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_output_dir(target_date: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = DEFAULT_OUTPUT_ROOT / f"{target_date}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ejecuta un dry-run seguro de SCBI Phase 1 sin contaminar el runtime oficial."
    )
    parser.add_argument("--date", required=True, help="Fecha objetivo YYYY-MM-DD.")
    args = parser.parse_args()

    output_dir = _build_output_dir(args.date)
    ledger_path = output_dir / "SCBI_FORWARD_LEDGER_DRY_RUN.csv"
    status_path = output_dir / "SCBI_FORWARD_DAILY_STATUS_DRY_RUN.csv"

    runner = _load_runner_module()
    runner.LEDGER_CSV = str(ledger_path)
    runner.STATUS_CSV = str(status_path)
    runner.seal_runtime_state = lambda reason: (True, [])

    print(f"[DRY-RUN] Output dir: {output_dir}")
    print(f"[DRY-RUN] Ledger path: {ledger_path}")
    print(f"[DRY-RUN] Status path: {status_path}")
    runner.process_day(args.date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
