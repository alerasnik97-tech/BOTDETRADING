from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_coverage_pipeline_lib import INTAKE_ROOT, SPECS, coverage_report, ensure_pipeline_dirs, write_pipeline_heartbeat, write_status


def main() -> int:
    parser = argparse.ArgumentParser(description="Inicializa/inventaria intake manual de cobertura EURUSD sin descargar fuentes externas.")
    parser.add_argument("--target-date", required=True)
    args = parser.parse_args()

    ensure_pipeline_dirs()
    report = coverage_report(target_date=args.target_date, validator_integration=False)
    write_status(report)
    write_pipeline_heartbeat()

    print("[READY] Intake manual controlado:")
    for key in ("H1", "M5", "NEWS"):
        print(f"  - {key}: {SPECS[key].intake_path.relative_to(ROOT)}")
    print(f"[INFO] Colocar archivos nuevos en {INTAKE_ROOT.relative_to(ROOT)} y correr run_data_coverage_promotion.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
