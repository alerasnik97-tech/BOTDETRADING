from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_coverage_pipeline_lib import coverage_report, write_status


def main() -> int:
    parser = argparse.ArgumentParser(description="Check canonico de cobertura H1/M5/news para EURUSD.")
    parser.add_argument("--target-date", required=True, help="Fecha objetivo en formato YYYY-MM-DD.")
    parser.add_argument("--prepared-root", default=None, help="Override para fixtures/auditoria; uso diario debe omitirlo.")
    parser.add_argument("--news-path", default=None, help="Override para fixtures/auditoria; uso diario debe omitirlo.")
    parser.add_argument("--coverage-only", action="store_true", help="Modo auditoria: no bloquea por rerun del validator diario.")
    parser.add_argument("--no-validator", action="store_true", help="Modo fixture: no integra el validator endurecido.")
    args = parser.parse_args()

    report = coverage_report(
        target_date=args.target_date,
        prepared_root=Path(args.prepared_root) if args.prepared_root else None,
        news_path=Path(args.news_path) if args.news_path else None,
        validator_integration=not args.no_validator,
        enforce_rerun_check=not args.coverage_only,
    )
    write_status(report)

    if report["coverage_blockers"]:
        for blocker in report["coverage_blockers"]:
            print(f"[BLOCK] {blocker}")
    validator = report["validator_integration"]
    if validator["enabled"] and validator["decision"] != "PASS":
        for blocker in validator["blockers"]:
            print(f"[BLOCK] VALIDATOR:{blocker}")

    if report["decision"] == "PASS":
        print(f"[PASS] {report['taxonomy_outcome']}: cobertura operable para {args.target_date}.")
        return 0
    print(f"[FAIL-CLOSED] {report['taxonomy_outcome']}: cobertura/daily gate no operable para {args.target_date}.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
