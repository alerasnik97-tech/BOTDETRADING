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

from data_coverage_pipeline_lib import (
    RESULTS_DIR,
    coverage_report,
    now_utc_iso,
    promote_dataset,
    write_pipeline_heartbeat,
    write_status,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Promocion append-only de intake H1/M5/news hacia data canonica EURUSD.")
    parser.add_argument("--dataset", choices=["H1", "M5", "NEWS", "ALL"], required=True)
    parser.add_argument("--target-date", required=True, help="Fecha que se desea habilitar con la cobertura promovida.")
    parser.add_argument("--promote", action="store_true", help="Sin este flag, solo hace dry-run promocionable.")
    parser.add_argument("--intake-path", default=None, help="Override para un unico dataset.")
    parser.add_argument("--canonical-path", default=None, help="Override para fixtures/auditoria de un unico dataset.")
    args = parser.parse_args()

    datasets = ["H1", "M5", "NEWS"] if args.dataset == "ALL" else [args.dataset]
    results = []
    for dataset in datasets:
        if args.dataset == "ALL" and (args.intake_path or args.canonical_path):
            raise SystemExit("--intake-path/--canonical-path solo se permiten con un dataset unico.")
        result = promote_dataset(
            kind=dataset,
            intake_path=Path(args.intake_path) if args.intake_path else None,
            canonical_path=Path(args.canonical_path) if args.canonical_path else None,
            promote=args.promote,
            target_date=args.target_date,
        )
        results.append(result)
        print(f"[{result['status']}] {dataset}: {result['reason']} rows={result['rows_appended']}")

    using_overrides = bool(args.intake_path or args.canonical_path)
    if using_overrides:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        override_report = {
            "generated_at_utc": now_utc_iso(),
            "mode": "OVERRIDE_FIXTURE_OR_AUDIT",
            "target_date": args.target_date,
            "promotion_results": results,
        }
        (RESULTS_DIR / "promotion_override_last.json").write_text(
            json.dumps(override_report, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        report = {"decision": "SKIPPED", "taxonomy_outcome": "OVERRIDE_FIXTURE_OR_AUDIT"}
    else:
        report = coverage_report(target_date=args.target_date)
        report["promotion_results"] = results
        write_status(report)
        write_pipeline_heartbeat()

    blocked = [item for item in results if item["status"] == "BLOCK"]
    if blocked:
        print("[FAIL-CLOSED] DATA_PROMOTION_BLOCKED")
        return 1
    if report["decision"] == "PASS":
        print(f"[PASS] {report['taxonomy_outcome']}: cobertura promovida y validator integrado para {args.target_date}.")
        return 0
    print(f"[INFO] Promocion evaluada; daily gate actual: {report['decision']} ({report['taxonomy_outcome']}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
