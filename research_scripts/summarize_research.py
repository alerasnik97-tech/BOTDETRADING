from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Añadir raíz del proyecto al path para importar external_scbi_research_harness
sys.path.insert(0, str(Path(__file__).parent.parent))

from external_scbi_research_harness.orchestrator import resolve_paths
from external_scbi_research_harness.reporting import summarize_existing_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-rankea y resume un CSV ya generado por la matriz externa.")
    parser.add_argument("--workspace-root", default=None, help="Raiz del workspace. Default: CWD del proyecto.")
    parser.add_argument(
        "--results-csv",
        default=None,
        help="CSV fuente. Default: external_scbi_research_harness/outputs/matrix_axis_scan/research_matrix_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directorio de salida. Default: carpeta sibling summary_refresh",
    )
    parser.add_argument("--profile-label", default="existing_results", help="Etiqueta descriptiva para el resumen regenerado")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = resolve_paths(args.workspace_root)
    default_results = paths.output_root / "matrix_axis_scan" / "research_matrix_results.csv"
    results_csv = Path(args.results_csv) if args.results_csv else default_results
    if not results_csv.exists():
        raise FileNotFoundError(f"No existe el CSV de resultados: {results_csv}")

    output_dir = Path(args.output_dir) if args.output_dir else results_csv.parent / "summary_refresh"
    files = summarize_existing_results(results_csv, output_dir=output_dir, profile=args.profile_label)

    ranked = pd.read_csv(results_csv)
    print("[SUMMARY] Resumen regenerado")
    print(f"  source={results_csv}")
    print(f"  rows={len(ranked)}")
    print("[FILES]")
    for label, path in files.items():
        print(f"  {label}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
