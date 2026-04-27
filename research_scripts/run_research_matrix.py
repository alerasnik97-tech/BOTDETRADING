from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Añadir raíz del proyecto al path para importar external_scbi_research_harness
sys.path.insert(0, str(Path(__file__).parent.parent))

from external_scbi_research_harness.matrix import build_variants
from external_scbi_research_harness.orchestrator import execute_variant_matrix, load_research_inputs, resolve_paths
from external_scbi_research_harness.reporting import write_matrix_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Corre matriz externa de research sobre la baseline real actual.")
    parser.add_argument("--start-date", default="2020-01-01", help="Fecha inicial YYYY-MM-DD")
    parser.add_argument("--end-date", default="2025-12-31", help="Fecha final YYYY-MM-DD")
    parser.add_argument(
        "--profile",
        default="axis_scan",
        choices=("axis_scan", "full_factorial"),
        help="Perfil de variantes a ejecutar",
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=None,
        help="Limite opcional de variantes para smoke run o cortes controlados",
    )
    parser.add_argument("--workspace-root", default=None, help="Raiz del workspace. Default: CWD del proyecto.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directorio de salida. Default: external_scbi_research_harness/outputs/matrix_<profile>",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = resolve_paths(args.workspace_root)
    output_dir = Path(args.output_dir) if args.output_dir else paths.output_root / f"matrix_{args.profile}"

    variants = build_variants(
        start_date=args.start_date,
        end_date=args.end_date,
        profile=args.profile,
        max_variants=args.max_variants,
    )
    print(f"[PLAN] Variantes a ejecutar: {len(variants)} | perfil={args.profile}")

    print("[LOAD] Cargando datasets canonicos externos...")
    h1, m5, news = load_research_inputs(paths, start_date=args.start_date, end_date=args.end_date)

    ranked_results, _ = execute_variant_matrix(variants, h1=h1, m5=m5, news=news)
    files = write_matrix_outputs(output_dir, ranked_results=ranked_results, profile=args.profile)

    print("[SUMMARY] Matriz completada")
    print(f"  variant_count={len(ranked_results)}")
    if not ranked_results.empty:
        top = ranked_results.iloc[0]
        print(f"  top_variant={top['variant_id']}")
        print(f"  ranking_score={top['ranking_score']}")
        print(f"  sample_size={top['sample_size']}")
        print(f"  pf={top['pf']}")
        print(f"  expectancy={top['expectancy']}R")
        print(f"  max_drawdown={top['max_drawdown']}R")
    print("[FILES]")
    for label, path in files.items():
        print(f"  {label}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
