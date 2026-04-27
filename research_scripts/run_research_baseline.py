from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Añadir raíz del proyecto al path para importar external_scbi_research_harness
sys.path.insert(0, str(Path(__file__).parent.parent))

from external_scbi_research_harness.matrix import build_baseline_config
from external_scbi_research_harness.orchestrator import execute_variant, load_research_inputs, resolve_paths
from external_scbi_research_harness.reporting import write_baseline_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Corre la replica externa exacta del runner productivo actual.")
    parser.add_argument("--start-date", default="2020-01-01", help="Fecha inicial YYYY-MM-DD")
    parser.add_argument("--end-date", default="2025-12-31", help="Fecha final YYYY-MM-DD")
    parser.add_argument("--workspace-root", default=None, help="Raiz del workspace. Default: CWD del proyecto.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directorio de salida. Default: external_scbi_research_harness/outputs/baseline_truth_model",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = resolve_paths(args.workspace_root)
    output_dir = Path(args.output_dir) if args.output_dir else paths.output_root / "baseline_truth_model"

    print("[LOAD] Cargando datasets canonicos externos...")
    h1, m5, news = load_research_inputs(paths, start_date=args.start_date, end_date=args.end_date)

    config = build_baseline_config(args.start_date, args.end_date)
    print(f"[RUN] Ejecutando {config.variant_id} sobre {args.start_date} -> {args.end_date}")
    run_result, row = execute_variant(config, h1=h1, m5=m5, news=news)
    files = write_baseline_outputs(output_dir, config, run_result)

    print("[SUMMARY] Baseline truth model completado")
    print(f"  sample_size={row['sample_size']}")
    print(f"  win_rate={row['win_rate']}")
    print(f"  pf={row['pf']}")
    print(f"  expectancy={row['expectancy']}R")
    print(f"  max_drawdown={row['max_drawdown']}R")
    print(f"  sweeps_considered={run_result['stats'].get('sweeps_considered', 0)}")
    print(f"  trades_executed={run_result['stats'].get('trades_executed', 0)}")
    print("[FILES]")
    for label, path in files.items():
        print(f"  {label}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
