from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ZIP = PROJECT_ROOT / "000_PARA_CHATGPT.zip"
MANIFEST_PATH = PROJECT_ROOT / "ZIP_CONTENTS_MANIFEST.md"
AUDIT_PATH = PROJECT_ROOT / "ZIP_PACKAGING_AUDIT.md"
ROOT_STAGE_DIR = PROJECT_ROOT / "__zip_stage"
BUILDER_STAGE_DIR = PROJECT_ROOT / "scripts" / ".bundle_build_tmp"


# Strict whitelist: if a file is not explicitly named here or added by the
# result-selection helpers below, it stays out of the bundle.
ROOT_DOCS = [
    "AGENTS.md",
    "PROJECT_CHARTER.md",
    "CURRENT_STATE_OF_LAB.md",
    "EURUSD_OOS_FINAL_VERDICT_2.0.md",
    "H6_PAPER_EXECUTION_FREEZE.md",
    "H6_PAPER_STAGE_DAY_0_STATUS.md",
    "H6_PAPER_STAGE_RUNBOOK.md",
    "H6_PAPER_STAGE_METRICS.md",
    "H6_PAPER_STAGE_OBSERVATION_PLAN.md",
    "H6_PAPER_STAGE_CHECKPOINT_5_SIGNALS.md",
    "H6_PAPER_STAGE_CHECKPOINT_20_SIGNALS.md",
    "H6_SPREAD_SLIPPAGE_CALIBRATION_AUDIT.md",
    "H6_FORWARD_ONLY_FREEZE.md",
    "H6_FORWARD_ONLY_GATE_PLAN.md",
    "H6_END_OF_MONTH_FORWARD_FREEZE.md",
    "H6_END_OF_MONTH_FORWARD_GATE.md",
    "RISK_PROTOCOL.md",
    "RESEARCH_OPERATING_SYSTEM.md",
    "AM_APPROVAL_FINAL_VERDICT.md",
    "ZIP_CONTENTS_MANIFEST.md",
]

LAB_STATE_FILES = [
    "research_lab/__init__.py",
    "research_lab/STRATEGY_MASTER_MATRIX.md",
    "research_lab/config.py",
    "research_lab/data_loader.py",
    "research_lab/engine.py",
    "research_lab/news_filter.py",
    "research_lab/report.py",
    "research_lab/ict_primitives.py",
    "research_lab/eurusd_am_post_news_external_liquidity_shift_runner.py",
    "results/H6_SHADOW_LEDGER_OFFICIAL.csv",
    "results/H6_SHADOW_LEDGER_OBSERVED.csv",
    "results/H6_RESEARCH_VS_SHADOW_OFFICIAL.csv",
    "results/H6_RESEARCH_VS_SHADOW_OBSERVED.csv",
    "results/H6_FORWARD_ONLY_DAILY_STATUS.csv",
    "results/H6_SHADOW_BLOCKED_SIGNALS_LOG.csv",
]


STRATEGY_FILES = [
    "research_lab/strategies/__init__.py",
    "research_lab/strategies/common.py",
    "research_lab/strategies/eurusd_am_post_news_external_liquidity_shift.py",
]


TEST_FILES = [
    "research_lab/tests/__init__.py",
    "research_lab/tests/test_h6_paper_shadow_runner.py",
]

DATA_FILES = [
    "data/news_eurusd_am_fortress_v3.csv",
    "data/news_eurusd_am_fortress_v3_summary.json",
    "data/news_eurusd_pm_research_safe.csv",
    "data/news_eurusd_pm_research_safe_summary.json",
]

TOOLING_FILES = [
    "scripts/build_chatgpt_bundle.py",
    "scripts/h6_paper_shadow_runner.py",
]

ICT_RESULT_ROOT_FILES = {
    "run_manifest.json",
    "ict_strategy_scorecard.csv",
}
ICT_RESULT_STRATEGY_FILES = {
    "summary.json",
    "period_summaries",
    "ranking.csv",
    "selected_params",
    "verdict",
}

PM_MICRO_ROOT_FILES = {
    "combo_ranking.csv",
    "period_overview.csv",
    "recomendacion_final.md",
}
PM_MICRO_STRATEGY_FILES = {
    "summary.json",
    "selected_params.json",
    "selection_metadata.json",
    "pm_safe_news_summary.json",
}

IFVG_RESULT_ROOT_FILES = {
    "run_manifest.json",
    "ifvg_strategy_scorecard.csv",
}
IFVG_RESULT_STRATEGY_FILES = {
    "summary.json",
    "period_summaries.json",
    "ranking.csv",
    "selected_params.json",
    "serious_gate.json",
    "verdict.json",
}

PM_VSE_RESULT_ROOT_FILES = {
    "run_manifest.json",
    "pm_safe_vse_scorecard.csv",
}
PM_VSE_RESULT_STRATEGY_FILES = {
    "summary.json",
    "period_summaries.json",
    "ranking.csv",
    "selected_params.json",
    "serious_gate.json",
    "verdict.json",
}

AM_SB_V2_RESULT_ROOT_FILES = {
    "run_manifest.json",
    "am_strategy_scorecard.csv",
}
AM_SB_V2_RESULT_STRATEGY_FILES = {
    "summary.json",
    "period_summaries.json",
    "ranking.csv",
    "selected_params.json",
    "serious_gate.json",
    "frame_contract.json",
    "verdict.json",
}

AM_ODR_RESULT_ROOT_FILES = {
    "run_manifest.json",
    "am_strategy_scorecard.csv",
}
AM_ODR_RESULT_STRATEGY_FILES = {
    "summary.json",
    "period_summaries.json",
    "ranking.csv",
    "selected_params.json",
    "serious_gate.json",
    "frame_contract.json",
    "verdict.json",
}

AM_ORET_RESULT_ROOT_FILES = {
    "run_manifest.json",
    "am_strategy_scorecard.csv",
}
AM_ORET_RESULT_STRATEGY_FILES = {
    "summary.json",
    "period_summaries.json",
    "ranking.csv",
    "selected_params.json",
    "serious_gate.json",
    "frame_contract.json",
    "verdict.json",
}

EURUSD_H1_M15_RESULT_ROOT_FILES = {
    "run_manifest.json",
    "eurusd_h1_m15_scorecard.csv",
}
EURUSD_H1_M15_RESULT_STRATEGY_FILES = {
    "summary.json",
    "period_summaries.json",
    "selected_params.json",
    "serious_gate.json",
    "verdict.json",
}

AM_ELS_RESULT_ROOT_FILES = {
    "run_manifest.json",
    "eurusd_am_post_news_external_liquidity_shift_scorecard.csv",
}
AM_ELS_RESULT_STRATEGY_FILES = {
    "summary.json",
    "period_summaries.json",
    "ranking.csv",
    "selected_params.json",
    "serious_gate.json",
    "frame_contract.json",
    "verdict.json",
}


EXCLUSION_RULES = [
    "Todo archivo no incluido en la whitelist explícita queda afuera.",
    "No se incluye `.git`, venvs, vendors, caches, staging, `__pycache__`, `.pytest_cache`, `.tmp`, `.pkg` ni artefactos del sistema.",
    "No se incluyen datasets pesados, data raw, `data_precision*`, ni árboles masivos de `data_*`.",
    "No se incluyen `legacy`, `legacy_archive_2026`, `reports` completos ni `results` completos.",
    "No se incluyen `equity_curve.csv`, resultados históricos enteros ni drilldowns innecesarios.",
    "No se incluyen scripts debug, runners obsoletos ni documentación superada si no son parte del estado vigente.",
]


@dataclass(frozen=True)
class BundleContext:
    ict_results_dir: Path
    pm_micro_dir: Path
    ifvg_results_dir: Path
    pm_vse_results_dir: Path
    am_silver_bullet_v2_dir: Path
    am_opening_drive_reversal_dir: Path
    am_opening_range_expansion_retest_dir: Path
    eurusd_h1_m15_liquidity_dir: Path
    am_els_dir: Path



def _bundle_related_root_zips() -> list[Path]:
    return sorted(
        path.resolve()
        for path in PROJECT_ROOT.glob("*.zip")
        if path.name.startswith("000_PARA_CHATGPT") and path.resolve() != OUTPUT_ZIP.resolve()
    )


def _legacy_root_temp_dirs() -> list[Path]:
    return sorted(
        path.resolve()
        for path in PROJECT_ROOT.glob("bundle_build_*")
        if path.is_dir()
    )


def _cleanup_root_residue() -> None:
    if ROOT_STAGE_DIR.exists():
        shutil.rmtree(ROOT_STAGE_DIR, ignore_errors=True)
    for temp_dir in _legacy_root_temp_dirs():
        shutil.rmtree(temp_dir, ignore_errors=True)
    for zip_path in _bundle_related_root_zips():
        if zip_path.exists():
            zip_path.unlink()


def _cleanup_builder_stage() -> None:
    if BUILDER_STAGE_DIR.exists():
        shutil.rmtree(BUILDER_STAGE_DIR, ignore_errors=True)


def _latest_child_dir(relative_dir: str) -> Path:
    base = PROJECT_ROOT / relative_dir
    candidates = sorted((item for item in base.iterdir() if item.is_dir()), key=lambda item: item.name)
    if not candidates:
        raise FileNotFoundError(f"No encontré carpetas dentro de {relative_dir}")
    return candidates[-1]


def _require_file(relative_path: str) -> Path:
    path = PROJECT_ROOT / relative_path
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Falta archivo requerido para el bundle: {relative_path}")
    return path.resolve()


def _bundle_context() -> BundleContext:
    return BundleContext(
        ict_results_dir=_latest_child_dir("results/ict_atomic_setups").resolve(),
        pm_micro_dir=_latest_child_dir("results/pm_micro_reclaim_m3").resolve(),
        ifvg_results_dir=_latest_child_dir("results/ifvg_repricing_pm").resolve(),
        pm_vse_results_dir=_latest_child_dir("results/pm_volatility_squeeze_retest_m5").resolve(),
        am_silver_bullet_v2_dir=_latest_child_dir("results/am_silver_bullet_ny_v2").resolve(),
        am_opening_drive_reversal_dir=_latest_child_dir("results/am_opening_drive_reversal").resolve(),
        am_opening_range_expansion_retest_dir=_latest_child_dir("results/am_opening_range_expansion_retest").resolve(),
        eurusd_h1_m15_liquidity_dir=_latest_child_dir("results/eurusd_h1_liquidity_sweep_m15").resolve(),
        am_els_dir=_latest_child_dir("results/eurusd_am_post_news_external_liquidity_shift").resolve(),
    )



def _collect_static_files() -> list[Path]:
    relative_paths = ROOT_DOCS + LAB_STATE_FILES + STRATEGY_FILES + TEST_FILES + DATA_FILES + TOOLING_FILES
    return [_require_file(relative_path) for relative_path in relative_paths]


def _collect_ict_result_files(context: BundleContext) -> list[Path]:
    files: list[Path] = []
    for file_name in sorted(ICT_RESULT_ROOT_FILES):
        path = context.ict_results_dir / file_name
        if path.exists() and path.is_file():
            files.append(path.resolve())
    for strategy_dir in sorted((item for item in context.ict_results_dir.iterdir() if item.is_dir()), key=lambda item: item.name):
        for candidate in sorted(strategy_dir.iterdir(), key=lambda item: item.name):
            if candidate.is_file() and candidate.name in ICT_RESULT_STRATEGY_FILES:
                files.append(candidate.resolve())
    return files


def _collect_pm_micro_result_files(context: BundleContext) -> list[Path]:
    files: list[Path] = []
    for file_name in sorted(PM_MICRO_ROOT_FILES):
        path = context.pm_micro_dir / file_name
        if path.exists() and path.is_file():
            files.append(path.resolve())
    for strategy_dir in sorted((item for item in context.pm_micro_dir.iterdir() if item.is_dir()), key=lambda item: item.name):
        for candidate in sorted(strategy_dir.iterdir(), key=lambda item: item.name):
            if candidate.is_file() and candidate.name in PM_MICRO_STRATEGY_FILES:
                files.append(candidate.resolve())
    return files


def _collect_ifvg_result_files(context: BundleContext) -> list[Path]:
    files: list[Path] = []
    ifvg_dir = context.ifvg_results_dir
    for file_name in sorted(IFVG_RESULT_ROOT_FILES):
        path = ifvg_dir / file_name
        if path.exists() and path.is_file():
            files.append(path.resolve())
    for strategy_dir in sorted((item for item in ifvg_dir.iterdir() if item.is_dir()), key=lambda item: item.name):
        for candidate in sorted(strategy_dir.iterdir(), key=lambda item: item.name):
            if candidate.is_file() and candidate.name in IFVG_RESULT_STRATEGY_FILES:
                files.append(candidate.resolve())
    return files


def _collect_pm_vse_result_files(context: BundleContext) -> list[Path]:
    files: list[Path] = []
    pm_vse_dir = context.pm_vse_results_dir
    for file_name in sorted(PM_VSE_RESULT_ROOT_FILES):
        path = pm_vse_dir / file_name
        if path.exists() and path.is_file():
            files.append(path.resolve())
    for strategy_dir in sorted((item for item in pm_vse_dir.iterdir() if item.is_dir()), key=lambda item: item.name):
        for candidate in sorted(strategy_dir.iterdir(), key=lambda item: item.name):
            if candidate.is_file() and candidate.name in PM_VSE_RESULT_STRATEGY_FILES:
                files.append(candidate.resolve())
    return files


def _collect_am_sb_v2_result_files(context: BundleContext) -> list[Path]:
    files: list[Path] = []
    result_dir = context.am_silver_bullet_v2_dir
    for file_name in sorted(AM_SB_V2_RESULT_ROOT_FILES):
        path = result_dir / file_name
        if path.exists() and path.is_file():
            files.append(path.resolve())
    for strategy_dir in sorted((item for item in result_dir.iterdir() if item.is_dir()), key=lambda item: item.name):
        for candidate in sorted(strategy_dir.iterdir(), key=lambda item: item.name):
            if candidate.is_file() and candidate.name in AM_SB_V2_RESULT_STRATEGY_FILES:
                files.append(candidate.resolve())
    return files


def _collect_am_odr_result_files(context: BundleContext) -> list[Path]:
    files: list[Path] = []
    result_dir = context.am_opening_drive_reversal_dir
    for file_name in sorted(AM_ODR_RESULT_ROOT_FILES):
        path = result_dir / file_name
        if path.exists() and path.is_file():
            files.append(path.resolve())
    for strategy_dir in sorted((item for item in result_dir.iterdir() if item.is_dir()), key=lambda item: item.name):
        for candidate in sorted(strategy_dir.iterdir(), key=lambda item: item.name):
            if candidate.is_file() and candidate.name in AM_ODR_RESULT_STRATEGY_FILES:
                files.append(candidate.resolve())
    return files


def _collect_am_oret_result_files(context: BundleContext) -> list[Path]:
    files: list[Path] = []
    result_dir = context.am_opening_range_expansion_retest_dir
    for file_name in sorted(AM_ORET_RESULT_ROOT_FILES):
        path = result_dir / file_name
        if path.exists() and path.is_file():
            files.append(path.resolve())
    for strategy_dir in sorted((item for item in result_dir.iterdir() if item.is_dir()), key=lambda item: item.name):
        for candidate in sorted(strategy_dir.iterdir(), key=lambda item: item.name):
            if candidate.is_file() and candidate.name in AM_ORET_RESULT_STRATEGY_FILES:
                files.append(candidate.resolve())
    return files


def _collect_eurusd_h1_m15_result_files(context: BundleContext) -> list[Path]:
    files: list[Path] = []
    result_dir = context.eurusd_h1_m15_liquidity_dir
    for file_name in sorted(EURUSD_H1_M15_RESULT_ROOT_FILES):
        path = result_dir / file_name
        if path.exists() and path.is_file():
            files.append(path.resolve())
    for strategy_dir in sorted((item for item in result_dir.iterdir() if item.is_dir()), key=lambda item: item.name):
        for candidate in sorted(strategy_dir.iterdir(), key=lambda item: item.name):
            if candidate.is_file() and candidate.name in EURUSD_H1_M15_RESULT_STRATEGY_FILES:
                files.append(candidate.resolve())
    return files


def _collect_am_els_result_files(context: BundleContext) -> list[Path]:
    files: list[Path] = []
    result_dir = context.am_els_dir
    for file_name in sorted(AM_ELS_RESULT_ROOT_FILES):
        path = result_dir / file_name
        if path.exists() and path.is_file():
            files.append(path.resolve())
    for strategy_dir in sorted((item for item in result_dir.iterdir() if item.is_dir()), key=lambda item: item.name):
        for candidate in sorted(strategy_dir.iterdir(), key=lambda item: item.name):
            if candidate.is_file() and candidate.name in AM_ELS_RESULT_STRATEGY_FILES:
                files.append(candidate.resolve())
    return files


def _relative_to_root(path: Path) -> str:

    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _dedupe(paths: list[Path]) -> list[Path]:
    deduped: dict[Path, None] = {}
    for path in paths:
        deduped[path.resolve()] = None
    return sorted(deduped.keys())


def _collect_bundle_files(context: BundleContext, include_manifest: bool) -> list[Path]:
    files = _collect_static_files()
    # Solo incluimos resultados de ELS (H6) para el core canónico
    files.extend(_collect_am_els_result_files(context))

    if include_manifest:
        files.append(MANIFEST_PATH.resolve())
    return _dedupe(files)


def _manifest_text(context: BundleContext, bundle_files: list[Path]) -> str:
    sections: dict[str, list[str]] = {
        "Root Documents": [],
        "Lab State Code": [],
        "Strategies": [],
        "Tests": [],
        "Operational Data": [],
        "Tooling": [],
        "Selected AM Post-News Liquidity Shift Results (H6)": [],
    }


    ict_prefix = _relative_to_root(context.ict_results_dir)
    micro_prefix = _relative_to_root(context.pm_micro_dir)
    ifvg_prefix = _relative_to_root(context.ifvg_results_dir)
    pm_vse_prefix = _relative_to_root(context.pm_vse_results_dir)
    am_sb_v2_prefix = _relative_to_root(context.am_silver_bullet_v2_dir)
    am_odr_prefix = _relative_to_root(context.am_opening_drive_reversal_dir)
    am_oret_prefix = _relative_to_root(context.am_opening_range_expansion_retest_dir)
    am_els_prefix = _relative_to_root(context.am_els_dir)
    eurusd_h1_m15_prefix = _relative_to_root(context.eurusd_h1_m15_liquidity_dir)


    for path in bundle_files:
        relative = _relative_to_root(path)
        if relative in ROOT_DOCS:
            sections["Root Documents"].append(relative)
        elif relative in LAB_STATE_FILES:
            sections["Lab State Code"].append(relative)
        elif relative in STRATEGY_FILES:
            sections["Strategies"].append(relative)
        elif relative in TEST_FILES:
            sections["Tests"].append(relative)
        elif relative in DATA_FILES:
            sections["Operational Data"].append(relative)
        elif relative in TOOLING_FILES:
            sections["Tooling"].append(relative)
        elif relative.startswith(am_els_prefix + "/"):
            sections["Selected AM Post-News Liquidity Shift Results (H6)"].append(relative)

    total_bytes = sum(path.stat().st_size for path in bundle_files if path.exists())
    lines = [
        "# ZIP Contents Manifest",
        "",
        f"Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Packaging Goal",
        "",
        "Este zip se construye con whitelist estricta.",
        "No es un backup bruto del repo completo.",
        "",
        "## Included",
        "",
    ]
    for title, entries in sections.items():
        if not entries:
            continue
        lines.append(f"### {title}")
        lines.append("")
        for entry in entries:
            lines.append(f"- `{entry}`")
        lines.append("")

    lines.extend(["## Excluded", ""])
    for rule in EXCLUSION_RULES:
        lines.append(f"- {rule}")
    lines.extend(
        [
            "",
            "## Whitelist Policy",
            "",
            "- Si un archivo no está nombrado en la whitelist del builder o en los selectores de resultados mínimos, no entra.",
            "- Los resultados incluidos se podan a resúmenes y selección mínima; no viajan árboles completos.",
            "- El proceso final deja un único zip visible en la raíz: `000_PARA_CHATGPT.zip`.",
            "",
            "## Current Result Sources",
            "",
            f"- Latest AM Post-News Liquidity Shift result folder selected (H6): `{am_els_prefix}`",

            "",
            "## Summary",
            "",
            f"- Included file count: `{len(bundle_files)}`",
            f"- Included raw bytes before compression: `{total_bytes}`",
            f"- Output zip target: `{OUTPUT_ZIP.name}`",
            "- Visible root zip count expected after build: `1`",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_manifest(context: BundleContext) -> None:
    preview_files = _collect_bundle_files(context, include_manifest=False)
    preview_with_manifest = _dedupe(preview_files + [MANIFEST_PATH.resolve()])
    MANIFEST_PATH.write_text(_manifest_text(context, preview_with_manifest), encoding="utf-8")


def build_bundle(*, dry_run: bool = False) -> dict[str, int | str]:
    context = _bundle_context()
    _write_manifest(context)
    bundle_files = _collect_bundle_files(context, include_manifest=True)

    previous_size = OUTPUT_ZIP.stat().st_size if OUTPUT_ZIP.exists() else 0
    raw_total = sum(path.stat().st_size for path in bundle_files)
    root_zip_count_before = len(sorted(PROJECT_ROOT.glob("*.zip")))
    extra_root_zips_before = len(_bundle_related_root_zips())
    had_root_stage_before = ROOT_STAGE_DIR.exists()
    legacy_temp_dirs_before = len(_legacy_root_temp_dirs())

    if dry_run:
        return {
            "file_count": len(bundle_files),
            "raw_total_bytes": raw_total,
            "previous_zip_bytes": previous_size,
            "output_zip_bytes": previous_size,
            "root_zip_count_before": root_zip_count_before,
            "extra_root_zips_before": extra_root_zips_before,
            "had_root_stage_before": int(had_root_stage_before),
            "legacy_temp_dirs_before": legacy_temp_dirs_before,
        }

    _cleanup_builder_stage()
    BUILDER_STAGE_DIR.mkdir(parents=True, exist_ok=True)
    temp_zip = BUILDER_STAGE_DIR / OUTPUT_ZIP.name
    try:
        with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in bundle_files:
                archive.write(path, arcname=_relative_to_root(path))
        new_size = temp_zip.stat().st_size
        os.replace(temp_zip, OUTPUT_ZIP)
    finally:
        _cleanup_builder_stage()

    _cleanup_root_residue()
    root_zip_count_after = len(sorted(PROJECT_ROOT.glob("*.zip")))
    return {
        "file_count": len(bundle_files),
        "raw_total_bytes": raw_total,
        "previous_zip_bytes": previous_size,
        "output_zip_bytes": new_size,
        "root_zip_count_before": root_zip_count_before,
        "extra_root_zips_before": extra_root_zips_before,
        "had_root_stage_before": int(had_root_stage_before),
        "legacy_temp_dirs_before": legacy_temp_dirs_before,
        "root_zip_count_after": root_zip_count_after,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye el 000_PARA_CHATGPT.zip canónico usando whitelist estricta y deja un solo zip en la raíz.")
    parser.add_argument("--dry-run", action="store_true", help="Calcula el bundle sin reemplazar el zip.")
    args = parser.parse_args()

    result = build_bundle(dry_run=args.dry_run)
    print(f"files={result['file_count']}")
    print(f"raw_total_bytes={result['raw_total_bytes']}")
    print(f"previous_zip_bytes={result['previous_zip_bytes']}")
    print(f"output_zip_bytes={result['output_zip_bytes']}")
    print(f"root_zip_count_before={result['root_zip_count_before']}")
    print(f"extra_root_zips_before={result['extra_root_zips_before']}")
    print(f"had_root_stage_before={result['had_root_stage_before']}")
    print(f"legacy_temp_dirs_before={result['legacy_temp_dirs_before']}")
    if not args.dry_run:
        print(f"root_zip_count_after={result['root_zip_count_after']}")


if __name__ == "__main__":
    main()
