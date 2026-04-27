from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


CANONICAL_ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo").resolve()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ZIP = PROJECT_ROOT / "000_PARA_CHATGPT.zip"
MANIFEST_PATH = PROJECT_ROOT / "ZIP_CONTENTS_MANIFEST.md"
AUDIT_PATH = PROJECT_ROOT / "ZIP_PACKAGING_AUDIT.md"
BUILDER_STAGE_DIR = PROJECT_ROOT / "scripts" / ".bundle_build_tmp"
LEGACY_STAGE_DIR = PROJECT_ROOT / "__zip_stage"


CANONICAL_FILES: list[tuple[str, str]] = [
    ("CURRENT_STATE_OF_LAB.md", "Estado operativo vigente del laboratorio en STANDBY y benchmark H6."),
    ("MAPA_ESTRATEGIAS.md", "Mapa completo de estrategias y ranking de performance del laboratorio."),
    ("EURUSD_DESIGN_CONSTRAINT_BRIEF.md", "Restricciones de diseño y conocimiento negativo acumulado."),
    ("EURUSD_MANUAL_EDGE_FINAL_DECISION.md", "Veredicto final canonico de la linea manual-edge EURUSD."),
    ("CAMPAIGN_DECISION_C4.md", "Reporte oficial de decision y veredicto final de la Campaña 4."),
    ("EURUSD_CAMPAIGN_3B_FINAL_DECISION.md", "Reporte oficial de decision y veredicto final de la Campaña 3B."),
    ("EURUSD_MANUAL_ANNOTATION_LEDGER.csv", "Ledger consolidado final de 80 trades con trazabilidad completa."),
    ("EURUSD_MANUAL_ANNOTATION_ANALYSIS_RESULTS.csv", "Salida auditada del analisis consolidado sobre 80 trades."),
    ("EURUSD_MANUAL_ANNOTATION_SCHEMA.md", "Taxonomia cerrada usada para validar e interpretar las anotaciones humanas."),
    ("fast_signal_execution_dashboard.md", "Resumen operativo compacto del cierre manual-edge."),
    ("CODEX_LOCAL_SAFETY_PROTOCOL.md", "Protocolo local de seguridad y fail-closed."),
    ("CODEX_PRE_FLIGHT_CHECKLIST.md", "Checklist reutilizable para validar el perimetro y el workflow."),
    ("CODEX_EXECUTION_LOG_TEMPLATE.md", "Plantilla austera para trazabilidad futura de corridas locales."),
    ("CAMPAIGN_GATEKEEPER_PROTOCOL.md", "Protocolo maestro para la gestion, apertura y cierre de campañas."),
    ("CAMPAIGN_INTAKE_TEMPLATE.md", "Plantilla obligatoria para el inicio formal de nuevas campañas."),
    ("CAMPAIGN_FINAL_DECISION_TEMPLATE.md", "Plantilla estricta para el cierre y veredicto de campañas."),
    ("CAMPAIGN_OUTPUT_CONTRACT.md", "Contrato que define los artefactos minimos obligatorios de una campaña."),
    ("RESEARCH_DECISION_MATRIX.md", "Matriz de decision para estandarizar los veredictos de investigacion."),
    ("CAMPAIGN_EXECUTION_LOG_TEMPLATE.md", "Plantilla para el registro auditable de actividades de campaña."),
    ("scripts/preflight_project_boundary_check.py", "Preflight reutilizable para validar el perimetro."),
    ("scripts/campaign_preflight.py", "Preflight especifico para validar requisitos de campaña."),
    ("scripts/validate_campaign_outputs.py", "Validador de contratos de salida de campaña."),
    ("scripts/build_chatgpt_bundle.py", "Builder canonico endurecido para reconstruir desde cero el zip unico."),
    ("ZIP_CONTENTS_MANIFEST.md", "Inventario exacto del zip canonico."),
    ("ZIP_PACKAGING_AUDIT.md", "Auditoria del proceso de reconstruccion desde cero."),
    ("campaign4_results_dump.csv", "Evidencia fisica de los resultados del backtest multisegmento de la Campaña 4."),
    ("research_lab/strategies/eurusd_c4_ict_align.py", "Codigo de la estrategia ICT-ALIGN auditada en la Campaña 4."),
    ("scripts/campaign4_runner.py", "Ejecutor multisegmento usado para la validacion fisica de la Campaña 4."),
]

EXCLUSION_RULES = [
    "Se excluyen backups, archivos intermedios, staging y cualquier archivo con sufijo `_BACKUP_`.",
    "Se excluyen zips obsoletos de raiz distintos de `000_PARA_CHATGPT.zip`.",
    "Se excluyen scripts, codigo, datasets y resultados historicos que no son necesarios para entender el estado vigente y operar seguro dentro del proyecto.",
    "Se excluyen artefactos de etapas intermedias o handoffs ya superados.",
    "Se excluye todo archivo no nombrado explicitamente en la lista canonica.",
]


@dataclass(frozen=True)
class BuildStats:
    file_count: int
    raw_total_bytes: int
    previous_zip_bytes: int
    output_zip_bytes: int
    root_zip_count_before: int
    root_zip_count_after: int
    removed_root_zips: tuple[str, ...]


def _fail(message: str) -> None:
    raise RuntimeError(f"FAIL-CLOSED: {message}")


def _ensure_canonical_root() -> None:
    if PROJECT_ROOT.resolve() != CANONICAL_ROOT:
        _fail(f"root no canonico: {PROJECT_ROOT.resolve()}")
    if not CANONICAL_ROOT.exists() or not CANONICAL_ROOT.is_dir():
        _fail(f"root canonico inexistente: {CANONICAL_ROOT}")


def _ensure_within_project(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(CANONICAL_ROOT)
    except ValueError as exc:
        raise RuntimeError(f"FAIL-CLOSED: path fuera del proyecto: {resolved}") from exc
    return resolved


def _require_file(relative_path: str) -> Path:
    if "_BACKUP_" in relative_path:
        _fail(f"archivo backup no permitido en el bundle: {relative_path}")
    path = _ensure_within_project(PROJECT_ROOT / relative_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Falta archivo requerido para el bundle: {relative_path}")
    return path.resolve()


def _canonical_relative_paths() -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for relative_path, _ in CANONICAL_FILES:
        if relative_path in seen:
            _fail(f"duplicado logico en whitelist: {relative_path}")
        seen.add(relative_path)
        ordered.append(relative_path)
    return ordered


def _bundle_files() -> list[Path]:
    return [_require_file(relative_path) for relative_path in _canonical_relative_paths()]


def _root_level_extra_zips() -> list[Path]:
    return sorted(
        _ensure_within_project(path)
        for path in PROJECT_ROOT.glob("*.zip")
        if _ensure_within_project(path) != OUTPUT_ZIP.resolve()
    )


def _cleanup_temp_dirs() -> None:
    for directory in (BUILDER_STAGE_DIR, LEGACY_STAGE_DIR):
        target = _ensure_within_project(directory)
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)


def _remove_extra_root_zips() -> list[str]:
    removed: list[str] = []
    for path in _root_level_extra_zips():
        if path.exists():
            path.unlink()
            removed.append(path.name)
    return removed


def _manifest_text() -> str:
    lines = [
        "# ZIP Contents Manifest",
        "",
        f"Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Packaging Goal",
        "",
        "Bundle canonico unico, austero y coherente con el estado final vigente del laboratorio.",
        "",
        "## Included",
        "",
        "| Archivo | Motivo de inclusion |",
        "| --- | --- |",
    ]
    for relative_path, reason in CANONICAL_FILES:
        lines.append(f"| `{relative_path}` | {reason} |")

    lines.extend(["", "## Excluded", ""])
    for rule in EXCLUSION_RULES:
        lines.append(f"- {rule}")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Included file count: `{len(CANONICAL_FILES)}`",
            "- Canonical output: `000_PARA_CHATGPT.zip`",
            "- Scope: cierre manual-edge EURUSD + hardening del workflow local dentro del proyecto.",
        ]
    )
    return "\n".join(lines) + "\n"


def _audit_text(previous_zip_bytes: int, removed_root_zips: list[str]) -> str:
    removed_text = ", ".join(f"`{name}`" for name in removed_root_zips) if removed_root_zips else "ninguno"
    lines = [
        "# ZIP Packaging Audit",
        "",
        f"Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Rebuild Status",
        "",
        "- Reconstruccion desde cero: SI",
        "- Reemplazo completo del zip anterior: SI",
        f"- Zip previo detectado: `{previous_zip_bytes}` bytes",
        f"- Zips extra eliminados de la raiz: {removed_text}",
        "",
        "## Canonical Content Criterion",
        "",
        "- Se mantiene solo el set minimo vigente para entender el cierre manual-edge EURUSD y operar seguro dentro del proyecto.",
        "- Se conserva la referencia al benchmark H6 solo mediante el estado actual del laboratorio.",
        "- Se agregan solo los artefactos de seguridad local reutilizables necesarios para futuras corridas.",
        "",
        "## Exclusion Criterion Applied",
        "",
    ]
    for rule in EXCLUSION_RULES:
        lines.append(f"- {rule}")

    lines.extend(
        [
            "",
            "## Integrity Checks",
            "",
            f"- Archivos canonicos incluidos: `{len(CANONICAL_FILES)}`",
            "- Ausencia de duplicados logicos por nombre interno: SI",
            "- Ausencia de backups dentro del zip: SI",
            "- Ausencia de archivos intermedios dentro del zip: SI",
            "- Coherencia con estado final `NOT_TRANSLATABLE / STOP_AND_FREEZE`: SI",
            "- H6 preservado como benchmark vigente e intocable: SI",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_packaging_docs(previous_zip_bytes: int, removed_root_zips: list[str]) -> None:
    MANIFEST_PATH.write_text(_manifest_text(), encoding="utf-8")
    AUDIT_PATH.write_text(_audit_text(previous_zip_bytes, removed_root_zips), encoding="utf-8")


def _build_zip(bundle_files: list[Path]) -> int:
    _cleanup_temp_dirs()
    stage_dir = _ensure_within_project(BUILDER_STAGE_DIR)
    stage_dir.mkdir(parents=True, exist_ok=True)
    temp_zip = _ensure_within_project(stage_dir / OUTPUT_ZIP.name)
    try:
        with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in bundle_files:
                relative = path.resolve().relative_to(CANONICAL_ROOT).as_posix()
                archive.write(path, arcname=relative)
        size = temp_zip.stat().st_size
        os.replace(temp_zip, OUTPUT_ZIP)
        return size
    finally:
        _cleanup_temp_dirs()


def _verify_zip(bundle_files: list[Path]) -> None:
    expected_names = [path.resolve().relative_to(CANONICAL_ROOT).as_posix() for path in bundle_files]
    with zipfile.ZipFile(OUTPUT_ZIP, "r") as archive:
        names = archive.namelist()
        if len(names) != len(expected_names):
            _fail(f"conteo interno inesperado en zip: {len(names)} vs {len(expected_names)}")
        if len(set(names)) != len(names):
            _fail("duplicados internos detectados en el zip")
        if names != expected_names:
            _fail(f"orden o contenido inesperado en zip: {names}")
        forbidden = [name for name in names if "_BACKUP_" in name or name.endswith(".zip")]
        if forbidden:
            _fail(f"entradas prohibidas dentro del zip: {forbidden}")


def build_bundle(*, dry_run: bool = False) -> BuildStats:
    _ensure_canonical_root()
    previous_zip_bytes = OUTPUT_ZIP.stat().st_size if OUTPUT_ZIP.exists() else 0
    root_zip_count_before = len(list(PROJECT_ROOT.glob("*.zip")))

    if dry_run:
        bundle_files = _bundle_files()
        raw_total = sum(path.stat().st_size for path in bundle_files)
        return BuildStats(
            file_count=len(bundle_files),
            raw_total_bytes=raw_total,
            previous_zip_bytes=previous_zip_bytes,
            output_zip_bytes=previous_zip_bytes,
            root_zip_count_before=root_zip_count_before,
            root_zip_count_after=root_zip_count_before,
            removed_root_zips=tuple(path.name for path in _root_level_extra_zips()),
        )

    removed_root_zips = _remove_extra_root_zips()
    _write_packaging_docs(previous_zip_bytes, removed_root_zips)
    bundle_files = _bundle_files()
    raw_total_bytes = sum(path.stat().st_size for path in bundle_files)
    output_zip_bytes = _build_zip(bundle_files)
    _verify_zip(bundle_files)
    root_zip_count_after = len(list(PROJECT_ROOT.glob("*.zip")))
    if root_zip_count_after != 1:
        _fail(f"deberia quedar un solo zip en raiz y quedaron {root_zip_count_after}")

    return BuildStats(
        file_count=len(bundle_files),
        raw_total_bytes=raw_total_bytes,
        previous_zip_bytes=previous_zip_bytes,
        output_zip_bytes=output_zip_bytes,
        root_zip_count_before=root_zip_count_before,
        root_zip_count_after=root_zip_count_after,
        removed_root_zips=tuple(removed_root_zips),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruye desde cero el 000_PARA_CHATGPT.zip canonico.")
    parser.add_argument("--dry-run", action="store_true", help="Calcula el bundle sin reemplazar el zip.")
    args = parser.parse_args()

    result = build_bundle(dry_run=args.dry_run)
    print(f"files={result.file_count}")
    print(f"raw_total_bytes={result.raw_total_bytes}")
    print(f"previous_zip_bytes={result.previous_zip_bytes}")
    print(f"output_zip_bytes={result.output_zip_bytes}")
    print(f"root_zip_count_before={result.root_zip_count_before}")
    print(f"root_zip_count_after={result.root_zip_count_after}")
    print(f"removed_root_zips={','.join(result.removed_root_zips) if result.removed_root_zips else 'none'}")


if __name__ == "__main__":
    main()
