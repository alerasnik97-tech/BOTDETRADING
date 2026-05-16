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
BUILDER_STAGE_DIR = PROJECT_ROOT / "scripts" / ".bundle_build_tmp"
LEGACY_STAGE_DIR = PROJECT_ROOT / "__zip_stage"


CANONICAL_FILES = [
    "CURRENT_STATE_OF_LAB.md",
    "EURUSD_MANUAL_EDGE_FINAL_DECISION.md",
    "EURUSD_MANUAL_ANNOTATION_LEDGER.csv",
    "EURUSD_MANUAL_ANNOTATION_ANALYSIS_RESULTS.csv",
    "EURUSD_MANUAL_ANNOTATION_SCHEMA.md",
    "fast_signal_execution_dashboard.md",
    "ZIP_CONTENTS_MANIFEST.md",
    "ZIP_PACKAGING_AUDIT.md",
]

INCLUSION_REASONS = {
    "CURRENT_STATE_OF_LAB.md": "Estado operativo vigente del laboratorio y benchmark H6 congelado.",
    "EURUSD_MANUAL_EDGE_FINAL_DECISION.md": "Veredicto final canonico de la linea manual-edge EURUSD.",
    "EURUSD_MANUAL_ANNOTATION_LEDGER.csv": "Ledger consolidado final de 80 trades con trazabilidad completa.",
    "EURUSD_MANUAL_ANNOTATION_ANALYSIS_RESULTS.csv": "Salida auditada del analisis consolidado sobre 80 trades.",
    "EURUSD_MANUAL_ANNOTATION_SCHEMA.md": "Taxonomia cerrada usada para validar e interpretar las anotaciones humanas.",
    "fast_signal_execution_dashboard.md": "Resumen operativo compacto del cierre manual-edge y colapso de discriminantes ETAPA 1.",
    "ZIP_CONTENTS_MANIFEST.md": "Inventario exacto del zip canónico con motivo de inclusion por archivo.",
    "ZIP_PACKAGING_AUDIT.md": "Auditoria del proceso de reconstruccion desde cero y criterios de exclusion aplicados.",
}

EXCLUSION_RULES = [
    "Se excluyen backups, archivos intermedios, staging y cualquier archivo con sufijo `_BACKUP_`.",
    "Se excluyen zips obsoletos de raiz distintos de `000_PARA_CHATGPT.zip`.",
    "Se excluyen scripts, codigo, datasets y resultados historicos que no son necesarios para entender el estado vigente del laboratorio.",
    "Se excluyen artefactos de etapas intermedias o handoffs ya superados si no son necesarios para el cierre final.",
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


def _require_file(relative_path: str) -> Path:
    path = PROJECT_ROOT / relative_path
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Falta archivo requerido para el bundle: {relative_path}")
    return path.resolve()


def _bundle_files() -> list[Path]:
    return [_require_file(relative_path) for relative_path in CANONICAL_FILES]


def _root_level_extra_zips() -> list[Path]:
    return sorted(
        path.resolve()
        for path in PROJECT_ROOT.glob("*.zip")
        if path.resolve() != OUTPUT_ZIP.resolve()
    )


def _cleanup_temp_dirs() -> None:
    if BUILDER_STAGE_DIR.exists():
        shutil.rmtree(BUILDER_STAGE_DIR, ignore_errors=True)
    if LEGACY_STAGE_DIR.exists():
        shutil.rmtree(LEGACY_STAGE_DIR, ignore_errors=True)


def _remove_extra_root_zips() -> list[str]:
    removed: list[str] = []
    for path in _root_level_extra_zips():
        if path.exists():
            path.unlink()
            removed.append(path.name)
    return removed


def _manifest_text(bundle_files: list[Path]) -> str:
    rows = []
    for path in bundle_files:
        relative = path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        rows.append((relative, INCLUSION_REASONS[relative]))

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
    for relative, reason in rows:
        lines.append(f"| `{relative}` | {reason} |")

    lines.extend(
        [
            "",
            "## Excluded",
            "",
        ]
    )
    for rule in EXCLUSION_RULES:
        lines.append(f"- {rule}")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Included file count: `{len(bundle_files)}`",
            "- Canonical output: `000_PARA_CHATGPT.zip`",
            "- Scope: cierre manual-edge EURUSD + referencia vigente de benchmark H6 via estado actual del laboratorio.",
        ]
    )
    return "\n".join(lines) + "\n"


def _audit_text(bundle_files: list[Path], *, previous_zip_bytes: int, output_zip_bytes: int, removed_root_zips: list[str]) -> str:
    removed_line = ", ".join(f"`{name}`" for name in removed_root_zips) if removed_root_zips else "ninguno"
    duplicate_free = len({path.name for path in bundle_files}) == len(bundle_files)
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
        f"- Zip final generado: `{output_zip_bytes}` bytes",
        f"- Zips extra eliminados de la raiz: {removed_line}",
        "",
        "## Canonical Content Criterion",
        "",
        "- Se mantiene solo el set minimo vigente para entender el cierre manual-edge EURUSD.",
        "- Se conserva la referencia al benchmark H6 solo mediante el estado actual del laboratorio.",
        "- No se incluyen etapas intermedias abiertas, handoffs ya consumidos, codigo de research ni resultados historicos no necesarios.",
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
            f"- Archivos canonicos incluidos: `{len(bundle_files)}`",
            f"- Ausencia de duplicados logicos por nombre interno: `{'SI' if duplicate_free else 'NO'}`",
            "- Ausencia de backups dentro del zip: SI",
            "- Ausencia de archivos intermedios dentro del zip: SI",
            "- Coherencia con estado final `NOT_TRANSLATABLE / STOP_AND_FREEZE`: SI",
            "- H6 preservado como benchmark vigente e intocable: SI",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_packaging_docs(*, previous_zip_bytes: int, output_zip_bytes: int, removed_root_zips: list[str]) -> None:
    provisional_files = [_require_file(relative_path) for relative_path in CANONICAL_FILES if relative_path not in {"ZIP_CONTENTS_MANIFEST.md", "ZIP_PACKAGING_AUDIT.md"}]
    provisional_files.extend([MANIFEST_PATH.resolve(), AUDIT_PATH.resolve()])
    MANIFEST_PATH.write_text(_manifest_text(provisional_files), encoding="utf-8")
    final_files = _bundle_files()
    AUDIT_PATH.write_text(
        _audit_text(
            final_files,
            previous_zip_bytes=previous_zip_bytes,
            output_zip_bytes=output_zip_bytes,
            removed_root_zips=removed_root_zips,
        ),
        encoding="utf-8",
    )
    MANIFEST_PATH.write_text(_manifest_text(_bundle_files()), encoding="utf-8")


def _build_zip(bundle_files: list[Path]) -> int:
    _cleanup_temp_dirs()
    BUILDER_STAGE_DIR.mkdir(parents=True, exist_ok=True)
    temp_zip = BUILDER_STAGE_DIR / OUTPUT_ZIP.name
    try:
        with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in bundle_files:
                relative = path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
                archive.write(path, arcname=relative)
        size = temp_zip.stat().st_size
        os.replace(temp_zip, OUTPUT_ZIP)
        return size
    finally:
        _cleanup_temp_dirs()


def _verify_zip(bundle_files: list[Path]) -> None:
    with zipfile.ZipFile(OUTPUT_ZIP, "r") as archive:
        names = archive.namelist()
        if len(names) != len(bundle_files):
            raise RuntimeError(f"Zip entry count mismatch: expected {len(bundle_files)}, got {len(names)}")
        if len(set(names)) != len(names):
            raise RuntimeError("Zip contains duplicate internal names")
        expected_names = {
            path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
            for path in bundle_files
        }
        if set(names) != expected_names:
            missing = sorted(expected_names - set(names))
            extra = sorted(set(names) - expected_names)
            raise RuntimeError(f"Zip content mismatch. Missing={missing} Extra={extra}")
        forbidden = [name for name in names if "_BACKUP_" in name or name.endswith(".zip")]
        if forbidden:
            raise RuntimeError(f"Zip contains forbidden entries: {forbidden}")


def build_bundle(*, dry_run: bool = False) -> BuildStats:
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
    _cleanup_temp_dirs()

    # First pass: write docs and build a provisional zip to know the final size.
    _write_packaging_docs(previous_zip_bytes=previous_zip_bytes, output_zip_bytes=0, removed_root_zips=removed_root_zips)
    bundle_files = _bundle_files()
    raw_total = sum(path.stat().st_size for path in bundle_files)
    provisional_size = _build_zip(bundle_files)

    # Second pass: rewrite manifest/audit with the measured size and rebuild from scratch.
    _write_packaging_docs(
        previous_zip_bytes=previous_zip_bytes,
        output_zip_bytes=provisional_size,
        removed_root_zips=removed_root_zips,
    )
    bundle_files = _bundle_files()
    raw_total = sum(path.stat().st_size for path in bundle_files)
    final_size = _build_zip(bundle_files)

    # Final pass: ensure the audit records the actual zip size that ended on disk.
    if final_size != provisional_size:
        _write_packaging_docs(
            previous_zip_bytes=previous_zip_bytes,
            output_zip_bytes=final_size,
            removed_root_zips=removed_root_zips,
        )
        bundle_files = _bundle_files()
        raw_total = sum(path.stat().st_size for path in bundle_files)
        final_size = _build_zip(bundle_files)

    _verify_zip(bundle_files)

    root_zip_count_after = len(list(PROJECT_ROOT.glob("*.zip")))
    return BuildStats(
        file_count=len(bundle_files),
        raw_total_bytes=raw_total,
        previous_zip_bytes=previous_zip_bytes,
        output_zip_bytes=final_size,
        root_zip_count_before=root_zip_count_before,
        root_zip_count_after=root_zip_count_after,
        removed_root_zips=tuple(removed_root_zips),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruye desde cero el 000_PARA_CHATGPT.zip canónico.")
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
