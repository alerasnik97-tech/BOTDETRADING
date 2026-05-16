from __future__ import annotations

import argparse
import sys
from pathlib import Path


CANONICAL_ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo").resolve()
CRITICAL_FILES = [
    "000_PARA_CHATGPT.zip",
    "scripts/build_chatgpt_bundle.py",
    "CURRENT_STATE_OF_LAB.md",
    "EURUSD_MANUAL_EDGE_FINAL_DECISION.md",
    "ZIP_CONTENTS_MANIFEST.md",
    "ZIP_PACKAGING_AUDIT.md",
]


def _resolve_target(root: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-flight local boundary check for the canonical trading workspace.")
    parser.add_argument("--root", default=str(CANONICAL_ROOT), help="Expected project root.")
    parser.add_argument("--path", dest="paths", action="append", default=[], help="Path to validate against the project root.")
    parser.add_argument("--skip-critical-files", action="store_true", help="Skip critical file existence checks.")
    args = parser.parse_args()

    requested_root = Path(args.root).resolve(strict=False)
    errors: list[str] = []

    if requested_root != CANONICAL_ROOT:
        errors.append(f"Root no canonico: {requested_root}")

    if not CANONICAL_ROOT.exists() or not CANONICAL_ROOT.is_dir():
        errors.append(f"Root canonico inexistente: {CANONICAL_ROOT}")

    cwd = Path.cwd().resolve(strict=False)
    if not _is_within(cwd, CANONICAL_ROOT):
        errors.append(f"CWD fuera del proyecto: {cwd}")

    checked_paths = args.paths or [str(CANONICAL_ROOT)]
    for raw_path in checked_paths:
        resolved = _resolve_target(CANONICAL_ROOT, raw_path)
        if not _is_within(resolved, CANONICAL_ROOT):
            errors.append(f"Path fuera del root permitido: {raw_path} -> {resolved}")

    if not args.skip_critical_files:
        for relative_path in CRITICAL_FILES:
            target = (CANONICAL_ROOT / relative_path).resolve(strict=False)
            if not target.exists() or not target.is_file():
                errors.append(f"Falta archivo critico: {relative_path}")

    print("=" * 60)
    print("CODEX PROJECT BOUNDARY CHECK")
    print("=" * 60)
    print(f"Canonical root: {CANONICAL_ROOT}")
    print(f"Current cwd: {cwd}")
    print("Checked paths:")
    for raw_path in checked_paths:
        print(f"  - {raw_path}")

    if errors:
        print("\nSTATUS: FAIL-CLOSED")
        for error in errors:
            print(f"  [ERROR] {error}")
        return 2

    print("\nSTATUS: PASS")
    print("  Todas las rutas auditadas permanecen dentro del proyecto canonico.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
