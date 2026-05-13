"""EOM blocker official single-ZIP rebuild.

Builds only the root 000_PARA_CHATGPT.zip, with explicit inclusion for the
MANIPULANTE 3.0 EOM-fix evidence and hard exclusions for raw data, caches,
virtualenvs, git internals, backups, and nested ZIP files.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path


BASE = Path(__file__).resolve().parents[3]
OUT = BASE / "000_PARA_CHATGPT.zip"
LOCK_DIR = BASE / "06_GOVERNANCE_AND_COMPLIANCE" / "artifact_delivery" / "single_zip_delivery_lock"

SKIP_EXT = {
    ".parquet",
    ".h5",
    ".feather",
    ".bin",
    ".zip",
    ".7z",
    ".rar",
    ".pyc",
    ".pyo",
    ".bundle",
}

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "venv",
    "venv_v37",
    "cache",
    "caches",
    "_chatgpt_export",
    "bot_market_data",
    "data",
    "manual_data",
    "tick",
    "ticks",
    "outputs",
    "backups",
    "configs",
    "cold_storage",
    "quarantines",
    "old_deliveries",
    "zipped_snapshots",
    "git_bundles",
    "restore_points",
    "usdjpy_2016_2019",
    "usdjpy_2016_2021",
    "usdjpy_2022_2025",
    "data_manual",
    "archivo_historico",
    "_archivo_historico_no_subir",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def should_skip(path: Path) -> bool:
    rel = path.relative_to(BASE)
    parts = [p.lower() for p in rel.parts]
    if any(part in SKIP_DIRS for part in parts[:-1]):
        return True
    if path.name in {
        "EOMFIX_SINGLE_ZIP_FINAL_VERIFICATION.txt",
        "EOMFIX_SINGLE_ZIP_REBUILD_OUTPUT.txt",
    }:
        return True
    if path.suffix.lower() in SKIP_EXT:
        return True
    if "subir_a_chatgpt" in path.name.lower():
        return True
    return False


def add_tree(files: dict[str, Path], root: Path) -> None:
    if not root.exists():
        return
    for current, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIRS]
        for name in names:
            fp = Path(current) / name
            if should_skip(fp):
                continue
            rel = fp.relative_to(BASE).as_posix()
            files.setdefault(rel, fp)


def add_file(files: dict[str, Path], path: Path) -> None:
    if path.exists() and not should_skip(path):
        files.setdefault(path.relative_to(BASE).as_posix(), path)


def collect_files() -> dict[str, Path]:
    files: dict[str, Path] = {}
    add_tree(files, BASE / "06_GOVERNANCE_AND_COMPLIANCE")
    add_tree(
        files,
        BASE
        / "03_RESEARCH_LAB"
        / "BOT_V2_DAYTIME_LAB"
        / "reports"
        / "v38_manipulante3_htf_ltf",
    )
    add_tree(files, BASE / "03_RESEARCH_LAB" / "BOT_V2_DAYTIME_LAB" / "src" / "v7_engine")
    add_file(files, BASE / "03_RESEARCH_LAB" / "BOT_V2_DAYTIME_LAB" / "gate6_mini_runner.py")
    add_file(files, BASE / "03_RESEARCH_LAB" / "BOT_V2_DAYTIME_LAB" / "run_manipulante3_pilot.py")
    for folder in [
        "01_CORE_PRODUCTION",
        "02_INCUBATION_STAGING",
        "03_RESEARCH_LAB",
        "04_INFRASTRUCTURE_ENGINEERING",
        "05_MARKET_DATA_VAULT",
        "06_GOVERNANCE_AND_COMPLIANCE",
        "07_BACKUPS",
    ]:
        add_file(files, BASE / folder / "OWNERSHIP_RULES.md")
        add_file(files, BASE / folder / "_AGENT_LOCK.md")
    return dict(sorted(files.items()))


def latest_control_board_path() -> str | None:
    board_root = (
        BASE
        / "06_GOVERNANCE_AND_COMPLIANCE"
        / "multi_agent_control"
        / "parallel_control_board"
    )
    cycles = sorted([p for p in board_root.glob("cycle_*") if p.is_dir()])
    if not cycles:
        return None
    return cycles[-1].relative_to(BASE).as_posix()


def verify_required(names: set[str]) -> dict[str, bool]:
    latest_board = latest_control_board_path()
    checks = {
        "eom_blocker_resolution": any("eom_blocker_resolution/" in n for n in names),
        "maximum_confirmation_rerun": any("maximum_confirmation_rerun/" in n for n in names),
        "v38_reports": any("reports/v38_manipulante3_htf_ltf/" in n for n in names),
        "fixed_runner": any(n.endswith("max_confirmation_eom_fixed_rerun.py") for n in names),
        "eom_integrity_helper": any(n.endswith("src/v7_engine/eom_integrity.py") for n in names),
        "eom_integrity_tests": any(n.endswith("test_manipulante3_eom_integrity.py") for n in names),
        "independent_verify": any("INDEPENDENT_VERIFY" in n for n in names),
        "final_decision": any("DECISION" in n for n in names),
        "edge_translation_diagnosis": any(n.endswith("MANIPULANTE3_EDGE_TRANSLATION_DIAGNOSIS.md") for n in names),
        "next_edge_hypothesis": any(n.endswith("NEXT_EDGE_HYPOTHESIS_RECOMMENDATION.md") for n in names),
        "data_news_audit": any("data_quality_audits/parallel_data_news_audit/" in n for n in names),
        "latest_control_board": bool(latest_board and any(n.startswith(latest_board + "/") for n in names)),
        "git_status": any(n.endswith("EOM_BLOCKER_GIT_STATUS_AFTER.txt") for n in names),
    }
    return checks


def main() -> int:
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    existed_before = OUT.exists()
    if existed_before:
        OUT.unlink()
    exists_after_delete = OUT.exists()
    time.sleep(2)

    manifest_path = LOCK_DIR / "EOMFIX_SINGLE_ZIP_BUILD_MANIFEST.json"
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(BASE),
        "official_zip": str(OUT),
        "old_zip_existed": existed_before,
        "exists_after_delete": exists_after_delete,
        "policy": "single official root ZIP only; no nested ZIPs; no raw data; no cache; no venv; no git internals",
        "latest_control_board": latest_control_board_path(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    files = collect_files()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel, fp in files.items():
            zf.write(fp, rel)

    sha = sha256_file(OUT)
    size_bytes = OUT.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    with zipfile.ZipFile(OUT, "r") as zf:
        names = set(zf.namelist())
        testzip = zf.testzip()
        total_files = len(names)

    required = verify_required(names)
    root_zips = sorted([p.name for p in BASE.glob("*.zip")])
    recursive_zips = sorted(
        [p.relative_to(BASE).as_posix() for p in BASE.rglob("*.zip") if p.is_file()]
    )

    verification = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "official_zip": str(OUT),
        "old_zip_existed": existed_before,
        "exists_after_delete": exists_after_delete,
        "total_files": total_files,
        "size_bytes": size_bytes,
        "size_mb": round(size_mb, 4),
        "sha256": sha,
        "testzip": testzip,
        "root_zips": root_zips,
        "root_zip_count": len(root_zips),
        "recursive_zip_count": len(recursive_zips),
        "recursive_nonofficial_zip_count": len(
            [p for p in recursive_zips if p != "000_PARA_CHATGPT.zip"]
        ),
        "recursive_zips": recursive_zips,
        "required": required,
        "blocked": bool(
            testzip is not None
            or root_zips != ["000_PARA_CHATGPT.zip"]
            or not all(required.values())
        ),
    }

    (LOCK_DIR / "EOMFIX_SINGLE_ZIP_FINAL_VERIFICATION.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in verification.items()) + "\n",
        encoding="utf-8",
    )
    (BASE / "000_PARA_CHATGPT.sha256.txt").write_text(sha, encoding="utf-8")
    (BASE / "VERIFICACION_ZIP_CHATGPT.txt").write_text(
        f"Size: {size_mb:.2f} MB\nFiles: {total_files}\nHash: {sha}\n"
        f"testzip: {testzip}\nroot_zip_count: {len(root_zips)}\n",
        encoding="utf-8",
    )
    with (LOCK_DIR / "EOMFIX_SINGLE_ZIP_CONTENTS_MANIFEST.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.writer(fh)
        w.writerow(["archive_path", "size_bytes"])
        for rel, fp in files.items():
            w.writerow([rel, fp.stat().st_size])

    print(json.dumps(verification, indent=2))
    return 1 if verification["blocked"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
