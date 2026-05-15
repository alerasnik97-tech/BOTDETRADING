#!/usr/bin/env python3
# ============================================================
# test_fixture_artifacts_tracked.py
# ============================================================
# Guard: verifies that every artifact declared in fixture
# MANIFEST_*.json files:
#   1. physically EXISTS on disk
#   2. is TRACKED by git (not in .gitignore)
#   3. does NOT live under a /results/ sub-folder inside fixtures
#      (results/ is in .gitignore; fixtures must use /ranking/)
#
# This is a clean-clone reproducibility guard: if a fixture
# artifact is untracked, a fresh clone will not have it and
# tests will silently fail or pass with wrong data.
# ============================================================
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_FIXTURES_DIR = _HERE.parent / "fixtures"
_REPO_ROOT: Path | None = None


def _git_repo_root() -> Path | None:
    global _REPO_ROOT
    if _REPO_ROOT is not None:
        return _REPO_ROOT
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True,
            cwd=str(_FIXTURES_DIR),
        )
        if r.returncode == 0 and r.stdout.strip():
            _REPO_ROOT = Path(r.stdout.strip()).resolve()
            return _REPO_ROOT
    except Exception:
        pass
    return None


def _git_tracked_files() -> set[str]:
    """Return set of repo-relative paths tracked by git (forward-slash normalised)."""
    root = _git_repo_root()
    if root is None:
        return set()
    try:
        r = subprocess.run(
            ["git", "ls-files"],
            capture_output=True, text=True,
            cwd=str(root),
        )
        if r.returncode == 0:
            return {line.strip().replace("\\", "/") for line in r.stdout.splitlines() if line.strip()}
    except Exception:
        pass
    return set()


def _repo_rel(path: Path) -> str | None:
    root = _git_repo_root()
    if root is None:
        return None
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return None


class TestFixtureArtifactsTracked(unittest.TestCase):
    """
    Reproducibility guard: every MANIFEST_*.json in fixtures/output_*
    must have its declared output_hashes files:
      - existing on disk
      - tracked by git
      - NOT under a /results/ sub-path inside fixtures
    """

    def setUp(self):
        if not _FIXTURES_DIR.is_dir():
            self.skipTest(f"fixtures directory not found: {_FIXTURES_DIR}")
        self.tracked = _git_tracked_files()
        self.git_available = bool(self.tracked)

    # ------------------------------------------------------------------ #
    # Helper
    # ------------------------------------------------------------------ #
    def _collect_manifest_artifacts(self):
        """
        Yields (manifest_path, artifact_rel_key, artifact_abs_path)
        for every key in output_hashes of every MANIFEST_*.json found
        under fixtures/output_*.
        """
        for output_dir in sorted(_FIXTURES_DIR.iterdir()):
            if not output_dir.is_dir() or not output_dir.name.startswith("output_"):
                continue
            for mf in sorted(output_dir.glob("MANIFEST_*.json")):
                try:
                    data = json.loads(mf.read_text(encoding="utf-8"))
                except Exception as exc:
                    self.fail(f"Cannot parse {mf}: {exc}")

                oh = data.get("output_hashes", {})
                if not isinstance(oh, dict):
                    continue
                for rel_key in oh:
                    # artifact declared as a relative path from the output_dir
                    artifact_abs = (output_dir / rel_key).resolve()
                    yield mf, rel_key, artifact_abs

    # ------------------------------------------------------------------ #
    # Test: no /results/ sub-folder inside fixtures
    # ------------------------------------------------------------------ #
    def test_no_results_subfolder_in_fixtures(self):
        """
        Fixtures must use 'ranking/' not 'results/' for ranking outputs.
        .gitignore contains 'results/' which would cause artifacts to be
        untracked on a clean clone.
        """
        offenders = []
        for output_dir in _FIXTURES_DIR.iterdir():
            if not output_dir.is_dir():
                continue
            results_sub = output_dir / "results"
            if results_sub.exists():
                offenders.append(str(results_sub))
        self.assertEqual(
            offenders, [],
            f"Found /results/ sub-folders inside fixtures (must be renamed to /ranking/):\n"
            + "\n".join(f"  {p}" for p in offenders),
        )

    # ------------------------------------------------------------------ #
    # Test: every artifact referenced in output_hashes exists on disk
    # ------------------------------------------------------------------ #
    def test_all_fixture_artifacts_exist(self):
        missing = []
        for mf, rel_key, artifact_abs in self._collect_manifest_artifacts():
            if not artifact_abs.is_file():
                missing.append(f"{mf.name} -> {rel_key} (resolved: {artifact_abs})")
        self.assertEqual(
            missing, [],
            "Fixture artifacts declared in MANIFEST output_hashes but missing on disk:\n"
            + "\n".join(f"  {m}" for m in missing),
        )

    # ------------------------------------------------------------------ #
    # Test: every artifact is tracked by git
    # ------------------------------------------------------------------ #
    def test_all_fixture_artifacts_tracked_by_git(self):
        if not self.git_available:
            self.skipTest("git not available or no tracked files found")
        untracked = []
        for mf, rel_key, artifact_abs in self._collect_manifest_artifacts():
            if not artifact_abs.is_file():
                continue  # handled by test_all_fixture_artifacts_exist
            rel = _repo_rel(artifact_abs)
            if rel is None:
                untracked.append(f"{mf.name} -> {rel_key} (cannot compute repo-relative path)")
                continue
            if rel not in self.tracked:
                untracked.append(f"{mf.name} -> {rel_key} (repo-rel: {rel})")
        self.assertEqual(
            untracked, [],
            "Fixture artifacts not tracked by git (clean-clone reproducibility broken):\n"
            + "\n".join(f"  {u}" for u in untracked),
        )

    # ------------------------------------------------------------------ #
    # Test: no artifact path contains /results/ inside fixtures
    # ------------------------------------------------------------------ #
    def test_no_results_path_in_manifest_output_hashes(self):
        """
        Even if the folder was renamed, the manifest must not reference
        a results/ sub-path (which would still be untracked).
        """
        bad = []
        for mf, rel_key, _ in self._collect_manifest_artifacts():
            if "/results/" in rel_key.replace("\\", "/"):
                bad.append(f"{mf.name} -> {rel_key}")
        self.assertEqual(
            bad, [],
            "Fixture MANIFEST output_hashes reference paths under /results/ "
            "(must be /ranking/):\n"
            + "\n".join(f"  {b}" for b in bad),
        )


if __name__ == "__main__":
    unittest.main()
