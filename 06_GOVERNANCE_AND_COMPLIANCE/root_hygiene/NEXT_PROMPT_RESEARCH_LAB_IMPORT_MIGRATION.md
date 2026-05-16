# NEXT PROMPT — RESEARCH_LAB IMPORT MIGRATION

> Supersedes the original stub. The original goal ("move research_lab out of
> root") is **already done** — `research_lab/` now lives at
> `03_RESEARCH_LAB/research_lab/` (commit `8ee830e6`). The remaining work is
> canonical import resolution.

**Goal:** Make `import research_lab` resolve canonically without requiring a
manual `PYTHONPATH=03_RESEARCH_LAB`, so tooling/CI/devs get a stable contract.

## Context (from Phase D audit)
- Package location: `03_RESEARCH_LAB/research_lab/`.
- Observed: `import research_lab` fails from bare repo root, **succeeds with
  `PYTHONPATH=03_RESEARCH_LAB`**.
- ≥ 98 import references across ≥ 40 `.py` files (mostly the package's own
  `tests/` and quarantined `07_BACKUPS/scratch_quarantine/` copies). High blast
  radius → relocation is NOT an option; only resolution config changes.
- F06 suite (119 tests) is green when invoked with that PYTHONPATH.

## Scope of the next prompt
1. Choose ONE canonical resolution mechanism (recommended order):
   a. `pyproject.toml` / editable install (`pip install -e`) exposing
      `research_lab` from `03_RESEARCH_LAB/`; **or**
   b. a committed `conftest.py` / `.pth` / `sitecustomize` that adds
      `03_RESEARCH_LAB` to `sys.path`; **or**
   c. document `PYTHONPATH=03_RESEARCH_LAB` as the official contract and wire it
      into CI + run scripts.
2. Apply the chosen mechanism without editing import statements en masse.
3. Verify: `import research_lab` works from bare repo root, and F06 suite stays
   119/119.
4. Update `.github` workflows only if they invoke research_lab/F06 and rely on
   the old path (audit first; change minimally).

## Hard rules
- Do not run pipelines/backtests/strategies/optimizations/validations
  (only the import check and the F06 **unit-test** suite are allowed).
- Do not touch data (csv/parquet/raw) or 2025/2026 periods.
- Do not move `research_lab/` again. No mass import rewrites.
- No force push, no `main`, no history rewrite, no ZIPs, no `git clean -fdx`.
- Branch: `governance/root-hygiene-20260516`.
