# NEXT PROMPT — REPORTS PATH MIGRATION

**Goal:** Eliminate the latent output-contract drift where active pipelines
would recreate a root-level `reports/` directory, so the strict root cannot be
re-polluted at runtime.

## Context (from Phase D audit)
- `reports/` was moved to `03_RESEARCH_LAB/reports/` (commit `8ee830e6`,
  byte-identical renames). It is **MIXED**: historical archives
  (`canonical_*`, `infra_audits`, `reports_legacy`, `phase12_final_audit.md`)
  plus active output contracts (`news_reliability/`, `engine_safety/`,
  `official_anchors/`, `vps_readiness/`).
- Root cause of drift: `03_RESEARCH_LAB/research_lab/news_phase3_mass_validate.py`
  sets `PROJECT_ROOT = Path(__file__).resolve().parents[2]` (= repo root) and
  writes to `PROJECT_ROOT / "reports" / "news_reliability" / ...`. Other lab
  modules may share this pattern.

## Scope of the next prompt
1. Enumerate **every** active writer that constructs a `reports/` (or
   `PROJECT_ROOT/"reports"`) output path. Use `rg` over `03_RESEARCH_LAB`
   (exclude `reports/`, `*_BACKUP_*`, `07_BACKUPS`, `legacy_scripts`).
2. Decide the canonical output root (recommended: `03_RESEARCH_LAB/reports/`
   so live artifacts and the relocated tree coincide; OR an explicit
   `OUTPUTS_ROOT` constant in `research_lab/config.py`).
3. Update writers to the canonical path via a single shared constant — **one
   tested change per module**, no behavior change to business logic.
4. Separately, propose moving purely-historical subtrees to
   `06_GOVERNANCE_AND_COMPLIANCE/legacy_reports/` (governance archive) — plan
   only, do not mix with the writer-path change.
5. Re-run the safe checks: `import research_lab` (with `PYTHONPATH=03_RESEARCH_LAB`)
   and the F06 unittest suite (119 tests). Both must stay green.

## Hard rules
- Do not run pipelines/backtests/strategies/optimizations/validations.
- Do not touch data (csv/parquet/raw) or 2025/2026 periods.
- Do not move `reports/` content again until the writer paths are fixed.
- No force push, no `main`, no history rewrite, no ZIPs, no `git clean -fdx`.
- Branch: `governance/root-hygiene-20260516`.
