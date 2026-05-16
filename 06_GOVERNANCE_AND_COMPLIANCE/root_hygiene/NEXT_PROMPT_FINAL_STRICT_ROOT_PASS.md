# NEXT PROMPT — FINAL STRICT ROOT PASS

**Goal:** Final certification that the repository root is strictly the 8
canonical folders + `.gitignore` (+ documented technical exceptions), with all
latent runtime drift removed, then push and (optionally) open the governance PR.

## Entry conditions (must all be true before starting)
- `NEXT_PROMPT_REPORTS_PATH_MIGRATION.md` completed (no writer recreates root
  `reports/`).
- `NEXT_PROMPT_RESEARCH_LAB_IMPORT_MIGRATION.md` completed (`import research_lab`
  canonical).
- Branch `governance/root-hygiene-20260516`, working tree clean.

## Current baseline (from Phase D audit, HEAD `8ee830e6`)
Root already conforms structurally:
- Canonical: `01_CORE_PRODUCTION` (gitignored-by-design), `02_INCUBATION_STAGING`,
  `03_RESEARCH_LAB`, `04_INFRASTRUCTURE_ENGINEERING`, `05_MARKET_DATA_VAULT`,
  `06_GOVERNANCE_AND_COMPLIANCE`, `07_BACKUPS`, `08_CLOUD_FREE_RUN_LAB`,
  `.gitignore`.
- Technical exceptions: `.github`, `README.md`, `requirements.txt`,
  `requirements-vps-optional.txt`, `.git`.
- Non-canonical / non-exception root items: **0**.

## Scope of the final pass
1. Re-list tracked + filesystem root; assert the table above byte-for-byte
   (count of unexpected items must be 0).
2. Decide the disposition of the 4 technical exceptions: keep as documented
   exceptions (recommended) vs. relocate (`requirements*.txt` →
   `04_INFRASTRUCTURE_ENGINEERING`?). Record an explicit owner decision; do not
   relocate `.github`/`README.md` (workflow + repo-landing dependencies).
3. Confirm `01_CORE_PRODUCTION` policy: stays gitignored-by-design (production
   engine/configs/live logs) — do not force into VCS.
4. Re-run safe checks: `import research_lab` (now canonical) + F06 suite
   (119/119). Both green.
5. Push `governance/root-hygiene-20260516` (non-force). Optionally open a PR
   into the integration branch — **never** into `main` without explicit owner
   approval.

## Hard rules
- No backtests/strategies/F06 runs/optimizations/validations (only import check
  + F06 unit-test suite).
- No data touched (csv/parquet/raw), no 2025/2026 period analysis.
- No force push, no `main` merge without approval, no history rewrite, no ZIPs,
  no `git clean -fdx`. Investigate unexpected state before deleting/overwriting.
