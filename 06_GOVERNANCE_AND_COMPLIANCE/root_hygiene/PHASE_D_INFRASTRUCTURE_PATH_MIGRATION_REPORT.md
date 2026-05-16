# PHASE D — INFRASTRUCTURE & PATH MIGRATION REPORT

**Branch:** `governance/root-hygiene-20260516`
**Audit date:** 2026-05-16
**HEAD at audit:** `8ee830e64f4ce9fc8a22b090d9be772bd9af010a`
**Prior Phase C commit:** `4c434df204d0f01bfcbfb57b27f0af01fb12e740`
**Status:** `PHASE_D_VERIFIED_NO_NEW_MOVES_REQUIRED`

---

## 0. Important Context Reconciliation (read first)

The Phase D execution prompt described the current state as: *last commit `4c434df2`
(Phase C), `reports` blocked, `research_lab` still in root.*

**Reality at audit time differs** — that context was stale. A commit
`8ee830e6 "Phase D Complete: Root Canonicalization (Strict 8-Folder Structure)
and Data Vault Path Migration"` **already exists locally, on top of Phase C,
and is unpushed (branch ahead 1).** Phase D's structural moves were therefore
already applied and committed before this audit ran.

Per Git-safety policy, the pre-existing commit was **investigated, not
re-executed and not amended**. This document is a **verification + governance
pass** over the already-applied work: it audits the real final state, confirms
safety, records what is still pending, and produces the required next-step
documents. No destructive or history-rewriting action was taken.

---

## 1. Root State

### 1.1 Initial root (conceptual, pre-`8ee830e6`)
Per prior phase reports, before `8ee830e6` the root still carried: a root
`reports/` tree, a root `research_lab/` package, root `scripts/`, a duplicate
root `BOT_V2_DAYTIME_LAB`, and a `.tmp.driveupload/` Google-Drive scratch tree.

### 1.2 Final root (observed at audit, post-`8ee830e6`)

Git-tracked top level:

| Item | Classification |
|---|---|
| `02_INCUBATION_STAGING` | STRICT_ALLOWLIST (canonical) |
| `03_RESEARCH_LAB` | STRICT_ALLOWLIST (canonical) — now holds `research_lab/` + `reports/` |
| `04_INFRASTRUCTURE_ENGINEERING` | STRICT_ALLOWLIST (canonical) — holds relocated scripts |
| `05_MARKET_DATA_VAULT` | STRICT_ALLOWLIST (canonical) — primary DATA_ROOT |
| `06_GOVERNANCE_AND_COMPLIANCE` | STRICT_ALLOWLIST (canonical) |
| `07_BACKUPS` | STRICT_ALLOWLIST (canonical) |
| `08_CLOUD_FREE_RUN_LAB` | STRICT_ALLOWLIST (canonical) |
| `.gitignore` | STRICT_ALLOWLIST |
| `.github` | TECHNICAL EXCEPTION (GitHub Actions / workflow dependency) |
| `README.md` | TECHNICAL EXCEPTION |
| `requirements.txt` | TECHNICAL EXCEPTION |
| `requirements-vps-optional.txt` | TECHNICAL EXCEPTION |

Filesystem-only (untracked by design):

| Item | Classification |
|---|---|
| `01_CORE_PRODUCTION` | STRICT_ALLOWLIST (canonical #1) — present on disk; **all contents gitignored** (production `execution_engine`, `configs`, `risk_controls`, `strategies`, `releases`, `logs_live`). Empty/ignored ⇒ not in `git ls-tree HEAD`. This is intentional, not a defect. |
| `.git` | Normal hidden VCS directory (allowed) |

**Result:** the root **already conforms** to the strict 8-folder standard plus
the four documented technical exceptions. There are **no loose scripts, no root
`reports/`, no root `research_lab/`, no ZIPs and no data in the root.**
`ROOT_AFTER` non-canonical/non-exception count = **0**.

---

## 2. `reports` Audit (Step 3)

- **Location:** moved by `8ee830e6` from root `reports/` → `03_RESEARCH_LAB/reports/`
  (pure renames, `R100` — byte-identical, no content change).
- **Size:** 101 tracked files. Subdirectories include:
  - **Historical / archive:** `canonical_context_master_20260416_163108/`,
    `canonical_forensic_next_step_20260416_160901/`,
    `canonical_microstructure_iter2_20260416_172155/`, `infra_audits/`,
    `reports_legacy/`, `phase12_final_audit.md`.
  - **Active output contracts:** `news_reliability/`, `engine_safety/`,
    `official_anchors/`, `vps_readiness/`.
- **Classification: `MIXED_REPORTS`** — both historical archives and
  live pipeline output targets.
- **Reference audit:** active business code referencing a `reports/` path is
  minimal. Key finding: `03_RESEARCH_LAB/research_lab/news_phase3_mass_validate.py`
  resolves `PROJECT_ROOT = Path(__file__).resolve().parents[2]` = **repository
  root**, then writes to `PROJECT_ROOT / "reports" / "news_reliability" / ...`.
  Because root `reports/` no longer exists, **running that script would recreate
  a root-level `reports/` directory**, re-polluting the strict root.
- **Risk now:** none — scripts are not executed in this phase and were not run.
  This is **latent output-contract drift**, not an active break (F06 suite is
  green, see §6).
- **Action taken:** none (correctly out of scope — editing an active pipeline's
  path resolution is risky and requires its own tested change).
  Deferred to **`NEXT_PROMPT_REPORTS_PATH_MIGRATION.md`**.

---

## 3. Root Scripts / Infrastructure Audit (Step 4)

- Root `scripts/` was relocated by `8ee830e6` to
  `04_INFRASTRUCTURE_ENGINEERING/legacy_scripts/root_scripts_archive/`
  (103 files: campaign runners, audits, builders, `.ps1` bootstraps, backups).
- `04_INFRASTRUCTURE_ENGINEERING` already contains canonical infra buckets:
  `bootstrap/`, `git_infra/`, `checks/`, `harnesses/`, `maintenance_scripts/`,
  `legacy_ops/`, `legacy_scripts/`, `legacy_phases/`, `legacy_tests/`,
  `monitoring/`, `vps/`, `telegram/`, `mt5_lab/`, `ide_configs/`.
- **Loose infrastructure scripts remaining in root: NONE.**
- `bootstrap.ini` / `preflight_check.py` / `git_operations.py` / `phase34*` /
  `news_impact_analysis_v2.py` — not present in root (relocated in earlier
  phases or under `legacy_scripts/`). No additional safe move available without
  risk; **no new moves applied.**

---

## 4. `research_lab` Audit (Step 5)

- **Location:** moved by `8ee830e6` from root `research_lab/` →
  `03_RESEARCH_LAB/research_lab/`. `research_lab/scratch/*.py` deleted.
- **Reference magnitude:** ≥ 98 import references across ≥ 40 `.py` files
  (predominantly the package's own `tests/` suite and quarantined
  `07_BACKUPS/scratch_quarantine/` copies).
- **Import behaviour observed:**
  - From bare repo root: `import research_lab` → `ModuleNotFoundError`.
  - With `PYTHONPATH=03_RESEARCH_LAB`: `import research_lab` → **OK**.
- **Decision: KEEP IN PLACE (`03_RESEARCH_LAB/research_lab`). Do NOT move
  again.** The volume of references and the PYTHONPATH-sensitive resolution make
  any further relocation high-risk and outside this phase's safe scope.
- **Forward work:** formalize as an installable/PYTHONPATH-resolved package so
  `import research_lab` works canonically without manual env tweaks. Deferred to
  **`NEXT_PROMPT_RESEARCH_LAB_IMPORT_MIGRATION.md`**.

---

## 5. Data-Path Reference Audit (Step 6)

- Active config `03_RESEARCH_LAB/research_lab/config.py` **already points every
  data path to `05_MARKET_DATA_VAULT/...`** (EURUSD/USDJPY prepared dirs,
  high-precision dukascopy dirs, news imports, canonical news files). No stale
  root data paths remain in active configuration.
- Remaining old-name matches (`data_usdjpy`, `DATA MANUAL`, etc.) live only in
  **non-active** files: `*_BACKUP_*.py`, `07_BACKUPS/**`,
  `legacy_scripts/root_scripts_archive/**`, and a test fixture
  (`tests/test_usdjpy_readiness.py`). These are archival/quarantined and require
  no change.
- **Data-path action: none required** — active code already migrated.
  **No data files were read, loaded, moved or analysed.**

---

## 6. Safety Verification (Step 8)

| Check | Command | Result |
|---|---|---|
| research_lab import | `PYTHONPATH=03_RESEARCH_LAB python -c "import research_lab"` | **OK** |
| research_lab import (bare root) | `python -c "import research_lab"` | ModuleNotFound (expected post-relocation; drives §4 next-prompt) |
| F06 pipeline tests | `python -m unittest discover -s 03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/pipelines/f06_evidence_rebuild/tests -p "test_*.py"` | **119/119 OK** (4.1s) |

The Phase D moves in `8ee830e6` did **not** regress the F06 pipeline or the
research_lab import contract (under the documented PYTHONPATH).

---

## 7. Moves Applied vs Blocked

### Applied (by pre-existing commit `8ee830e6`, verified here — not re-done)
- `reports/` → `03_RESEARCH_LAB/reports/`
- `research_lab/` → `03_RESEARCH_LAB/research_lab/`
- root `scripts/` → `04_INFRASTRUCTURE_ENGINEERING/legacy_scripts/root_scripts_archive/`
- duplicate root `BOT_V2_DAYTIME_LAB` quarantined
- `.tmp.driveupload/` removed
- `research_lab` PROJECT_ROOT resolution + `config.py` data vault remap

### Applied in this verification pass
- Governance docs only (this report + next-step prompts) added under
  `06_GOVERNANCE_AND_COMPLIANCE/root_hygiene/`. **No repository content moved.**

### Blocked / deliberately deferred (by safety rules)
- Re-pointing active `reports/` writers off the repo root → `NEXT_PROMPT_REPORTS_PATH_MIGRATION.md`
- Formalizing `research_lab` packaging/PYTHONPATH → `NEXT_PROMPT_RESEARCH_LAB_IMPORT_MIGRATION.md`
- Final strict-root certification pass → `NEXT_PROMPT_FINAL_STRICT_ROOT_PASS.md`
- `01_CORE_PRODUCTION` is gitignored-by-design; not forced into VCS.

---

## 8. Plan for the Final Strict Root

The root is already structurally strict. To *certify* it and remove the last
latent risks, in order:

1. **Reports path migration** — make active writers target a canonical outputs
   path (or `03_RESEARCH_LAB/reports/`) so no run can recreate root `reports/`.
2. **research_lab import migration** — canonical PYTHONPATH / packaging so
   `import research_lab` resolves without manual env setup.
3. **Final strict-root pass** — re-verify zero non-canonical root items, all
   tests green, then push and (optionally) open the governance PR.

---

## 9. Safety Confirmation

- `main` not touched. No force push. No history rewrite / no amend of `8ee830e6`.
- No backtests, strategies, F06 runs, optimizations or validations executed
  (only the explicitly-authorized import check and the F06 **unit-test** suite).
- No data (csv/parquet/raw) read, moved or analysed. No 2025/2026 period work.
- No active imports or workflows broken (verified: §6).
- No trading/engine/business logic modified. No `git clean -fdx`. No ZIPs.
- Only governance documentation was added.
