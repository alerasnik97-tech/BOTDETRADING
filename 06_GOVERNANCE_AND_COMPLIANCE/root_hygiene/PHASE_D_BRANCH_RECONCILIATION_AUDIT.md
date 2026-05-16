# PHASE D BRANCH RECONCILIATION AUDIT

**Audit date:** 2026-05-16
**Repo:** `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo`
**Auditor branch:** `governance/root-hygiene-20260516` @ `d2c51339`
**Mode:** FORENSIC AUDIT + RECONCILIATION PLAN ONLY (no merge / no apply / no
data regen / no lab).

## 1. Status

**`PHASE_D_RECONCILIATION_CANONICAL_BRANCH_CONFIRMED`**
Secondary, explicit: **`NO_COMMON_ANCESTOR`** between `governance/...` and
`clean-sync-branch` (confirmed local AND origin). Lab remains **NOT
authorized** (Phase E blockers still stand). No false green light.

## 2. Executive Summary

- The official Phase A→B→C→D→E lineage and the `PHASE_D_CANONICALIZATION_REPORT.md`
  / `PHASE_E_FINAL_PRE_LAB_INTEGRITY_AUDIT.md` exist **only** on
  `governance/root-hygiene-20260516` (and its in-sync origin). **Governance is
  the canonical Phase-D branch.**
- The earlier signal that "Phase D was pushed to `clean-sync-branch`" is
  **not substantiated**. Phase D commits (`8ee830e6` etc.) are on governance,
  not on clean-sync-branch.
- `clean-sync-branch` is an **unrelated orphan history** (its root commit is
  `cc7eed4a [v39/github] institutional sync - professional surgical clean
  start`). `git merge-base` returns **empty** for both local and origin pairs →
  GitHub's "no common ancestor" is **TRUE**.
- The two branches have **wholesale divergence** (two-dot diff: 4,229 deletions
  / 1,855 additions; rename detection overflowed). clean-sync-branch carries
  its own parallel lab/governance evolution (V39→V50B, R1 freeze, security
  token revoke, engine lockdown) and its own governance tree — but **no Phase
  D/E artifacts**.
- Therefore a direct merge is **impossible/forbidden** (`--allow-unrelated-
  histories`). Reconciliation of clean-sync-branch's lab/engine/security work
  is a **separate, owner-driven, curated** effort — not a Phase D action and
  not blind.

## 3. Branch Forensics

| branch | local_head | remote_head | phase_A | phase_B | phase_C | phase_D | common_ancestor w/ gov | safe_to_use | decision |
|---|---|---|---|---|---|---|---|---|---|
| `governance/root-hygiene-20260516` | `d2c51339` | `d2c51339` (in sync) | YES | YES | YES | YES (+E) | — (self) | **YES** | **CANONICAL** |
| `clean-sync-branch` | `786eb3c3` | `786eb3c3` (in sync) | NO | NO | NO | **NO** | **NONE** (unrelated) | NO (for Phase-D line) | DO NOT MERGE — separate owner decision |
| `research/f06-clean-train-only-rerun-20260515` | (origin-tracked) | — | partial | partial | partial | NO | YES (merge-base `c1dae887`), fully behind gov | superseded | none (contained in gov) |
| `research/f06-d5-behavior-neutral-telemetry-20260516` | (origin-tracked) | — | partial | partial | partial | NO | YES (merge-base `ef89fddc`), fully behind gov | superseded | none (contained in gov) |

Root commits — governance: `acf105f8`, `c53a973f`; clean-sync-branch:
`cc7eed4a` (disjoint → unrelated histories). Neither branch is an ancestor of
the other.

## 4. Root State

`STRICT_ROOT_LOCAL = YES`. Working tree (governance) contains the 8 canonical
folders + `.gitignore`; documented exceptions `.git`, `.github`, `README.md`,
`requirements.txt`, `requirements-vps-optional.txt`. **Violations: 0.**
(`01_CORE_PRODUCTION` present on disk, gitignored-by-design — not tracked.)

## 5. Phase D Artifact Audit

| Artifact | Exists | Path | Tracked | Branch | Notes |
|---|---|---|---|---|---|
| `PHASE_D_CANONICALIZATION_REPORT.md` | YES | `06_GOVERNANCE_AND_COMPLIANCE/root_hygiene/` | YES | governance only | added by `8ee830e6` (2,055 B) |
| `PHASE_E_FINAL_PRE_LAB_INTEGRITY_AUDIT.md` | YES | same dir | YES | governance only | 11,778 B |
| `canonical_anchor_events.csv` | YES | `05_MARKET_DATA_VAULT/data/official_anchors/out/` | **NO (gitignored)** | none (local-only) | 168 rows + header; sha256 `1e7eb737…` |
| `forex_factory_cache.csv` | **NO** | — | — | — | MISSING everywhere |
| `news_eurusd_v2_utc.csv` | **NO** | — | — | — | MISSING everywhere |

clean-sync-branch tracks **0** of the Phase D/E reports.

## 6. Data Generation Audit — `canonical_anchor_events.csv`

- Generated during the Phase D run by the **"official anchor pipeline"**
  (`research_lab/official_anchors/...`) — a **scope deviation** (Phase D was
  hygiene/path-migration, NOT data regeneration).
- **Untracked + gitignored** → local-only, never committed or pushed → **no
  repo-history contamination** (low blast radius).
- Row count **168 data rows + 1 header (169 lines)** — matches the report's
  "168 filas" claim. Schema header is a valid anchor-events contract.
- Pipeline connectors are **internet-capable** (`connectors/bls_cpi_ppi.py`,
  `bls_employment.py`, `bls_cpi_ppi_hybrid.py` use `urllib.request.urlopen`
  against `bls.gov`) and span **2025/2026** (`years=[2024,2025,2026]`,
  default `--end 2026-12-31`). An offline `connectors/stubs.py` also exists.
- Whether the actual regeneration used the **live network** path or the offline
  stubs **cannot be determined from the artifact alone** — provenance is
  **UNVERIFIED**.
- Verdict: `data_generation_allowed = NO` (outside declared scope; net/2025-26
  capable pipeline). The file is **KEPT (do not delete — rule)**, **NOT
  trusted as lab evidence**, **NOT to be regenerated**. Requires owner +
  Claude provenance audit (which connector, network on/off, date range) before
  any lab use.

## 7. Diff Audit (no merge performed)

`git diff --stat governance..clean-sync-branch` (two-dot, valid even without a
merge base — direct tree comparison): **D=4,229, A=1,855, R100=13, M=3**;
rename detection overflowed (`renameLimit` warning). Top divergent areas:
`07_BACKUPS` (3,081), `03_RESEARCH_LAB` (2,037), `06_GOVERNANCE_AND_COMPLIANCE`
(500). Three-dot/merge-based reconciliation is **not applicable**
(no common ancestor).

Classification: **not a small/cherry-pickable Phase D delta** — it is a
**full divergence between two unrelated histories**, each independently
canonicalized. clean-sync-branch additionally contains its own governance docs
(`p0_p1_lab_hardening/`, `github_source_of_truth_policy/ROOT_HYGIENE_POLICY.md`,
`data_news_vault_integrity/`) absent from governance.

→ `NO_COMMON_ANCESTOR_BRANCH_RECONCILIATION_REQUIRED`.

## 8. Safe Tests

- `PYTHONPATH=03_RESEARCH_LAB python -c "import research_lab"` → **OK**.
- F06 pipeline unittest discovery → **Ran 119 — OK** (exit 0). No regression.
- (Out of this block's scope but still standing from Phase E: broader
  `research_lab/tests` 105/149, active `light_runner` ImportError — see
  `PHASE_E_FINAL_PRE_LAB_INTEGRITY_AUDIT.md`.)

## 9. Recommended Reconciliation Option

**OPTION A is CONFIRMED for Phase D** — `governance/root-hygiene-20260516`
remains canonical; Phase D is **already correctly closed there** (nothing to
re-apply for Phase D itself).

**OPTION C is the mandated vehicle for the SEPARATE clean-sync-branch
problem** — create `governance/phase-d-reconciliation-20260516` **from
governance** as the workspace for any owner-approved, **curated** integration
of clean-sync-branch's lab/engine/security commits (cherry-pick / patch
extraction), because:
- there is **no common ancestor** (blind merge forbidden/impossible);
- the user's institutional preference is OPTION C under any doubt;
- clean-sync-branch must **not** be force-pushed, merged, or deleted.

`apply_now = NO` — this phase is forensic audit + plan only.

## 10. What Is Still Forbidden

No `main`. No force push. No merge (esp. `--allow-unrelated-histories`). No
destructive rebase. No branch/file deletion. No backtest / strategy / F06-real
/ optimization / sweep / validation / holdout. No 2025-2026 analysis. No
engine/trading-logic change. No further file moves. No data regeneration. No
downloads. No ZIP workflow. Lab NOT authorized.

## 11. Next Step

1. Owner decides **source-of-truth policy**: governance (canonical Phase-D) vs
   clean-sync-branch (parallel lab trunk) — they are unrelated histories.
2. Execute `NEXT_PROMPT_APPLY_PHASE_D_RECONCILIATION.md` (creates the
   reconciliation branch from governance; curated, non-destructive only).
3. Resolve Phase E blockers (B2 import / B3 path / B4 data / B5 tests) on the
   canonical line before any lab pre-check.

## 12. Copy-Paste Summary for ChatGPT

```
PHASE D BRANCH RECONCILIATION — STATUS: CANONICAL_BRANCH_CONFIRMED
(secondary: NO_COMMON_ANCESTOR clean-sync vs governance — TRUE, local+origin)

- Canonical branch = governance/root-hygiene-20260516 @ d2c51339
  (in sync w/ origin). It alone has Phase A->B->C->D->E + the
  PHASE_D/PHASE_E reports.
- clean-sync-branch @ 786eb3c3 is an UNRELATED orphan history
  (root cc7eed4a "v39 clean start"), NO common ancestor with governance
  (git merge-base empty, local AND origin) -> matches GitHub.
- "Phase D pushed to clean-sync-branch" is NOT substantiated; Phase D is
  on governance only. clean-sync has NO Phase D/E artifacts.
- gov..csb two-dot diff = 4229 D / 1855 A: wholesale divergence, not a
  cherry-pickable Phase D delta. Direct merge impossible/forbidden.
- canonical_anchor_events.csv: 168 rows (matches report), UNTRACKED +
  gitignored (no repo contamination), regenerated by net/2025-26-capable
  official anchor pipeline OUT OF SCOPE -> provenance UNVERIFIED, not
  lab-trustworthy, keep (don't delete), don't regenerate.
- forex_factory_cache.csv & news_eurusd_v2_utc.csv: MISSING everywhere.
- Safe tests: research_lab import OK; F06 119/119 OK.

DECISION: governance canonical for Phase D (OPTION A confirmed). The
clean-sync-branch lab/engine/security state is a SEPARATE owner-driven
curated reconciliation via OPTION C (new branch governance/phase-d-
reconciliation-20260516). NO merge, NO force push, NO apply now.
Lab still NOT authorized.
```
