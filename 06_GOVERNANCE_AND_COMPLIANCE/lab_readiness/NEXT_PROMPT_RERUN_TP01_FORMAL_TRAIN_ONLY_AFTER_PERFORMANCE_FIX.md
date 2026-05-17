# NEXT PROMPT — RERUN TP01 FORMAL TRAIN-ONLY (AFTER PERFORMANCE FIX)

Status gate: **TP01_PERFORMANCE_FIX_APPROVED_WITH_WARNINGS_READY_FOR_FORMAL_RERUN**
Authorizing audit: `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/TP01_PERFORMANCE_FIX_EXTERNAL_AUDIT_REPORT.md`
Audit branch: `audit/tp01-performance-fix-before-formal-rerun-20260516`
Fix commit: `fb529a67c53caff665aeaf0d0cb692aa46abf65d`

The TP01 performance fix has passed external audit with non-blocking warnings. You are authorized to relaunch the TP01 formal train-only backtest. This prompt is the exact, scoped order for that rerun.

---

## OBJECTIVE

Relaunch and complete the formal train-only backtest of `tp01_london_ny_momentum_pullback` over **2015-01-01 → 2024-12-31** at native resolution, using the audited performance fix. Record the **real wall-clock runtime**.

---

## EXACT RUN METADATA (must match exactly)

- strategy: `tp01_london_ny_momentum_pullback` (ONE strategy only)
- mode: formal **train-only**
- date range: **2015-01-01 → 2024-12-31** (train window only)
- params: `DEFAULT_PARAMS` (unchanged; no overrides)
- code at: branch containing fix commit `fb529a67c53caff665aeaf0d0cb692aa46abf65d`
- engine: unmodified
- data: unmodified, train-only dataset

---

## HARD CONSTRAINTS (absolute — do not violate)

- NO optimization
- NO sweep / grid / parameter search
- NO validation run
- NO holdout / sealed_holdout (do NOT touch or load holdout data)
- NO 2025 / NO 2026 data (train window stops at 2024-12-31)
- NO news filter / NO forex_factory / NO news data
- NO high_precision mode / NO level2 / NO BID-ASK precision package
- NO F06 real / NO F06 adapter
- ONE strategy only — no multi-strategy batch
- Do NOT modify `engine.py`, `data_loader.py`, the strategy, the tests, or any data
- Do NOT touch `main`; no force push; no rebase; no clean-sync
- Do NOT use `git add .`; stage only explicit result/report files
- Do NOT create files in repo root; no ZIP workflow

---

## CACHE / SAFETY NOTES CARRIED FROM AUDIT (W1–W3)

- The strategy uses a module-global `_CACHE`. For this scoped run (single frame, single param set) exactly **one** cache entry is created (~57.6 MB for ~3.6M bars). This is expected and acceptable.
- The cache is safe for this run because `run_backtest` is read-only on `frame` and does not mutate it in-place during the bar loop (verified in the audit). Do NOT introduce any in-place frame mutation, frame reuse across runs in the same process, or multi-run loops in the same Python process — those would expose the W1 staleness hazard. Use a **fresh Python process** for this run.
- Expected runtime is **minutes, not seconds** (linear O(N); `np.percentile` over the lookback runs on every in-window bar). Worst-case projection ≈ 12–13 min; realistic ≈ a few minutes plus a one-time O(N) precompute. This is expected — do NOT treat a multi-minute runtime as a regression.

---

## PRE-RUN CHECKLIST

1. Confirm no active python / formal-backtest process before starting.
2. Confirm branch HEAD contains `fb529a67c53caff665aeaf0d0cb692aa46abf65d`.
3. Confirm engine.py / data_loader.py / strategy / tests are unmodified vs the fix commit.
4. Confirm the dataset is train-only and stops at 2024-12-31 (no 2025/2026 rows).
5. Start the run in a fresh Python process; capture start and end timestamps.

---

## DURING / AFTER THE RUN — REQUIRED RECORDING

Record and report:

- real wall-clock runtime (start ts, end ts, total seconds/minutes)
- number of bars processed
- peak memory (approx)
- run completed fully (NOT killed) — yes/no
- output location (gitignored / non-root results dir)
- confirmation: no optimization / no sweep / no validation / no holdout / no 2025-2026 / no news / no high-precision used

---

## DELIVERABLE

A formal train-only TP01 result plus a short run report under `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/` (explicit-stage only; docs/report files only; no code/data/outputs in root, no `git add .`). Do NOT proceed to validation, holdout, or 2025/2026 — that is a separate, later, separately-authorized phase.

---

## POST-RUN (separate, do NOT do now)

After a successful formal rerun, schedule cache hardening W1/W2/W5 from the audit report (bounded cache + `clear_cache()` helper + content/version-based key + a same-object-mutation regression test). That is a distinct change set, not part of this rerun.
