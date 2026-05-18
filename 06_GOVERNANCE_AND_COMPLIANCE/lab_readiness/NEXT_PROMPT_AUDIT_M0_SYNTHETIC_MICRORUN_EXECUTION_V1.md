# NEXT PROMPT — AUDIT M0 SYNTHETIC MICRORUN EXECUTION V1

## 0. Nature Of This Document
Governance read-only audit template. It authorizes NO execution, micro-run,
dry-run, backtest, train, validation, holdout, 2025/2026 access,
optimization/sweep, Sub-Batch 1B, or parallel writers. It is strictly for a
future external read-only audit of the M0 synthetic microrun execution.

---

## 1. Context
- Execution branch: `research/m0-synthetic-microrun-bo01-mr02-v1-20260518`.
- Base commit: `be59a75279973c06cbc682c7d5999de492645692`.
- run_id: `M0_SYNTHETIC_BO01_MR02_20260518_092916`.
- Governance report: `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/M0_SYNTHETIC_MICRORUN_EXECUTION_REPORT_V1.md`.
- Local outputs are gitignored and are NOT committed; the auditor inspects them
  locally read-only.

---

## 2. Mandatory Prechecks
```powershell
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, ProcessName, StartTime, CPU, WorkingSet
Get-CimInstance Win32_Process -Filter "name='python.exe'" | Select-Object ProcessId, CommandLine | Format-List
```
If any active research process is running, ABORT with
`BLOCKED_ACTIVE_RESEARCH_PROCESS_DETECTED`. Confirm not on `main`, no
unexpected staged files, W-01 (11 files, confined) and W-02 preexisting and
untouched.

---

## 3. Audit Scope (read-only)
Verify, against real evidence (git show/diff/status, local manifest, report):
1. **Commit scope:** the execution commit changed ONLY
   `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/M0_SYNTHETIC_MICRORUN_EXECUTION_REPORT_V1.md`
   and `NEXT_PROMPT_AUDIT_M0_SYNTHETIC_MICRORUN_EXECUTION_V1.md`. No code,
   tests, data, strategy/engine/runner files, data vault, outputs, ZIP, or
   root files entered the commit.
2. **Synthetic-only:** strategies were loaded directly via importlib (no
   package `__init__` side effects); fixtures are in-memory, tz-aware M5; no
   file price reads; no data vault; synthetic calendar label is non-2025/2026.
3. **Local outputs:** only `M0_SYNTHETIC_MICRORUN_REPORT.md`,
   `output_manifest.json`, `command_log.txt`, and the temporary runner exist,
   strictly under the gitignored output root; none are committed; no
   `trades.csv`/`equity_curve.csv`/ZIP/screenshots.
4. **Checks:** all 16 synthetic checks PASS; every result is `None`
   (fail-closed/gate/no-setup); contract validity respected.
5. **No metrics:** no PF/win-rate/drawdown/Sharpe/expectancy/PnL/equity curve.
6. **No forbidden scope:** no backtest/train/dry-run/validation/holdout/
   2025/2026/optimization/sweep/Sub-Batch 1B/parallel writers; no source/test/
   data modification.
7. **W-01/W-02:** unchanged; preserved as future gates.
8. **No claims:** no edge/performance/profitability/champion/demo/real/FTMO.
9. **Language:** no affirmative `secure`/`perfect`/`guaranteed`/`100%`/
   `successfully`/`fully`/`edge`/`champion`/`rentable`.

---

## 4. Forbidden Actions
- NO modification of code, tests, or data.
- NO execution / micro-run / dry-run / backtest / train.
- NO validation / holdout / 2025/2026 / optimization / sweep / grid / walk-forward.
- NO Sub-Batch 1B; NO parallel writers; NO multi-agent writing.
- NO production / demo / real / FTMO.
- NO `git add .`; NO force push; NO main; NO reset --hard; NO git clean; NO git stash.
- NO touching W-01 or W-02; NO committing local outputs.

---

## 5. Allowed Actions
- Read-only inspection (files, local output root).
- `git status`, `git diff`, `git show`, `git log` (read-only).
- Creating exactly one audit report:
  `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/M0_SYNTHETIC_MICRORUN_EXECUTION_EXTERNAL_AUDIT_V1.md`.
- Staging, committing, and pushing ONLY that audit report to a new audit branch
  `audit/m0-synthetic-microrun-execution-review-vN-YYYYMMDD` (derived from the
  audited commit; vN incremented if a prior version exists; dated; not main;
  never force-pushed).

---

## 6. Activation
Requires the owner's explicit autonomous approval phrase:
"AUTORIZO AUDITORÍA EXTERNA READ-ONLY DE LA EJECUCIÓN M0 SYNTHETIC MICRORUN
BO01/MR02, SIN EJECUTAR NADA."
Citation, paraphrase, or short confirmation does NOT count. Otherwise ABORT
with `BLOCKED_MISSING_EXPLICIT_OWNER_APPROVAL`.

---
*End of Prompt*
