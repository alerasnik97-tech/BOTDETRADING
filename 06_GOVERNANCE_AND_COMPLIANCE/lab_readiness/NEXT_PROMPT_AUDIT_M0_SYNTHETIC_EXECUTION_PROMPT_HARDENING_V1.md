# NEXT PROMPT — AUDIT M0 SYNTHETIC EXECUTION PROMPT HARDENING V1

## 0. Nature Of This Document
This is a governance read-only audit template. It authorizes NO code execution,
micro-run, dry-run, backtest, train, validation, holdout, 2025/2026 access, or
optimization/sweep. It is strictly for a future external read-only audit of the
documentation-only hardening patch (W-A through W-E).

---

## 1. Context
The hardening patch was applied on branch
`research/m0-synthetic-execution-prompt-hardening-v1-20260518` (base commit
`77ffc21b6d511589e328be953edf5dae5606c111`). The patch is documentation-only and
must be audited read-only before any owner-use / execution decision. No
execution exists.

---

## 2. Mandatory Prechecks
```powershell
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, ProcessName, StartTime, CPU, WorkingSet
Get-CimInstance Win32_Process -Filter "name='python.exe'" | Select-Object ProcessId, CommandLine | Format-List
```
If any active Python backtest, validation unsealing, optimization sweep, micro-run,
dry-run, or unknown research process is running, ABORT IMMEDIATELY with:
`BLOCKED_ACTIVE_RESEARCH_PROCESS_DETECTED`
Also confirm: not on `main`; no unexpected staged files; W-01 dirty tree and
W-02 output debt preexisting and untouched.

---

## 3. Audit Scope (read-only)
Audit the real diff of the hardening branch versus its base commit and verify:
1. **Diff Integrity:** only the authorized markdown files under
   `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/` were modified or created; no
   code, tests, data, strategy files, engine, runner, data vault, outputs, ZIP,
   or root files entered the diff.
2. **W-A fixed:** the activation gate now explicitly voids self-citation,
   documentation/report/log/example/code-block/quote citation, paraphrase,
   translation, partial form, and short confirmations; requires a new autonomous
   owner declaration; adds `BLOCKED_AMBIGUOUS_OWNER_APPROVAL`; exact activation
   phrase unchanged.
3. **W-B fixed:** Forbidden Scope explicitly forbids Sub-Batch 1B,
   MR03/LS01/LS02, parallel writers, multi-agent writing, second-agent editing,
   portfolio expansion, additional strategies/families, dynamic discovery; M0
   restricted to BO01/MR02 single writer.
4. **W-C fixed:** synthetic fixture policy enumerates BO01/MR02
   daily_trade_count gate, BO01/MR02 active_position gate, and a negative
   control, each expecting `signal` → `None`; no performance metrics added.
5. **W-D fixed:** future report mandatory declarations now include no real data,
   no data vault, no backtest, no train, no dry-run, no validation, no holdout,
   no 2025/2026, no optimization/sweep, no Sub-Batch 1B, no parallel writers, no
   code/tests/data modified, and W-01/W-02 gate status.
6. **W-E fixed:** future audit prompt branch wording neutralized to a documented
   convention with derivation rules; no main; no force push.
7. **No code/test/data changes** anywhere in the diff.
8. **No execution** was performed during the patch.
9. **W-01/W-02 untouched** and preserved as future execution gates.
10. **No owner-less path:** nothing in the patched documents authorizes
    execution without the exact autonomous owner phrase.
11. **No language inflation:** no `secure`, `perfect`, `guaranteed`, `100%`,
    `successfully`, `fully`, `completely sealed/blocked`, `edge`, `champion`,
    `rentable` used affirmatively.
12. **History integrity:** the original external audit report's findings,
    severities, decision, and status are preserved (addendum is non-superseding;
    warnings not falsely declared resolved).

---

## 4. Forbidden Actions
- NO modification of code, tests, or data.
- NO execution of M0, micro-run, dry-run, backtest, or train.
- NO validation, holdout, or 2025/2026 access.
- NO optimization, sweep, grid search, or walk-forward.
- NO Sub-Batch 1B; NO parallel writers; NO multi-agent writing.
- NO production / demo / real / FTMO.
- NO `git add .`; NO force push; NO main; NO reset --hard; NO git clean; NO git stash.
- NO touching W-01 or W-02.

---

## 5. Allowed Actions
- Read-only inspection of the files in the workspace.
- `git status`, `git diff`, `git show`, `git log` (read-only).
- Creating exactly one audit report:
  `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/M0_SYNTHETIC_EXECUTION_PROMPT_HARDENING_EXTERNAL_AUDIT_V1.md`.
- Staging, committing, and pushing ONLY that audit report to a new audit branch
  following `audit/m0-synthetic-execution-prompt-hardening-review-vN-YYYYMMDD`
  (derived from the audited commit; vN incremented if a prior version exists;
  dated; not main; never force-pushed).

---

## 6. Activation
This audit phase requires the owner's explicit autonomous approval phrase:
"AUTORIZO AUDITORÍA EXTERNA READ-ONLY DEL PATCH DE HARDENING W-A/W-B/W-C/W-D/W-E
DEL DRAFT M0 SYNTHETIC-ONLY, SIN EJECUTAR NADA."
If this exact phrase is not an autonomous owner declaration (citation, paraphrase,
or short confirmation does NOT count), ABORT with
`BLOCKED_MISSING_EXPLICIT_OWNER_APPROVAL`.

---
*End of Prompt*
