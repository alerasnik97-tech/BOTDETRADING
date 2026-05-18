# NEXT PROMPT - FIX BO01 PHASE A EXECUTION SCRIPT DRAFT BLOCKER V1

Use this only if the owner explicitly authorizes a new Phase A-0 script draft generation.

## Required Owner Activation Phrase

"AUTORIZO REGENERAR EL SCRIPT DRAFT PHASE A-0 BO01 TRAIN-ONLY REAL-DATA PARA CORREGIR EL BLOCKER DE MANIFEST, SIN EJECUTAR PYTHON, SIN EJECUTAR EL SCRIPT, SIN CARGAR DATOS DE MERCADO, SIN LEER CSV REAL, SIN BACKTEST CON REAL DATA, SIN TRAIN, SIN VALIDATION, SIN HOLDOUT, SIN 2025/2026 Y SIN OPTIMIZATION/SWEEP."

If the phrase is missing exactly, abort with:

`BLOCKED_MISSING_EXPLICIT_OWNER_APPROVAL`

## Blocker To Fix

Audit verdict:

`AUDIT_BLOCKED_MANIFEST_INCONSISTENT`

The local script SHA256 matched the declared script hash, but
`SCRIPT_DRAFT_MANIFEST.json` declared:

- observed `generated_from_commit`: `1c55ecbd42250c6d041471c9ce2d6b399e9d6966`
- expected audited commit: `e65a7bf39ff6b73eff13a04a83e40d7be79e52ec`

## Required Correction

Generate a new Phase A-0 script-draft package. Do not manually patch the previously
audited script, manifest, or local report.

The new package must include:

- new `RUN_ID`;
- new local gitignored output root;
- new `PHASE_A_EXECUTION_SCRIPT_DRAFT.py`;
- new `SCRIPT_DRAFT_MANIFEST.json`;
- new `SCRIPT_DRAFT_REPORT.md`;
- updated governance report;
- updated next read-only script audit prompt.

The new manifest must make provenance unambiguous. It must either:

- record `generated_from_commit` equal to the final audited branch commit that will be
  reviewed; or
- split provenance into explicit fields such as `generated_from_base_commit`,
  `governance_commit`, and `audited_branch_head`, with the next audit prompt requiring
  those exact fields and values.

## Hard Prohibitions

- no Python execution;
- no script execution;
- no script import;
- no `py_compile`;
- no tests;
- no data loading;
- no CSV read;
- no M5/M15 read;
- no data vault access;
- no backtest;
- no train;
- no validation;
- no holdout;
- no 2025/2026;
- no optimization/sweep/grid search/walk-forward/parameter search;
- no demo/real/FTMO;
- no edge/profitability claims;
- no direct Phase A-1.

## Required Verification

Before committing governance docs:

- `Get-FileHash` on the new script;
- manifest SHA256 equals physical script SHA256;
- manifest provenance matches the audit target semantics;
- local output root is gitignored;
- local artifacts are not tracked;
- staged files include only authorized governance markdowns.

## Required Next Step After Fix

External read-only audit of the regenerated Phase A-0 script draft.
