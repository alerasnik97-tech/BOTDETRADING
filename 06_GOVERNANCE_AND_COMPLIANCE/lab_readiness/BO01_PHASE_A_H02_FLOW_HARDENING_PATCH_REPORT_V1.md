# BO01 PHASE A H02 FLOW HARDENING PATCH REPORT V1

## 1. Status

BO01_PHASE_A_H02_FLOW_HARDENING_PATCH_READY_FOR_EXTERNAL_AUDIT

## 2. Scope

- markdown only;
- no Python;
- no scripts created or executed;
- no data loading;
- no real-data backtest;
- no formal train;
- no validation;
- no holdout;
- no 2025/2026;
- no optimization/sweep;
- no code/test/data changes;
- no runner / BO01Strategy / MR02Strategy / engine / data_loader / data-vault changes.

Base branch: `audit/project-extreme-readonly-audit-v1-20260518` @
`137cfd576e4be108ef04b1304bca239099203252`.
Patch branch: `research/bo01-phase-a-h02-flow-hardening-v1-20260518`.

## 3. H-02 Addressed

H-02 (the path / partition / 2025-2026 / SHA256 / monotonicity proof lived in an
optional, unaudited loader / temporary execution script; the audited runner was only a
partial backstop) is NOT closed by subjective owner acceptance. It is converted into an
auditable, three-gate technical control written into the Phase A execution prompt draft:

- **Phase A-0 — script generation only**: produces the exact execution / data-proof
  script. It does not load data, does not read real CSV, does not read M5/M15, does not
  read `05_MARKET_DATA_VAULT` content, does not run Python on the script, does not run a
  backtest, does not compute metrics, and does not generate trades/equity. It only writes
  the script (plus its manifest/report) to a gitignored local outputs folder, with the
  governance draft report and the script-audit next-prompt as the only committable docs.
- **Script audit — dedicated read-only audit**: the generated script is audited
  destructively before any execution. The audit records the script SHA256.
- **Phase A-1 — execution only after audit passes**: Phase A-1 runs the unmodified
  script only after the script audit passes; before running it recomputes and verifies
  the script SHA256 against the audited hash, aborting with
  `BLOCKED_SCRIPT_HASH_MISMATCH` on any mismatch. Any modification of the script after
  the audit invalidates execution and forces re-audit. Direct Phase A execution from the
  prior prompt (without A-0 + script audit + A-1 hash check) is explicitly forbidden.

The Phase A handoff template was updated with a `phase: (A0 / A1)` field, expanded
STATUS values, and a `SCRIPT_AUDIT` block (audited vs recomputed script SHA256, hash
match, script_audit_passed, script_modified_after_audit).

## 4. H-01 Registered

H-01 (`ema_m15_200` / `atr14` causal correctness depends on the unaudited data-prep
pipeline) is formally pre-registered as a mandatory pre-Phase-B audit:

- H-01 does NOT block Phase A plumbing (A-0 or A-1), because Phase A draws no qualitative
  conclusions.
- H-01 DOES block Phase B and any edge / profitability interpretation, and blocks
  widening the window to `2015-01-01` / `2015-03-31` with behavioural conclusions.
- The Phase A prompt now lists the minimum causality checks required by the future H-01
  audit (resample rules, `merge_asof` / forward-fill, closed/label policy, M15 EMA on
  completed M15 bars only, ATR on completed bars only, timezone alignment, no future-bar
  leakage, no centered rolling, no `shift(-1)`, no accidental future High/Low/Close).
- H-01 is not considered resolved by this task; it remains pending its own dedicated
  read-only audit.

## 5. Files Patched

Existing markdowns modified (3):
1. `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/BO01_PHASE_A_TRAIN_ONLY_REALDATA_BACKTEST_EXECUTION_PROMPT_DRAFT_V1.md`
   (added section 2 split, section 2-BIS Phase A-0/A-1 gating, section 16 H-01
   pre-Phase-B blocker, handoff `phase`/STATUS/`SCRIPT_AUDIT` fields).
2. `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_BO01_PHASE_A_EXECUTION_PROMPT_DRAFT_V1.md`
   (added verification steps 8 and 9 for the H-02 flow split and H-01 pre-registration).
3. `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_OWNER_DECIDES_AFTER_PROJECT_EXTREME_AUDIT_V1.md`
   (withdrew the "owner accepts H-02 then runs Phase A" path; rewrote options A–D and
   the recommended sequence to the three-gate flow).

New markdowns created (2):
4. `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/BO01_PHASE_A_H02_FLOW_HARDENING_PATCH_REPORT_V1.md` (this file).
5. `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_BO01_PHASE_A_H02_FLOW_HARDENING_PATCH_V1.md`.

No other file was touched. No code, test, data, output, or data-vault file was modified.

## 6. Decision

Ready for external read-only audit of this H-02 flow hardening patch. This patch does not
authorize Phase A execution, does not load data, does not declare edge or profitability,
and does not authorize demo/real/FTMO.

## 7. Allowed Next Step

External read-only audit of this patch, using
`NEXT_PROMPT_AUDIT_BO01_PHASE_A_H02_FLOW_HARDENING_PATCH_V1.md`.

## 8. Forbidden Next Steps

- no direct Phase A execution yet;
- no Phase A-0 script generation until this patch is audited;
- no data loading;
- no real CSV reading;
- no backtest;
- no formal train;
- no validation;
- no holdout;
- no 2025/2026;
- no optimization/sweep;
- no demo/real/FTMO;
- no edge / profitability / rentabilidad claims;
- no modification of the audited runner or strategy classes.
