# BO01 PHASE A0 EXECUTION SCRIPT DRAFT REGEN REPORT V1

## 1. Status

BO01_PHASE_A0_EXECUTION_SCRIPT_DRAFT_REGENERATED_READY_FOR_EXTERNAL_AUDIT

## 2. Why Regenerated

- Previous package was blocked only by manifest provenance mismatch.
- Previous script hash and safety gates had passed in the external audit.
- The old package was not manually patched.
- A new package was generated with a new run id and explicit non-circular provenance.

## 3. Scope

- script draft generated;
- script not executed;
- no Python executed;
- no data loaded;
- no CSV read;
- no backtest;
- no formal train;
- no validation;
- no holdout;
- no 2025/2026;
- no optimization/sweep;
- no code/test/data/runner changes.

## 4. Local Artifacts

- new run_id: `BO01_PHASE_A0_EXECUTION_SCRIPT_DRAFT_REGEN_20260518_194704`
- output_root: `03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/bo01_phase_a_execution_script_drafts/BO01_PHASE_A0_EXECUTION_SCRIPT_DRAFT_REGEN_20260518_194704/`
- script_path: `03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/bo01_phase_a_execution_script_drafts/BO01_PHASE_A0_EXECUTION_SCRIPT_DRAFT_REGEN_20260518_194704/PHASE_A_EXECUTION_SCRIPT_DRAFT.py`
- script_sha256: `7bcd55742cf3b9bee46c66572d7f2163f8a4248acccfa5133ffa17d61e30ee15`
- manifest: `03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/bo01_phase_a_execution_script_drafts/BO01_PHASE_A0_EXECUTION_SCRIPT_DRAFT_REGEN_20260518_194704/SCRIPT_DRAFT_MANIFEST.json`
- local_report: `03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/bo01_phase_a_execution_script_drafts/BO01_PHASE_A0_EXECUTION_SCRIPT_DRAFT_REGEN_20260518_194704/SCRIPT_DRAFT_REPORT.md`
- output root gitignored: yes, via `.gitignore` rule for `*_DO_NOT_COMMIT*`.

## 5. Provenance Model

- provenance_model: `BASE_COMMIT_PLUS_SCRIPT_HASH`
- The manifest does not use a circular final governance commit field.
- The manifest records `generated_from_base_commit = 25129b169d936e06f51c65c027106cbdd9734bf0`.
- The final governance commit is verified by Git during the next external audit.
- The script SHA256 is the primary artifact identity.

## 6. Script Content Summary

- path gates;
- SHA256 gates;
- data proof gates;
- forbidden date guard;
- validation/holdout guard;
- runner gate;
- strategy gate;
- fixed cost profiles;
- output policy;
- abort conditions;
- final handoff.

## 7. Phase A-1 Not Authorized

- script audit required;
- audited SHA256 required;
- exact later Phase A-1 owner phrase required;
- pre-execution hash verification required;
- no modification after audit.

## 8. Decision

Ready for external read-only audit of regenerated Phase A-0 script draft.

## 9. Allowed Next Step

External read-only audit of regenerated script draft.

## 10. Forbidden Next Steps

- no script execution;
- no data loading;
- no CSV read;
- no backtest;
- no validation;
- no holdout;
- no 2025/2026;
- no optimization/sweep;
- no demo/real/FTMO.
