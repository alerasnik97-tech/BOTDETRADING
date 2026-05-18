# STRATEGY RESEARCH REGISTRY

## 1. Purpose
This registry serves as the official, decentralized, and audited source of truth for the lifecycle and status of all trading strategy candidates under active or historical investigation in the laboratory. It ensures complete traceability, statistical auditability, and mathematical safety, preventing data leakage, overfitting, or emotional recovery of rejected strategies.

This registry does not replace GitHub code repositories or individual dossier reports. Instead, it provides a high-level governance status matrix to guide deployment decisions, validation gating, and portfolio design.

## 2. Registry Rules
1.  **Unique Strategy ID:** Every strategy candidate investigated in the laboratory must be pre-registered with a unique ID following the family code taxonomy (e.g., `TP01`, `MR01`, `VE01`, `BO01`).
2.  **Explicit Status Gating:** No strategy can transition from one state to another (e.g., `TRAIN_RUN_PENDING` $\to$ `VALIDATION_APPROVAL_REQUIRED`) without updating this registry with a link to the corresponding formal audit report.
3.  **Strict Data Separation:** All strategies are locked in `TRAIN_ONLY` phase by default. Transition to `VALIDATION` or `HOLDOUT` is strictly forbidden unless all pre-conditions and gates are verified.
4.  **No Optimization Rescue:** If a strategy fails the train-only phase, it is permanently rejected. Tuning parameters or adding filters on the same dataset to "rescue" it is strictly prohibited. Any new hypothesis must be pre-registered as a separate strategy with a new ID.
5.  **No Edge Claim Without Gates:** A strategy cannot be declared to have an "edge" or be "champion" without passing the statistical, temporal, and economic gates documented in this registry.

## 3. Strategy Status Table

| Strategy ID | Family | Version | Status | Hypothesis | Branch | Latest Commit | Train Status | Validation Status | Holdout Status | Latest Audit | Classification | Sample Size | Active Years | PF Base | PF Stress | Expectancy (R) | Max DD (%) | Avg Trades/Month | Cost Degradation | Temporal Concentration | Correlation Notes | Blocked Actions | Next Allowed Action | Decision Date |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **VEORB** | Volatility Expansion | v1.0 | `REJECTED_REGIME_OBSOLETE` | Volatility expansion breakout of London opening range on M15 bars. | `research/veorb-official-runner-run-20260517` | `937376fd0e0d7433a1018b80e241a0de34173175` | `PASS` | `LOCKED` | `SEALED` | `VEORB_REGENERATED_DOSSIER_EXTERNAL_AUDIT_V1.md` | `VEORB_PRELIMINARY_INTERESTING_NEEDS_AUDIT` | 15 | 1 (2015) | 1.0620 | 1.0308 | 0.0361 | 1.52% | 0.1250 | Low degradation | 100% (2015) | Independent | No optimization, no validation, no holdout | Archive only, watch regimes | May 17, 2026 |
| **TP01** | Momentum Pullback | v1.0 | `REJECTED_LOW_EDGE_AND_REGIME_OBSOLESCENCE` | Intraday daytime momentum pullback entry on M5. | `audit/tp01-formal-train-run-v1-20260517` | `7f76acf7ac5bda582404ff86c4fcc37a7fd0d159` | `PASS` | `LOCKED` | `SEALED` | `TP01_FORMAL_TRAIN_EXTERNAL_AUDIT_V1.md` | `TP01_OFFICIALLY_REJECTED_LOW_EDGE_AND_REGIME_OBSOLESCENCE` | 191 | 4 (2015-2018) | 0.6312 | 0.5695 | -0.2839 | 27.35% | 1.5917 | High sensitivity | Stable (2015-2017) | Independent | No optimization, no validation, no holdout | Archive only, negative control | May 17, 2026 |
| **BO01** | London Breakout | v1.0 | `M0_SYNTHETIC_EXECUTION_PROMPT_DRAFT_PENDING_AUDIT` | London-session breakout of the Asian range (00:00-06:30 GMT) on M5 bars. | `research/draft-m0-synthetic-execution-prompt-v1-20260517` | `0743ad83c1c61e0a2dc8e269d5b70f3b6a506bc1` | `NOT_RUN` | `LOCKED` | `SEALED` | `MICRORUN_PROTOCOL_DESIGN_EXTERNAL_AUDIT_V1.md` | `SKELETON_PLUS_TESTS_NO_EDGE_NO_PERFORMANCE` | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | No micro-run, no dry-run, no backtest, no formal train, no validation, no holdout, no 2025/2026, no optimization/sweep, no Sub-Batch 1B, no parallel writers | External read-only audit of M0 synthetic execution prompt draft (design only; no execution; no micro-run; no dry-run; no backtest; no train; no validation; no holdout; no 2025/2026; no optimization/sweep; external audit required before any use) | May 17, 2026 |
| **MR02** | London Fakeout Reversion | v1.0 | `M0_SYNTHETIC_EXECUTION_PROMPT_DRAFT_PENDING_AUDIT` | Mean-reversion fakeout of the Asian range (00:00-06:30 GMT) on M5 with engulfing confirmation. | `research/draft-m0-synthetic-execution-prompt-v1-20260517` | `0743ad83c1c61e0a2dc8e269d5b70f3b6a506bc1` | `NOT_RUN` | `LOCKED` | `SEALED` | `MICRORUN_PROTOCOL_DESIGN_EXTERNAL_AUDIT_V1.md` | `SKELETON_PLUS_TESTS_NO_EDGE_NO_PERFORMANCE` | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | No micro-run, no dry-run, no backtest, no formal train, no validation, no holdout, no 2025/2026, no optimization/sweep, no Sub-Batch 1B, no parallel writers | External read-only audit of M0 synthetic execution prompt draft (design only; no execution; no micro-run; no dry-run; no backtest; no train; no validation; no holdout; no 2025/2026; no optimization/sweep; external audit required before any use) | May 17, 2026 |

## 3.1 Pre-Registered Skeleton-Stage Candidates — BO01, MR02

This subsection reconciles audit warning **W-03** (registry previously had no BO01/MR02 rows
despite skeletons and tests existing, which contravened the Registry Maintenance Protocol §1).

The following holds for **BO01** and **MR02** at the commit recorded above:

- BO01/MR02 do NOT have any demonstrated edge.
- BO01/MR02 do NOT have any performance result.
- BO01/MR02 are NOT validated.
- BO01/MR02 did NOT use validation data.
- BO01/MR02 did NOT use holdout data.
- BO01/MR02 did NOT use 2025/2026 data.
- BO01/MR02 have NO micro-run.
- BO01/MR02 have NO dry-run.
- BO01/MR02 have NO backtest.
- BO01/MR02 have NO formal train.
- BO01/MR02 are strategy code skeletons plus unit/contract tests only.
- The Sub-Batch 1A blocker patch passed external read-only audit; the extreme
  end-to-end audit returned PASS with documented warnings and no blockers.

Per-strategy governance record:

| Field | BO01 | MR02 |
| :--- | :--- | :--- |
| strategy_id | BO01 | MR02 |
| family | London Breakout (`LBC`) | London Fakeout Reversion (`LBF`) |
| current_state | `M0_SYNTHETIC_EXECUTION_PROMPT_DRAFT_PENDING_AUDIT` | `M0_SYNTHETIC_EXECUTION_PROMPT_DRAFT_PENDING_AUDIT` |
| branch | `research/draft-m0-synthetic-execution-prompt-v1-20260517` | same |
| commit | `0743ad83c1c61e0a2dc8e269d5b70f3b6a506bc1` | same |
| evidence_artifact | `M0_SYNTHETIC_EXECUTION_PROMPT_DRAFT_REPORT_V1.md` | same |
| allowed_next_action | External read-only audit of M0 synthetic execution prompt draft (design only; no execution; no micro-run; no dry-run; no backtest; no train; no validation; no holdout; no 2025/2026; no optimization/sweep; external audit required before any use) | same |
| forbidden_actions | micro-run, dry-run, backtest, formal train, validation, holdout, 2025/2026, optimization/sweep, Sub-Batch 1B, parallel writers | same |
| owner_gate_required | Yes | Yes |
| audit_required_before_execution | Yes — draft must be externally audited, then separate owner approval, then a separate external audit before any execution | Yes |

No edge, performance, profitability, champion, demo, real, or FTMO status is
asserted for BO01 or MR02. Their presence in this registry records lifecycle
state only, not merit.

## 3.2 TP-01 Lineage Traceability Note

This subsection reconciles audit warning **W-05** (the registry cites a
different lineage than the metric-fixed regeneration).

- TP-01 is officially rejected. The rejection decision does NOT change.
- Canonical rejection classification remains
  `TP01_OFFICIALLY_REJECTED_LOW_EDGE_AND_REGIME_OBSOLESCENCE`.
- Documented lineages (read-only evidence, not modified here):
  - Registry row "Latest Commit" cell: `7f76acf7ac5bda582404ff86c4fcc37a7fd0d159`
    — this hash is also the planning-branch / registry-report commit
    (`RESEARCH_REGISTRY_AND_FIRST_BATCH_REPORT_V1.md` "Active Commit";
    `NEXT_PROMPT_FIRST_BATCH_IMPLEMENTATION_SPECS_V1.md`), recorded on
    branch `audit/tp01-formal-train-run-v1-20260517`.
  - TP-01 formal-train external audit (`TP01_FORMAL_TRAIN_EXTERNAL_AUDIT_V1.md`)
    states research branch `research/tp01-formal-train-run-v1-20260517` and
    audit commit `ba9b81d7442eb744a4e8a158b2a551068f9f0fce`.
  - Metric-fixed regeneration (`TP01_POST_FIX_RECONCILIATION_REPORT.md`,
    `TP01_REGENERATED_DOSSIER_EXTERNAL_AUDIT_V3_REPORT.md`): branch
    `research/tp01-official-runner-regeneration-v2-20260517`, commit
    `c1dd15872d448165539aacdf81f9b6912018a313`, run
    `TP01_OFFICIAL_RUNNER_REGEN_2015_2024_20260517_132002`.
- All three lineages converge on the SAME verdict: negative expectancy and
  regime obsolescence → permanent rejection. No rescue, no optimization, no
  validation, no holdout, no 2025/2026.
- Exact canonical-commit reconciliation across these branches is beyond a
  markdown hygiene patch and is therefore marked
  `TRACEABILITY_NOTE_PENDING_OWNER_REVIEW`: a separate read-only lineage
  audit should formally designate the single canonical commit. This note
  does not alter the rejection.

## 4. Registry Maintenance Protocol
The Strategy Research Registry must be updated immediately upon the completion of any of the following events:
1.  **Strategy Pre-Registration:** Before writing any code or executing dry-runs, a new entry with status `PRE_REGISTERED` must be added.
2.  **Tests and Contract Verification:** Once the code is written and the targeted contract/preflight unit tests pass, status shifts to `IMPLEMENTED_TESTS_PENDING` and then, after an external read-only audit of the skeleton/tests, to `IMPLEMENTED_TESTS_AUDITED_OWNER_PROTOCOL_DECISION_PENDING`. Passing tests (and passing the external audit) does NOT shift status to `TRAIN_RUN_PENDING` and does NOT authorize a micro-run, dry-run, backtest, or formal train. Any transition toward a micro-run protocol, a micro-run preflight, or a sealed train backtest requires an explicit owner decision plus the gates defined in `STRATEGY_STATUS_TAXONOMY.md`.
3.  **Train Run Completion:** After the official runner execution completes and seals, status shifts to `TRAIN_GATE_GREEN_NEEDS_AUDIT` or `TRAIN_GATE_FAILED` based on the reconciliation outcomes.
4.  **External Audit Verdict:** Upon completion of the formal read-only external audit, the status is updated to `REJECTED_*` or `VALIDATION_APPROVAL_REQUIRED`.
5.  **Validation/Holdout Completion:** Updates are added after validation or holdout unsealing, strictly under explicit owner authorization.
6.  **Retirement:** Any live strategy showing regime decay shifts to `RETIRED`.
