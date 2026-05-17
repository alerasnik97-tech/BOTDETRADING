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

## 4. Registry Maintenance Protocol
The Strategy Research Registry must be updated immediately upon the completion of any of the following events:
1.  **Strategy Pre-Registration:** Before writing any code or executing dry-runs, a new entry with status `PRE_REGISTERED` must be added.
2.  **Tests and Contract Verification:** Once the code is written and targeted preflight unit tests pass, status shifts to `TRAIN_RUN_PENDING`.
3.  **Train Run Completion:** After the official runner execution completes and seals, status shifts to `TRAIN_GATE_GREEN_NEEDS_AUDIT` or `TRAIN_GATE_FAILED` based on the reconciliation outcomes.
4.  **External Audit Verdict:** Upon completion of the formal read-only external audit, the status is updated to `REJECTED_*` or `VALIDATION_APPROVAL_REQUIRED`.
5.  **Validation/Holdout Completion:** Updates are added after validation or holdout unsealing, strictly under explicit owner authorization.
6.  **Retirement:** Any live strategy showing regime decay shifts to `RETIRED`.
