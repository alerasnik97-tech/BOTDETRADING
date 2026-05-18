# NEXT TRAIN-ONLY PROTOCOL BO01/MR02 DESIGN REPORT V1

## 1. Status
**`NEXT_TRAIN_ONLY_PROTOCOL_DESIGN_READY_FOR_EXTERNAL_AUDIT`**

## 2. Preconditions
- **M1 Execution Audited:** `YES` (verified plumbing integrity)
- **M1 Audit Remote Verified:** `YES` (Head SHA `10f2caf8507c135c59a66505b3ee36d19ed301ba` matches exactly on GitHub)
- **No Blockers:** `YES` (0 blockers)
- **Warnings Known:** `YES` (Windows CRLF newline, placeholder `<COMMIT_SHA>`, W-01/W-02 dirty backlogs documented)

---

## 3. Scope
This is a **MARKDOWN-ONLY** report presenting the proposed protocol design for the next research phase.
- **NO execution was performed.**
- **NO data was loaded.**
- **NO backtesting, training, validation, or holdout operations were conducted.**
- **NO 2025/2026 data was processed.**
- **NO parameter sweeps or optimizations were done.**

---

## 4. Protocol Summary
The proposed protocol, **`M2_TRAIN_ONLY_LIMITED_STRUCTURAL_EVALUATION_PROTOCOL`**, designs a controlled structural evaluation of breakout strategy `BO01` and mean reversion strategy `MR02` over a pre-declared train-only window. 
Its core goal is to evaluate operational stability, session concentration, hourly/monthly signal distributions, and fail-closed contract safety without calculating performance metrics (PnL, Drawdown, Sharpe, Win Rate).

---

## 5. Recommendation
**`Recommend M2 Conservative 3-month structural evaluation (Option A)`**

### Justification
- Running a conservative 3-month slice (`2015-01-01` to `2015-03-31`) provides sufficient statistical signal frequency for `BO01` and `MR02` while maintaining highly localized resource usage.
- This allows verifying contract compliance and session filters under a wider, but highly controllable, train-only time-series without overloading disk space or running heavy diagnostic sweeps.
- Proceeding straight to Option B (12 months) would bypass the incremental verification principle required by the quantitative risk guidelines.

---

## 6. Allowed Next Step
- **External read-only audit of next train-only protocol design.**

---

## 7. Forbidden Next Steps
- **NO immediate M2 execution.**
- **NO backtesting or training runs.**
- **NO validation or holdout dataset access.**
- **NO 2025/2026 data processing.**
- **NO parameter optimization or sweeps.**
