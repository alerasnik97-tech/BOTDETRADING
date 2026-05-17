# NEXT PROMPT: INSTITUTIONAL RESEARCH ACCELERATION AFTER TP01 REJECTION

**Document Reference:** GOV-PRM-ACCEL-V1-20260517  
**Status:** COMPLIANT  
**Date:** May 17, 2026  

---

## 1. Context & Rationale
Strategy `tp01_london_ny_momentum_pullback` (TP-01) has been officially audited and rejected as `TP01_OFFICIALLY_REJECTED_LOW_EDGE_AND_REGIME_OBSOLESCENCE` under branch `audit/tp01-formal-train-run-v1-20260517`. The laboratory has successfully preserved validation and holdout resources.

To scale the research pipeline professionally, the next phase is to initiate the **Institutional Research Acceleration Plan**. This involves setting up the systematic research registry, specifying formal rejection gates, classifying systematic strategy families (Momentum, Mean Reversion, Breakout, etc.), and pre-registering the first batch of strategy skeletons without executing backtests yet.

---

## 2. Research Target & Branching
-   **Active Branch:** `research/institutional-research-acceleration-20260517`
-   **Base Commit:** `ba9b81d7442eb744a4e8a158b2a551068f9f0fce` (or current audited head)
-   **Scope:** Read-Only Planning, Architecture & Intake Registry. **NO backtesting, NO data loading, NO holdout exposure**.

---

## 3. Mandatory Tasks for Next Agent
Act as a **Senior Quant Portfolio Architect** and **Governance Compliance Officer** to:

1.  **Create Research Registry:**
    -   Initialize the centralized Research Intake Registry inside `03_RESEARCH_LAB/strategy_research_intake/RESEARCH_INTAKE_REGISTRY.csv`.
    -   Define unique keys for strategy candidates (e.g., family prefix, numeric suffix, sub-variant).
2.  **Define Formal Rejection Gates:**
    -   Document strict multi-stage quantitative rejection gates (statistical, economic, temporal activity) to automate early failure detection.
3.  **Classify Strategy Families:**
    -   Draft the structural taxonomy of target trading concepts (e.g., Session Breakouts, Daily Bias Followers, Elastic Mean Reversions) tailored to intraday EURUSD.
4.  **Pre-Register First Batch of Skeletons:**
    -   Add initial priority strategy skeletons to the registry backlog (e.g., MR-01, MR-02, VE-01) with defined entry hypotheses and parameters.
5.  **Output Staging Security:**
    -   Ensure no code or data files are modified. Commit only planning CSV and markdown files under `03_RESEARCH_LAB` and `06_GOVERNANCE_AND_COMPLIANCE`.

---

## 4. Final Handoff Format
Your final report must present:
1.  **ACCELERATION STATUS:** READY / BLOCKED
2.  **TAXONOMY SCHEMAS:** Overview of the systematic family classifications.
3.  **REGISTRY ARTIFACTS:** Paths of the newly created intake files.
4.  **NEXT SEED RECOMMENDATION:** Confirmation of the first candidate skeleton selected for implementation.

*End of Acceleration Prompt*
