# PHASE 56H-B - GLOBAL CONSOLIDATION RECONCILIATION REPORT

**Date:** 2026-05-03T23:12:00
**Verdict:** **PHASE56H_B_GLOBAL_RESTATED_LOWER**

## Executive Summary
The reconciliation audit has identified a significant calculation error in the original Phase 56H report. The reported **+107.92 R** was an inflated value caused by the inclusion of 15 "unverified" months from legacy phases (selection bias sample) which were estimated using a flawed logic in the consolidator script.

The restated canonical total is **+43.34 R (Net FTMO)**.

## Audit Findings

### 1. Inconsistency Cause
The script `phase56h_consolidator.py` used in Phase 56H attempted to estimate "Net FTMO" R for months that only had "Base" R in the checkpoint. 
- **Legacy Months (15):** 2015-01 to 2015-11, 2017-05, 2017-08, 2020-04, 2024-10, 2025-02, 2025-11. These months had extremely high Base R (+79.95R).
- **Estimated Net R:** The script added ~70R to the total by estimating Net R for these months.
- **Canonical Summary:** The summary at the bottom of the checkpoint correctly excluded these 15 months, showing **+43.34 R**.

### 2. Duplicate Check
- **Duplicate Months:** NONE.
- **Mixed Sources:** YES. Base R (Selection Bias) was mixed with Net FTMO R (Audited Forensic).

### 3. Canonical Metrics (Restated)
- **Total Months:** 16 (Audited in Phase 56E-G)
- **Total Trades:** 322
- **Total R Net FTMO:** **+43.34 R**
- **PF Net FTMO:** **1.35** (Calculated from audited samples)
- **Expectancy Net FTMO:** **+0.1346 R**

## Reconciliation Verdict
**INVALIDATED (Phase 56H Results)**. 
The Phase 56H report is invalidated due to data inflation. The "Institutional Verdict" is downgraded from `EDGE_CONFIRMED` (PF 8.1) to `EDGE_CONFIRMED_WITH_WARNINGS` (PF 1.35) or `FRAGILE` depending on the risk model.

## Safety Validation
- MANIPULANTE Core: **UNTOUCHED**
- Strategy Lock: **PASS**
- MT5/Live: **ISOLATED**
- Git: **CLEAN**
