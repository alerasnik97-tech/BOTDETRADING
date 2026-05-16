# Gemini-to-Claude Delta Report

**Date**: 2026-05-16
**Auditor**: Claude Opus 4.7
**Subject**: Comparison of Gemini 3 Flash strategy intake vs Claude Opus 4.7 independent audit
**Branch**: governance/claude-strategy-intake-audit-20260516

---

## 1. Scope

Gemini 3 Flash read 6 external research documents and produced a 25-hypothesis backlog with Priority A/B/C classification. This report documents every material difference found by an independent re-read of the same sources.

---

## 2. Source Accessibility

| Source | Gemini Claims Read | Claude Verified |
|--------|-------------------|-----------------|
| Source 1: GPT Report (38pp) | YES | YES — FULL_READ |
| Source 2: NY Report (17pp) | YES | YES — FULL_READ |
| Source 3: MD Report (175KB) | YES | YES — FULL_READ |
| Source 4: grok_report.pdf (9pp) | YES | **NO — SOURCE_READ_FAILURE (image-based)** |
| Source 5: grok_report 2.pdf (10pp) | YES | **NO — SOURCE_READ_FAILURE (image-based)** |
| Source 6: Investigacion (22pp) | YES | YES — FULL_READ |

**Impact**: Cannot verify Gemini's claims about Sources 4-5. However, no Priority A candidate depends solely on these sources. Audit risk: LOW.

---

## 3. Priority A Classification Errors

| ID | Gemini Priority | Claude Audit | Change | Reason |
|----|----------------|--------------|--------|--------|
| MR-01 | A | A | NONE | Confirmed |
| VE-01 | A | A | NONE | Confirmed |
| TP-01 | A | A | NONE | Confirmed |
| SD-01 | A | A | NONE | Confirmed |
| **ED-01** | **A** | **DEFER_NEWS** | **DOWNGRADED** | **Requires uncertified news calendar data** |

**Critical Finding**: Gemini classified ED-01 (Post-News Stabilization) as Priority A and included it in STRATEGY_SKELETON_PROMPT.md for immediate implementation. This is a **GOVERNANCE VIOLATION** because:
1. The strategy explicitly requires news calendar timestamps (Source 1: "Required Data: Base Pack + timestamp de noticia de alto impacto")
2. `forex_factory_cache.csv` is MISSING from the data pipeline
3. `news_eurusd_v2_utc.csv` is MISSING from the data pipeline
4. Implementing a skeleton that cannot be backtested wastes engineering effort and violates TRAIN-ONLY protocol

---

## 4. Priority C → B Upgrades

| ID | Name | Gemini Priority | Claude Audit | Reason |
|----|------|----------------|--------------|--------|
| MR-03 | London Close Mean Reversion | C | B | Strong multi-source support (3 sources), well-specified, clean data requirements |
| SE-01 | Friday Reversion | C | B | Calendar anomaly well-documented in Source 6, BIS-backed time-of-week effect |
| SE-02 | London Lunch Fade | C | B | Strong support from Source 1 (GPT Report, explicit "London Lunch Fade" entry) and Source 2 |

---

## 5. New Status Categories (not in Gemini's taxonomy)

| Category | Count | Purpose |
|----------|-------|---------|
| DEFER_NEWS | 3 | Blocked on missing data — not rejected, just unimplementable now |
| REVIEW_NEEDS_SPECIFICATION | 3 | Cannot implement without further definition work |
| REJECT_DISCRETIONARY | 1 | Fundamentally unimplementable as algorithmic strategy |

Gemini used only A/B/C. This flattens critical distinctions between "good but blocked" (DEFER) and "implementable but low priority" (C).

---

## 6. Strategies Gemini MISSED from Sources

Gemini extracted 25 hypotheses from ~80 unique strategy ideas across all sources. The following high-quality candidates were omitted:

### From Source 1 (GPT Report) — 8 missed:
1. **Handoff Box Breakout** — Session breakout at LDN→NY handoff, well-specified
2. **Sigma Exhaustion Fade** — 2.5sigma extreme + momentum decay, Priority B quality
3. **Regime Shift Continuation** — RV regime change trigger
4. **Coil Release** — Tick-count compression breakout (requires tick data)
5. **Session Midpoint Snapback** — Midday MR to session midpoint
6. **Midday Re-Expansion** — Post-lunch volatility expansion
7. **Anchor Pullback Continuation** — Second pullback to APM in trend
8. **Spread Shock Fade** — Spread percentile extreme reversion (requires spread p90)

### From Source 2 (NY Report) — 4 missed:
1. **Late-Session Trend Pullback** — 12:00-18:30 pullback continuation
2. **Trend + Compression Pullback Hybrid** — Multi-factor trend continuation
3. **Monday NY Mean Reversion** — Day-of-week MR anomaly
4. **ATR Stretch Snapback** — Pure ATR extreme reversion

### From Source 6 (Investigacion) — 2 missed:
1. **London Fix Fade** (MR_LondonFix_Fade) — End-of-month calendar anomaly, BIS-documented
2. **TP_HVN_Retest** — Volume Profile HVN retest (requires institutional tick data)

**Total missed**: ~14 unique strategy ideas not present in Gemini's backlog.

**Assessment**: Gemini's coverage is ~31% of total strategy ideas (25 out of ~80). However, the top 5 from the primary source (GPT Report) were captured correctly, and most missed strategies are Priority B/C quality. The key gap is that some missed strategies (Handoff Box, Sigma Exhaustion) appear to be Priority B quality and were omitted without explanation.

---

## 7. Scoring Methodology Comparison

| Aspect | Gemini | Claude Assessment |
|--------|--------|-------------------|
| Scale | 68-95 | Reasonable spread |
| Top score | 95 (MR-01) | Appropriate — strongest multi-source support |
| Bottom score | 68 (SD-05) | Appropriate — weakest specification |
| Differentiation | 1-point gaps in B/C tier | Insufficient — 13 strategies within 8 points |
| Source-weighted | Unclear | Should weight by source rigor (Source 6 > Source 3) |

---

## 8. Correlation Risk Assessment (Gemini vs Claude)

Gemini did NOT flag correlation risk with existing Manipulante (liquidity sweep) strategy. Source 2 explicitly warns:

> "Reject if correlated with existing strategy outcomes"

And specifically flags these as HIGH correlation:
- Asia-to-NY Range Failure → maps to SD-03 (Asian Range Fakeout)
- Opening Sweep Failure → similar to SD-01/SD-05
- Session High/Low Rejection → similar to SD-01

**Gemini's gap**: No correlation analysis was performed. SD-03 was classified Priority B without noting the explicit source warning.

---

## 9. Numbering Error

Gemini's MD backlog skips entry #10 (jumps from #9 to #11). The CSV contains all 25 rows correctly. This is a formatting oversight, not a data loss issue. SD-03 (Asian Range Fakeout) should have been #10.

---

## 10. STRATEGY_SKELETON_PROMPT.md Governance Issue

Gemini's skeleton prompt directs implementation of 5 Priority A strategies including ED-01. This creates a governance violation:
- ED-01 requires news calendar data that does not exist in the pipeline
- Implementing the skeleton would produce a strategy that cannot be backtested
- The prompt should direct implementation of 4 strategies only (MR-01, VE-01, TP-01, SD-01)

**Resolution**: A corrected prompt has been produced (see NEXT_PROMPT_IMPLEMENT_CLAUDE_APPROVED_PRIORITY_A_SKELETONS.md)

---

## 11. Data Certification Status

| Data Type | Status | Impact |
|-----------|--------|--------|
| OHLCV M1/M5/M15 | CERTIFIED | All Priority A use this |
| ATR(14) | CERTIFIED (derivable) | All strategies |
| ADX(14) | CERTIFIED (derivable) | MR-01, TP-01 |
| EMA20/50 | CERTIFIED (derivable) | TP-01 |
| VWAP anchored 07:00 | CERTIFIED (derivable from M1) | MR-01, MR-02 |
| Donchian channels | CERTIFIED (derivable) | VE-01 |
| Session H/L | CERTIFIED (derivable) | SD-01, SD-02 |
| rv5/rv15 | CERTIFIED (computable from M1 returns) | VE-01 |
| Spread (bid-ask) | CERTIFIED (in tick data) | All (spread_ok filter) |
| **News calendar** | **MISSING** | **ED-01, ED-02, ED-03 BLOCKED** |
| Volume Profile | NOT AVAILABLE | Source 6 strategies only |
| VIX/implied vol | NOT AVAILABLE | HY_VolTrend_Sync only |

---

## 12. Summary of Changes

| Change Type | Count | Details |
|-------------|-------|---------|
| Priority A downgraded | 1 | ED-01 → DEFER_NEWS |
| Priority C upgraded to B | 3 | MR-03, SE-01, SE-02 |
| New category: DEFER_NEWS | 3 | ED-01, ED-02, ED-03 |
| New category: REVIEW_NEEDS_SPECIFICATION | 3 | HY-01, TP-03, HY-02 |
| New category: REJECT_DISCRETIONARY | 1 | SD-05 |
| Correlation risk flagged | 2 | SD-03 (HIGH), SD-01 (MODERATE) |
| Strategies missed by Gemini | ~14 | See Section 6 |
| Governance violation identified | 1 | ED-01 in skeleton prompt |
| Total hypotheses retained | 25 | No additions (audit scope = verify, not expand) |

---
Generated: 2026-05-16 by Claude Opus 4.7
