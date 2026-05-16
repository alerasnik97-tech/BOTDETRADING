# EURUSD Rejection and Deferred Log — Claude Audit

**Date**: 2026-05-16
**Auditor**: Claude Opus 4.7
**Branch**: governance/claude-strategy-intake-audit-20260516

---

## REJECTED Strategies

### SD-05: Programmable Structure Break + Fill
- **Original Priority**: C (Gemini)
- **Audit Decision**: REJECT_DISCRETIONARY
- **Family**: Session Dynamics
- **Scoring**: 68/100 (lowest in backlog)

**Rejection Reasons**:
1. **Discretionary entry**: "Cierre M5 tras mitigacion de zona de valor" — "zone of value" has no rigorous quantitative definition. ICT/SMC "order blocks" and "fair value gaps" are visual patterns, not computable signals.
2. **Variable exit**: "Variable segun estructura" — undefined algorithmically.
3. **HIGH correlation with Manipulante**: Source 2 (NY Report) explicitly flags sweep-based session strategies as highly correlated with existing liquidity sweep strategy. Direct quote: "Reject if correlated with existing strategy outcomes."
4. **Weakest source support**: No source provides a fully-specified algorithmic implementation. The concept relies on discretionary chart reading.
5. **Lowest scoring**: 68/100, bottom of the entire backlog.

**Reversal Condition**: Could be reconsidered IF:
- A precise, computable definition of "structural imbalance" is provided (e.g., volume-at-price gap > X ATR)
- Correlation with Manipulante is tested and found < 0.3 on daily PnL
- Entry/exit rules are made fully deterministic

---

## DEFERRED Strategies (Blocked on Missing Data)

### ED-01: Post-News Stabilization
- **Original Priority**: A (Gemini) — **GOVERNANCE VIOLATION**
- **Audit Decision**: DEFER_NEWS
- **Family**: Event Driven
- **Scoring**: 88/100

**Deferral Reasons**:
1. Source 1 (GPT Report) explicitly states required data: "Base Pack + timestamp de noticia de alto impacto + spread pre/post news"
2. `forex_factory_cache.csv` — NOT FOUND in data pipeline
3. `news_eurusd_v2_utc.csv` — NOT FOUND in data pipeline
4. Without news calendar, strategy cannot identify "post-news" windows
5. Without pre/post spread measurement, cannot determine "spread normalization"
6. Implementing skeleton without data = wasted engineering + TRAIN-ONLY violation

**Unblock Conditions**:
- [ ] Provide `forex_factory_cache.csv` or equivalent with columns: `[datetime_utc, currency, impact_level, event_name]`
- [ ] Cover full train period (2017-2024 minimum)
- [ ] Validate timestamps against known events (NFP, CPI, ECB, FOMC)
- [ ] Verify spread data captures pre/post event spread widening

---

### ED-02: Post-News Volatility Reversion
- **Original Priority**: C (Gemini)
- **Audit Decision**: DEFER_NEWS
- **Family**: Event Driven
- **Scoring**: 73/100

**Deferral Reasons**:
1. Same news calendar dependency as ED-01
2. Requires identifying "pre-news ATR" baseline (needs event timestamp)
3. Source 1 describes this as "News Overreaction Fade" with explicit news feed requirement

**Unblock Conditions**: Same as ED-01

---

### ED-03: PNMC-15 Momentum
- **Original Priority**: C (Gemini)
- **Audit Decision**: DEFER_NEWS
- **Family**: Event Driven
- **Scoring**: 72/100

**Deferral Reasons**:
1. Same news calendar dependency as ED-01
2. "15 min post-news stabilization" requires knowing when news occurred
3. "Direction of initial shock" requires identifying the news-driven move vs. normal volatility

**Unblock Conditions**: Same as ED-01

---

## REVIEW_NEEDS_SPECIFICATION (Not Rejected, But Not Implementable As-Is)

### HY-01: GARCH Adaptive
- **Original Priority**: C (Gemini)
- **Audit Decision**: REVIEW_NEEDS_SPECIFICATION
- **Family**: Hybrid
- **Scoring**: 78/100

**Specification Gaps**:
1. GARCH(1,1) online fitting: What is the lookback window? How often does the model refit?
2. Regime transition: What threshold distinguishes "trend" from "range" regime?
3. Convergence failure: What happens when GARCH fails to converge (common with FX returns)?
4. HMM alternative: How many hidden states? What emission distribution?
5. Computational cost: GARCH fitting on every bar is expensive — batch frequency?

**Resolution Path**: Source 6 (Investigacion #6 HY_GARCH_Adaptive) has the most complete specification. Need explicit: regime_threshold, lookback_days, refit_frequency, fallback_when_no_convergence.

---

### TP-03: Fibonacci 61.8% Pullback
- **Original Priority**: C (Gemini)
- **Audit Decision**: REVIEW_NEEDS_SPECIFICATION
- **Family**: Trend Pullback
- **Scoring**: 75/100

**Specification Gaps**:
1. **No microstructural basis**: 61.8% is a geometric ratio with no causal mechanism in FX microstructure. Unlike VWAP (actual traded price average) or session H/L (institutional reference), Fibonacci levels are self-referential.
2. **Swing detection undefined**: "Impulso M5" — how is the start and end of the impulse determined? Zigzag? ATR threshold? Structural H/L?
3. **Which Fibonacci level**: Source uses 61.8% but other sources use 50% or 78.6% — arbitrary.

**Resolution Path**: If kept, must define:
- Algorithmic swing detection (e.g., fractal high/low with N-bar lookback)
- Minimum impulse size (ATR multiple)
- Why 61.8% specifically (backtest evidence required, not geometric assertion)

---

### HY-02: HVFTF Trend Following
- **Original Priority**: C (Gemini)
- **Audit Decision**: REVIEW_NEEDS_SPECIFICATION
- **Family**: Hybrid
- **Scoring**: 70/100

**Specification Gaps**:
1. **M5/M15 conflict**: What happens when SuperTrend(M5) is LONG but SuperTrend(M15) is SHORT? Source says "filtered by M15" but doesn't specify: does M15 have veto power? Or does alignment create the signal?
2. **ADX threshold**: Which timeframe ADX? M5 or M15? Threshold value not specified.
3. **Limited edge vs. simplicity**: Multi-timeframe SuperTrend alignment is a common retail approach with extensive public backtests showing limited edge after costs.

**Resolution Path**: Define conflict resolution as: "ENTER only when M5 and M15 SuperTrend agree. M15 has absolute veto. ADX(M15) > 20 required."

---

## Statistics

| Category | Count | IDs |
|----------|-------|-----|
| REJECTED | 1 | SD-05 |
| DEFERRED (news data) | 3 | ED-01, ED-02, ED-03 |
| NEEDS SPECIFICATION | 3 | HY-01, TP-03, HY-02 |
| **Total non-implementable** | **7** | |
| Implementable (A+B) | 18 | All others |

---
Generated: 2026-05-16 by Claude Opus 4.7
