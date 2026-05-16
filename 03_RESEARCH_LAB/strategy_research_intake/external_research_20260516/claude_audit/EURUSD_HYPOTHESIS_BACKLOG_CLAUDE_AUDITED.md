# EURUSD Institutional Hypothesis Backlog — CLAUDE AUDITED (2026-05-16)

**Auditor**: Claude Opus 4.7
**Date**: 2026-05-16
**Basis**: Independent re-read of 4/6 sources (2 image-based PDFs unreadable)
**Original Author**: Gemini 3 Flash (25 hypotheses)
**Audit Result**: 25 hypotheses retained, 1 REJECTED, 3 DEFERRED, 3 REVIEW_NEEDED, 4 PRIORITY_A confirmed

---

## Classification Legend

| Status | Meaning |
|--------|---------|
| **PRIORITY_A** | Implementable NOW with certified data only (OHLCV + spread + ATR) |
| **PRIORITY_B** | Valid hypothesis, secondary implementation wave |
| **PRIORITY_C** | Valid but lower scoring or higher complexity |
| **DEFER_NEWS** | Requires uncertified news calendar data — blocked until data certified |
| **REVIEW_NEEDS_SPECIFICATION** | Ambiguous or underspecified — needs clarification before implementation |
| **REJECT_DISCRETIONARY** | Too discretionary, undefined edge, or excessive correlation with existing strategies |

---

## PRIORITY A — IMPLEMENT SMOKE TEST (4 strategies)

### 1. Anchor Elastic (MR-01)
- **Familia**: Mean Reversion
- **Gatillo**: Desviacion extrema (>1.2 ATR o 1.8 sigma) respecto al ancla de precio medio (APM/VWAP anclado 07:00 NY) sin regimen de tendencia fuerte (ADX < 22).
- **Entrada**: Cierre de vela M1 de reingreso a la banda despues del extremo.
- **Salida**: TP en APM (max 1.5R), SL en extremo de excursion + buffer.
- **Scoring**: 95/100
- **Audit Status**: CONFIRMED PRIORITY_A
- **Source Support**: Source 1 (GPT Report #1 Anchor Elastic), Source 3 (MD Report #5 VWAP Reversion), Source 6 (Investigacion #1 MR_VWAP_Stretch)
- **Data Required**: OHLCV M1, ATR(14), ADX(14), VWAP anchored 07:00. ALL CERTIFIED.
- **News Dependency**: NO (uses news_ok as exclusion filter only)
- **Correlation Risk**: LOW (unique APM anchor logic)

### 2. RV Shock Break (VE-01)
- **Familia**: Volatility Expansion
- **Gatillo**: Ruptura de canal Donchian de 30m tras compresion de Realized Volatility (rv15 <= p30).
- **Entrada**: Cierre M5 por encima de canal + shock de rv5 (>= 2x mediana).
- **Salida**: TP 2.0R, SL debajo de barra de ruptura.
- **Scoring**: 92/100
- **Audit Status**: CONFIRMED PRIORITY_A
- **Source Support**: Source 1 (GPT Report #2 RV Shock Break), Source 2 (NY Report #1 London Compression Breakout — similar mechanism), Source 6 (Investigacion #3 VE_NR7_NY_Break)
- **Data Required**: OHLCV M1/M5, Donchian(30m), rv5/rv15 computable from M1 returns. ALL CERTIFIED.
- **News Dependency**: NO
- **Correlation Risk**: LOW (RV-based trigger is distinct)
- **Note**: rv5/rv15 are computable from M1 close returns — no external feed required.

### 3. Trend Day EMA Pullback (TP-01)
- **Familia**: Trend Pullback
- **Gatillo**: Calificacion de Trend Day (07:00-09:30). Primer retroceso que toca EMA20 sin cerrar debajo de EMA50.
- **Entrada**: Cierre M1 confirmando giro a favor de tendencia.
- **Salida**: TP 2.0R o trailing bajo EMA20.
- **Scoring**: 91/100
- **Audit Status**: CONFIRMED PRIORITY_A
- **Source Support**: Source 1 (GPT Report #3 Trend Day EMA Pullback), Source 3 (MD Report #12 Institutional EMA Pullback), Source 6 (Investigacion #14 TP_EMA_Confluence)
- **Data Required**: OHLCV M1/M5, EMA20, EMA50, ADX. ALL CERTIFIED.
- **News Dependency**: NO
- **Correlation Risk**: LOW (trend-following, orthogonal to existing sweep strategy)

### 4. Europe Extreme Failure (SD-01)
- **Familia**: Session Dynamics
- **Gatillo**: Falsa extension del extremo europeo (02:00-07:00 NY). Excursion < 0.08 ATR seguida de reingreso rapido.
- **Entrada**: Reingreso en maximo 3 velas M1.
- **Salida**: TP en midpoint europeo.
- **Scoring**: 89/100
- **Audit Status**: CONFIRMED PRIORITY_A
- **Source Support**: Source 1 (GPT Report #4 Europe Extreme Failure), Source 2 (NY Report #6 Asia-to-NY Range Failure — related), Source 6 (Investigacion #19 SB_Asia_Sweep_NY)
- **Data Required**: OHLCV M1, session high/low (02:00-07:00 NY), ATR. ALL CERTIFIED.
- **News Dependency**: NO
- **Correlation Risk**: MODERATE (shares session-boundary logic with Manipulante but distinct trigger mechanism — false extension vs. sweep)
- **Mitigation**: Must compute correlation with existing strategy during walk-forward.

---

## DEFER_NEWS — Blocked Until News Data Certified (3 strategies)

### 5. Post-News Stabilization (ED-01)
- **Familia**: Event Driven
- **Gatillo**: Continuacion posnoticia tras normalizacion de spread (10-15 min despues del release).
- **Entrada**: Ruptura de maximo de consolidacion posnoticia en M1.
- **Salida**: TP 1.8R, SL lado opuesto de consolidacion.
- **Scoring**: 88/100
- **Original Priority**: A (Gemini) → **DOWNGRADED to DEFER_NEWS**
- **Reason**: Requires forex_factory_cache.csv (MISSING) and/or news_eurusd_v2_utc.csv (MISSING). Source 1 explicitly states: "Required Data: Base Pack + timestamp de noticia de alto impacto + spread pre/post news". Cannot implement without certified news calendar.
- **Source Support**: Source 1 (GPT Report #5), Source 2 (NY Report #11 Post-NFP Stabilization), Source 3 (MD Report #15)
- **Unblock Condition**: Provide certified news calendar CSV with columns [datetime_utc, currency, impact, event_name] covering train period.

### 20. Post-News Volatility Reversion (ED-02)
- **Familia**: Event Driven
- **Gatillo**: Reversion de spike inicial (ATR > 2x pre-news) tras 10 min del release.
- **Entrada**: Vela M1 de giro que rompe el maximo/minimo de la vela de shock.
- **Salida**: TP 0.75x ATR_pre, SL 1.0x ATR_pre.
- **Scoring**: 73/100
- **Audit Status**: DEFER_NEWS
- **Reason**: Same news calendar dependency as ED-01.
- **Source Support**: Source 1 (GPT Report #18 News Overreaction Fade), Source 2 (NY Report #12 FOMC Drift-Reentry)

### 21. PNMC-15 Momentum (ED-03)
- **Familia**: Event Driven
- **Gatillo**: Estabilizacion de 15 min posnoticia (StdDev < 0.5x ATR).
- **Entrada**: Ruptura del rango de estabilizacion en direccion del shock inicial.
- **Salida**: TP 2x ATR(15m), SL 1x ATR(15m).
- **Scoring**: 72/100
- **Audit Status**: DEFER_NEWS
- **Reason**: Same news calendar dependency as ED-01.
- **Source Support**: Source 2 (NY Report #11), Source 3 (MD Report #16 News Momentum)

---

## PRIORITY B — Secondary Implementation Wave (14 strategies)

### 6. London Session H/L Breakout (SD-02)
- **Familia**: Session Dynamics
- **Scoring**: 87/100
- **Audit Status**: PRIORITY_B
- **Source Support**: Source 2 (NY Report #1 London Compression), Source 3 (MD Report #9 London Session H/L), Source 6 (Investigacion #5 SB_IB_Fakeout)

### 7. VWAP Stretch Reversion (MR-02)
- **Familia**: Mean Reversion
- **Scoring**: 86/100
- **Audit Status**: PRIORITY_B
- **Source Support**: Source 3 (MD Report #5 VWAP Reversion), Source 6 (Investigacion #1 MR_VWAP_Stretch)
- **Note**: Overlaps with MR-01 Anchor Elastic. Must verify non-redundancy during implementation.

### 8. Institutional EMA Pullback (TP-02)
- **Familia**: Trend Pullback
- **Scoring**: 85/100
- **Audit Status**: PRIORITY_B
- **Source Support**: Source 3 (MD Report #12), Source 6 (Investigacion #14 TP_EMA_Confluence)
- **Note**: Overlaps with TP-01. Differentiation via timeframe (M15 vs M1 entry).

### 9. BB Squeeze Momentum (VE-02)
- **Familia**: Volatility Expansion
- **Scoring**: 84/100
- **Audit Status**: PRIORITY_B
- **Source Support**: Source 3 (MD Report #2 BB Squeeze + ADX), Source 6 (Investigacion #8 VE_BBSqueeze_Mom)

### 10. Asian Range Fakeout (SD-03)
- **Familia**: Session Dynamics
- **Scoring**: 83/100
- **Audit Status**: PRIORITY_B
- **Source Support**: Source 3 (MD Report #10 Asian Range Fakeout), Source 2 (NY Report #6 Asia-to-NY Range Failure)
- **Correlation Risk**: HIGH with existing Manipulante strategy (Source 2 explicitly flags this). Must compute correlation before promoting.

### 11. Initial Balance Failure (SD-04)
- **Familia**: Session Dynamics
- **Scoring**: 82/100
- **Audit Status**: PRIORITY_B
- **Source Support**: Source 6 (Investigacion #5 SB_IB_Fakeout), Source 2 (NY Report #7 Early NY False Breakout Fade)

### 12. London Close Mean Reversion (MR-03)
- **Familia**: Mean Reversion
- **Scoring**: 81/100
- **Audit Status**: PRIORITY_B
- **Source Support**: Source 1 (GPT Report #14 London Lunch Fade), Source 3 (MD Report #17 London Close Reversion), Source 6 (Investigacion #11 SB_LondonClose_Trap)

### 13. Friday Reversion (SE-01)
- **Familia**: Statistical Edge
- **Scoring**: 80/100
- **Audit Status**: PRIORITY_B
- **Source Support**: Source 6 (Investigacion #10 ST_Friday_Rev), Source 2 (NY Report #14 Friday Afternoon Flattening)

### 14. London Lunch Fade (SE-02)
- **Familia**: Statistical Edge
- **Scoring**: 79/100
- **Audit Status**: PRIORITY_B
- **Source Support**: Source 1 (GPT Report #14 London Lunch Fade), Source 2 (NY Report #18 Midday Quiet Range Reversion)

### 16. Keltner Snapback (MR-04)
- **Familia**: Mean Reversion
- **Scoring**: 77/100
- **Audit Status**: PRIORITY_B
- **Source Support**: Source 3 (MD Report #3 Keltner Breakout — inverse logic), Source 6 (Investigacion #12 MR_Keltner_Snapbk)

### 17. ATR Compression-Expansion (VE-03)
- **Familia**: Volatility Expansion
- **Scoring**: 76/100
- **Audit Status**: PRIORITY_B
- **Source Support**: Source 6 (Investigacion #13 VE_ATR_Spike_Cont), Source 2 (NY Report #15 Volatility Regime Breakout Filter)

### 19. Breakout-Retest Structural (TP-04)
- **Familia**: Trend Pullback
- **Scoring**: 74/100
- **Audit Status**: PRIORITY_B
- **Source Support**: Source 3 (MD Report #14 Breakout-Retest), Source 2 (NY Report #9 Post-Impulse Pullback Continuation)

### 22. NY Mid-Day Volatility Expansion (VE-04)
- **Familia**: Volatility Expansion
- **Scoring**: 71/100
- **Audit Status**: PRIORITY_B
- **Source Support**: Source 2 (NY Report #18 Midday Quiet Range Reversion — inverse), Source 3 (MD Report #18 NY Mid-Day Breakout)

### 24. M15 Trend + VWAP MR (HY-03)
- **Familia**: Hybrid
- **Scoring**: 69/100
- **Audit Status**: PRIORITY_B
- **Source Support**: Source 3 (MD Report #20 M15 VWAP Reversion), Source 6 (Investigacion #9 TP_VWAP_Pullback)

---

## REVIEW_NEEDS_SPECIFICATION (3 strategies)

### 15. GARCH Adaptive (HY-01)
- **Familia**: Hybrid
- **Scoring**: 78/100
- **Audit Status**: REVIEW_NEEDS_SPECIFICATION
- **Reason**: GARCH(1,1)/HMM regime detection is computationally expensive and not well-specified for live execution. Requires explicit definition of: regime transition thresholds, lookback window, online vs batch fitting, fallback when model fails to converge.
- **Source Support**: Source 6 (Investigacion #6 HY_GARCH_Adaptive) — best specification available

### 18. Fibonacci 61.8% Pullback (TP-03)
- **Familia**: Trend Pullback
- **Scoring**: 75/100
- **Audit Status**: REVIEW_NEEDS_SPECIFICATION
- **Reason**: Fibonacci levels are geometrically arbitrary. Source 3 (MD Report) provides implementation but no microstructural justification. No academic basis for 61.8% as a support level in FX microstructure. If kept, must define swing detection algorithm precisely (currently ambiguous "impulso M5").
- **Source Support**: Source 3 (MD Report #13 Fibonacci Retracement) — weak justification

### 23. HVFTF Trend Following (HY-02)
- **Familia**: Hybrid
- **Scoring**: 70/100
- **Audit Status**: REVIEW_NEEDS_SPECIFICATION
- **Reason**: SuperTrend multi-timeframe alignment is underspecified for conflict resolution (what happens when M5 and M15 SuperTrend disagree?). Implementation complexity HIGH with limited unique edge vs simpler trend-following.
- **Source Support**: Source 3 (MD Report #19 ATR-SuperTrend Hybrid)

---

## REJECT_DISCRETIONARY (1 strategy)

### 25. Programmable Structure Break + Fill (SD-05)
- **Familia**: Session Dynamics
- **Scoring**: 68/100
- **Audit Status**: REJECT_DISCRETIONARY
- **Reason**: "Structural imbalance" and "value zone mitigation" are ICT/SMC concepts without rigorous quantitative definition. Entry is discretionary ("Cierre M5 tras mitigacion de zona de valor" — how is "zone of value" identified algorithmically?). Source 2 explicitly warns HIGH correlation with existing Manipulante strategy for sweep-based entries.
- **Source Support**: Weak — no source provides a fully-specified algorithmic version

---

## Audit Statistics

| Category | Count | IDs |
|----------|-------|-----|
| PRIORITY_A (confirmed) | 4 | MR-01, VE-01, TP-01, SD-01 |
| DEFER_NEWS | 3 | ED-01, ED-02, ED-03 |
| PRIORITY_B | 14 | SD-02, MR-02, TP-02, VE-02, SD-03, SD-04, MR-03, SE-01, SE-02, MR-04, VE-03, TP-04, VE-04, HY-03 |
| REVIEW_NEEDS_SPECIFICATION | 3 | HY-01, TP-03, HY-02 |
| REJECT_DISCRETIONARY | 1 | SD-05 |
| **TOTAL** | **25** | |

---

## Numbering Note
Gemini's original backlog skips #10 (jumps from #9 to #11). This audit preserves original Gemini numbering for traceability. SD-03 (Asian Range Fakeout) has internal sequence 10 but external label #10 was absent in Gemini's MD output.

---
Audited: 2026-05-16 by Claude Opus 4.7
Branch: governance/claude-strategy-intake-audit-20260516
