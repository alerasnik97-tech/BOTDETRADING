# Parsed Notes: EURUSD 07_00-19_00 NY Strategy Research Report GPT.pdf

## Source
- **File**: EURUSD 07_00-19_00 NY Strategy Research Report GPT.pdf
- **SHA256**: D6B8E5F6F949E94BB6209356FDFB5FB0407F0DC4C58C5E501C4D638BFC51EB4D
- **Read Status**: FULL_READ (38 pages, 77KB text)

## Executive Summary
Comprehensive quantitative research report proposing 20 fully-specified strategies for EURUSD intraday (07:00-19:00 NY). Written from a senior quant researcher perspective with academic citations (BIS, Duke, Georgetown, NBER, Fed). Emphasizes microstructural logic, cost sensitivity, anti-overfitting governance. Explicitly recommends 5 first-backtest candidates.

## Strategy Ideas Found (20 total)

| # | Name | Family | Window | News Dep | HP Dep |
|---|---|---|---|---|---|
| 1 | Anchor Elastic | Mean Reversion | 09:30-15:30 | NO | NO |
| 2 | RV Shock Break | Volatility Expansion | 07:30-11:00 | NO | rv5/rv15 from ticks |
| 3 | Trend Day EMA Pullback | Trend Pullback | 09:30-13:30 | NO | NO |
| 4 | Europe Extreme Failure | Session Failure | 07:00-11:00 | NO | NO |
| 5 | Post-News Stabilization | Event Driven | Event-driven | **YES** | Spread data |
| 6 | Break-Retest Hybrid | Hybrid | 08:00-15:00 | NO | rv5 |
| 7 | Anchor Pullback Continuation | Trend Pullback | 09:30-14:30 | NO | NO |
| 8 | Handoff Box Breakout | Session Breakout | 09:00-11:30 | NO | NO |
| 9 | Sigma Exhaustion Fade | Mean Reversion | 08:30-14:30 | NO | NO |
| 10 | Regime Shift Continuation | Volatility Expansion | 07:30-11:30 | NO | NO |
| 11 | Coil Release | Volatility Expansion | 09:00-12:00 | NO | tick count |
| 12 | Session Midpoint Snapback | Mean Reversion | 11:00-15:00 | NO | NO |
| 13 | Midday Re-Expansion | Volatility Expansion | 12:45-15:30 | NO | NO |
| 14 | London Lunch Fade | Statistical Edge | London lunch | NO | NO |
| 15 | Spread Shock Fade | Mean Reversion | 07:00-16:00 | NO | **YES** (spread p90 intradía) |
| 16 | Handoff Box Breakout | Session Breakout | 09:00-11:30 | NO | NO |
| 17 | Midday Box Breakout | Session Breakout | 13:15-15:30 | NO | NO |
| 18 | News Overreaction Fade | Event Driven | Event-driven | **YES** | Spread data |
| 19 | Session Drift Differential | Statistical Edge | Multi-window | NO | NO |
| 20 | Decay Switch | Hybrid | 10:00-15:00 | NO | NO |

## Top 5 Recommended (by source author)
1. Anchor Elastic
2. RV Shock Break
3. Trend Day EMA Pullback
4. Europe Extreme Failure
5. Post-News Stabilization Continuation

## Key Operational Components
- APM (Anchor Price Mean) from 07:00 NY — VWAP if available, TWAP proxy otherwise
- Realized Volatility (rv5, rv15) — computable from M1 closes
- Base Pack: tick bid/ask, mid, spread, M1/M3/M5/M15, ATR, session ranges
- Max 1 entry per strategy per day
- spread_ok filter, news_ok filter
- Blackout around rollover 16:55-17:10 NY

## News Dependency
- Post-News Stabilization: **REQUIRES** news calendar timestamps, spread pre/post measurement
- News Overreaction Fade: **REQUIRES** news calendar, event identification
- All other strategies: Use news_ok as FILTER only (exclude around events)

## High Precision Dependency
- Spread Shock Fade: Requires spread percentile intradía (feed-dependent)
- Coil Release: Requires tick count per 3-min block
- RV Shock Break: rv5/rv15 computable from M1 returns (LOW precision risk)

## Implementation Ambiguity
- APM definition depends on VWAP data availability
- "Valid breakout" in Break-Retest Hybrid not self-contained
- Decay Switch has two branches (continuation/reversion) — high complexity

## Contradictions
- None material between strategies

## Notes Against Gemini Extraction
- Gemini captured top 5 correctly from this source
- Gemini MISSED 8+ strategies from this source (Handoff Box, Sigma Exhaustion, Regime Shift, Coil Release, Midday Re-Expansion, Session Midpoint Snapback, Anchor Pullback Continuation, Spread Shock Fade)
- Some of the missed ones are Priority B quality candidates (Sigma Exhaustion Fade, Handoff Box Breakout)
- Gemini correctly identified Post-News Stabilization but INCORRECTLY classified it as Priority A when it requires news data

---
Parsed: 2026-05-16 by Claude Opus 4.7
