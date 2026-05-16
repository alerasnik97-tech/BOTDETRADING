# Parsed Notes: EURUSD 07_00-19_00 NY Strategy Research Report.pdf

## Source
- **File**: EURUSD 07_00-19_00 NY Strategy Research Report.pdf
- **SHA256**: 0A8078BC1CFB17B34D39A4FAAFEEFD9D3721010085A8FE420E0996A6DBD01147
- **Read Status**: FULL_READ (17 pages, 38KB text)

## Executive Summary
20 strategies presented in English with full specifications including pseudocode. Focus on 5 families: volatility expansion, mean reversion, session breakout/failure, trend pullback, time-of-day/statistical. More conservative than GPT report — explicitly notes high correlation risk with existing "liquidity sweep" strategy for several candidates.

## Strategy Ideas Found (20 total)

| # | Name | Family | Window | News Dep | HP Dep |
|---|---|---|---|---|---|
| 1 | London Compression Breakout | Vol Expansion | 07:00-10:30 | NO | NO |
| 2 | NY Open Range Expansion | Vol Expansion | 08:30-11:30 | NO | NO |
| 3 | Pre-Data Compression Release | Vol Expansion | 08:00-10:00 | **YES** | YES |
| 4 | Session VWAP Dislocation | Mean Reversion | 07:00-15:00 | NO | NO |
| 5 | ATR Stretch Snapback | Mean Reversion | 07:00-19:00 | NO | NO |
| 6 | Asia-to-NY Range Failure | Session Failure | 07:00-09:30 | NO | NO |
| 7 | Early NY False Breakout Fade | Session Failure | 07:00-10:00 | NO | NO |
| 8 | Late-Session Trend Pullback | Trend Pullback | 12:00-18:30 | NO | NO |
| 9 | Post-Impulse Pullback Continuation | Trend Pullback | 08:30-16:00 | NO | NO |
| 10 | Pre-London Trend Alignment | Trend Pullback | 07:00-08:30 | NO | NO |
| 11 | Post-NFP Stabilization Breakout | Post-News | 08:45-10:00 | **YES** | YES |
| 12 | FOMC Drift-Reentry | Post-News | 14:05-16:30 | **YES** | YES |
| 13 | Monday NY Mean Reversion | Time-of-Day | 07:00-12:00 Mon | NO | NO |
| 14 | Friday Afternoon Flattening | Time-of-Day | 13:00-18:30 Fri | NO | NO |
| 15 | Volatility Regime Breakout Filter | Hybrid | 07:00-19:00 | NO | NO |
| 16 | Trend + Compression Pullback Hybrid | Hybrid | 07:00-19:00 | NO | NO |
| 17 | Opening Sweep Failure | Mean Reversion | 07:00-09:00 | NO | NO |
| 18 | Midday Quiet Range Reversion | Mean Reversion | 11:00-14:00 | NO | NO |
| 19 | Session High/Low Rejection | Session Failure | 07:00-19:00 | NO | NO |
| 20 | Time-Boxed Trend Exhaustion | Trend Pullback | 15:00-18:50 | NO | NO |

## Top 5 Recommended (by source)
1. London Compression Breakout
2. NY Open Range Expansion
3. Session VWAP Dislocation
4. Trend + Compression Pullback Hybrid
5. Volatility Regime Breakout Filter

## Correlation Risk Flags
- **HIGH correlation with Manipulante**: Asia-to-NY Range Failure, Early NY False Breakout Fade, Opening Sweep Failure, Session High/Low Rejection
- Source explicitly warns: "Reject if correlated with existing strategy outcomes"

## News Dependency
- Post-NFP Stabilization: NFP-only, requires calendar
- FOMC Drift-Reentry: FOMC-only, requires calendar
- Pre-Data Compression Release: Requires scheduled macro release times

## Notes Against Gemini Extraction
- Gemini did NOT directly adopt this source's top 5 (London Compression, NY Open Range, etc.)
- Instead Gemini favored GPT report's top 5
- Many high-quality non-news strategies from this source were omitted (Late-Session Trend Pullback, Trend+Compression Hybrid, Monday MR, ATR Stretch Snapback)
- Source explicitly identifies Opening Sweep Failure as HIGH correlation with existing sweep strategy — Gemini's SD-03 (Asian Range Fakeout) has similar risk

---
Parsed: 2026-05-16 by Claude Opus 4.7
