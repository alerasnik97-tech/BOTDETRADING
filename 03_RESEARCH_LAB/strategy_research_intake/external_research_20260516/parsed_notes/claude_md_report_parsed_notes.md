# Parsed Notes: EURUSD_Strategy_Research_Report.md

## Source
- **File**: EURUSD_Strategy_Research_Report.md
- **SHA256**: 7ADEA5D4CED5DC31B27437DAA2F8A8A7EE7D3AD64E51ABC2EA002461E0BC014C
- **Read Status**: FULL_READ (175KB markdown)

## Executive Summary
20 strategies with full Python pseudocode. More "textbook" approach — uses standard indicators (BB, Keltner, RSI, ADX, VWAP) without the microstructural depth of the GPT report. Provides detailed Python implementations. Strategies are more generic/retail-oriented but well-documented. Scored differently from other sources (Priority Score 71-96).

## Strategy Ideas Found (20 total)

| # | Name | Family | Window | News Dep | HP Dep |
|---|---|---|---|---|---|
| 1 | VE-ORB (Volatility Expansion ORB ATR) | Vol Expansion | 07:00-14:00 | Filter | NO |
| 2 | BB Squeeze + ADX | Vol Expansion | 09:30-16:00 | Filter | NO |
| 3 | Keltner Breakout (VEKB) | Vol Expansion | 07:00-11:00 | Filter | NO |
| 4 | Donchian Breakout | Vol Expansion | 09:00-15:00 | NO | NO |
| 5 | VWAP Reversion | Mean Reversion | 10:00-16:00 | Filter | NO |
| 6 | RSI(2) Reversion | Mean Reversion | 07:00-19:00 | NO | NO |
| 7 | Statistical Reversion | Mean Reversion | 07:00-19:00 | NO | NO |
| 8 | BB Double Tap | Mean Reversion | 08:00-17:00 | NO | NO |
| 9 | London Session H/L | Session Breakout | 07:00-09:00 | NO | NO |
| 10 | Asian Range Fakeout | Session Breakout | 07:00-08:30 | NO | NO |
| 11 | NY Opening Reversal | Session Breakout | 08:00-10:00 | NO | NO |
| 12 | Institutional EMA Pullback | Trend Pullback | 07:00-19:00 | NO | NO |
| 13 | Fibonacci Retracement | Trend Pullback | 07:00-19:00 | NO | NO |
| 14 | Breakout-Retest | Trend Pullback | 07:00-19:00 | NO | NO |
| 15 | Post-News Stabilization | Post-News | Variable | **YES** | YES |
| 16 | News Momentum | Post-News | Variable | **YES** | YES |
| 17 | London Close Reversion | Time-of-Day | 11:00-12:30 | NO | NO |
| 18 | NY Mid-Day Breakout | Time-of-Day | 12:00-14:00 | NO | NO |
| 19 | ATR-SuperTrend Hybrid | Hybrid | 07:00-19:00 | NO | NO |
| 20 | M15 VWAP Reversion | Hybrid | 07:00-19:00 | NO | NO |

## Key Characteristics
- Higher scoring system (95/100 top vs 89/100 in GPT report's equivalent)
- More retail-oriented indicator use (RSI, BB, Keltner standard periods)
- Less microstructural justification
- Provides full Python code (unusual for research reports)

## News Dependency
- Post-News Stabilization (#15): Requires event calendar
- News Momentum (#16): Requires event calendar

## High Precision Dependency
- None explicitly required beyond OHLCV + spread

## Implementation Ambiguity
- RSI(2) Reversion: Too generic, no session/regime context
- Statistical Reversion: Underspecified
- Fibonacci levels: Arbitrary (no microstructural basis)

## Notes Against Gemini Extraction
- Gemini correctly mapped most strategies to its taxonomy
- This source supports MR-02 (VWAP Stretch), TP-02 (Inst EMA Pullback), VE-02 (BB Squeeze), SD-02 (London Session H/L), SD-03 (Asian Range Fakeout)
- Fibonacci strategy (TP-03) is weakly justified — arbitrary geometric levels without market structure basis

---
Parsed: 2026-05-16 by Claude Opus 4.7
