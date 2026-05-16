# Parsed Notes: Investigación Estrategias Algorítmicas EURUSD.pdf

## Source
- **File**: Investigación Estrategias Algorítmicas EURUSD.pdf
- **SHA256**: 55F57C4FA71A3B92C8BBB6259E24CADD5DE41A88456265DAE8EA1CF586098F81
- **Read Status**: FULL_READ (22 pages, 52KB text)

## Executive Summary
Highly institutional document in Spanish. 20 strategies with academic rigor, explicit prop firm constraints (FTMO-style 4-5% daily DD, 8-10% max DD). Uses advanced concepts: GARCH(1,1), HMM regime detection, Volume Profile (HVN/LVN), VWAP bands. Explicit DSR/PBO validation requirements. Max 3 trades/day per strategy.

## Strategy Ideas Found (20 total)

| # | Name | Family | Window | News Dep | HP Dep |
|---|---|---|---|---|---|
| 1 | MR_VWAP_Stretch | Mean Reversion | 09:00-16:00 | Filter | NO |
| 2 | TP_HVN_Retest | Trend Pullback | 08:00-17:00 | NO | **YES** (Volume Profile) |
| 3 | VE_NR7_NY_Break | Vol Expansion | 08:00-11:00 | NO | NO |
| 4 | MR_LondonFix_Fade | Mean Reversion | 10:30-11:30 | NO | NO (calendar anomaly) |
| 5 | SB_IB_Fakeout | Session Failure | 09:00-11:00 | NO | NO |
| 6 | HY_GARCH_Adaptive | Hybrid | 07:00-19:00 | NO | NO (complex model) |
| 7 | PN_1H_Delay_Sent | Post-News | 09:30-15:00 | **YES** | NO |
| 8 | VE_BBSqueeze_Mom | Vol Expansion | 08:00-12:00 | Filter | NO |
| 9 | TP_VWAP_Pullback | Trend Pullback | 08:00-15:00 | NO | NO |
| 10 | ST_Friday_Rev | Time-of-Day | 12:00-16:00 Fri | NO | NO |
| 11 | SB_LondonClose_Trap | Session Failure | 11:00-13:00 | NO | NO |
| 12 | MR_Keltner_Snapbk | Mean Reversion | 12:00-15:00 | NO | NO |
| 13 | VE_ATR_Spike_Cont | Vol Expansion | 08:00-16:00 | NO | NO |
| 14 | TP_EMA_Confluence | Trend Pullback | 07:00-19:00 | NO | NO |
| 15 | PN_VolContract_Brk | Post-News | 09:00-14:00 | **YES** | NO |
| 16 | MR_SessionExtremes | Mean Reversion | 14:00-18:00 | NO | NO |
| 17 | VE_OpenRange_Exp | Vol Expansion | 08:30-10:30 | NO | Volume Profile |
| 18 | ST_NY_Lunch_Rng | Time-of-Day | 12:00-13:30 | NO | NO |
| 19 | SB_Asia_Sweep_NY | Session Failure | 07:00-09:30 | NO | NO |
| 20 | HY_VolTrend_Sync | Hybrid | 08:00-16:00 | NO | VIX/implied vol |

## Key Operational Components
- Strict prop firm risk model (4-5% daily DD, 8-10% max DD)
- Max 3 trades/day per strategy
- DSR (Deflated Sharpe Ratio) required for validation
- Walk-forward obligatorio
- ECN/STP execution assumed (spread 0.1-0.3 pips, $6/lot RT, 0.5-1.0 pip slippage)

## News Dependency
- PN_1H_Delay_Sent: Requires high-impact event calendar (NFP, CPI, ECB)
- PN_VolContract_Brk: Requires FOMC/CPI calendar

## High Precision Dependency
- TP_HVN_Retest: Requires Fixed Range Volume Profile (institutional tick data)
- HY_VolTrend_Sync: Requires VIX or implied volatility (external data)
- VE_OpenRange_Exp: Requires volume profile M1

## Notes Against Gemini Extraction
- Gemini captured HY-01 (GARCH Adaptive), SD-04 (IB Fakeout), SE-01 (Friday Reversion), VE-02 (BB Squeeze)
- Gemini MISSED: London Fix Fade (documented calendar anomaly, BIS-backed), TP_HVN_Retest (requires Volume Profile)
- London Fix Fade is interesting but very specialized (end-of-month only)
- Source has the most rigorous validation framework of all documents

---
Parsed: 2026-05-16 by Claude Opus 4.7
