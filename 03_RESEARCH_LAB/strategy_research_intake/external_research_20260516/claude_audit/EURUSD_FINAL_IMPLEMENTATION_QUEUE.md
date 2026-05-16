# EURUSD FINAL IMPLEMENTATION QUEUE

## 1. Wave 1: Priority A (Immediate Skeletons)
Estas estrategias han pasado el arbitraje final y se consideran seguras para la implementación de esqueletos de señal.

| ID | Name | Family | Complexity | Focus |
|----|------|--------|------------|-------|
| **MR-01** | Anchor Elastic | Mean Reversion | Medium | APM Reversion |
| **MR-02** | VWAP Stretch | Mean Reversion | Low | SD Bands Reversion |
| **VE-ORB** | Opening Range Break | Volatility | Low | Session Breakout |

## 2. Wave 2: Priority B (Pending Specifications)
Estrategias que requieren una definición matemática más rigurosa antes de proceder.

| ID | Name | Family | Reason for Hold |
|----|------|--------|-----------------|
| SD-01 | Europe Extreme Failure | Session | High Correlation with Manipulante |
| SD-02 | London Session H/L | Session | Needs Pre-Session Range Filter |
| SD-03 | Asian Range Fakeout | Session | Needs Liquidity Filter |

## 3. Under Review (Blocked)
| ID | Name | Family | Blocker |
|----|------|--------|---------|
| VE-01 | RV Shock Break | Volatility | Hallucinated Parameters (p30) |
| TP-01 | Trend Day Pullback | Trend | Subjetive "Trend Day" definition |

## 4. Deferred (News/Data)
| ID | Name | Family | Requirement |
|----|------|--------|-------------|
| ED-01 | Post-News Stabilization | Event | News Feed Certification |

## 5. Technical Requirements for Skeletons
1. **Module Location**: `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/strategies/`.
2. **Naming Convention**: `strat_mr_01_anchor_elastic.py`.
3. **Indicator Source**: Use `src/v6_utils` or local strategy utils.
4. **Data Contract**: Inputs must be OHLCV dataframes (Train 2015-2024 only).
