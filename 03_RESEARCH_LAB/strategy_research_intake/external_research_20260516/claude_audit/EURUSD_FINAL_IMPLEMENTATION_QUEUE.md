# EURUSD FINAL IMPLEMENTATION QUEUE

## 1. Wave 1: Priority A (Immediate Skeletons)
Estas estrategias han pasado el arbitraje final y se consideran seguras para la implementación de esqueletos de señal.

| ID | Name | Family | Complexity | Focus |
|----|------|--------|------------|-------|
| **MR-01** | Anchor Elastic | Mean Reversion | Medium | APM Reversion |
| **MR-02** | VWAP Stretch | Mean Reversion | Low | NY VWAP Reversion |
| **TP-01** | LDN-NY Momentum | Trend | Medium | EMA20 Pullback |
| **VE-ORB**| Opening Range Break | Volatility | Low | Session Expansion |

## 2. Wave 2: Priority B (Pending Specifications)
Estrategias que requieren una definición matemática más rigurosa antes de proceder.

| ID | Name | Family | Reason for Hold |
|----|------|--------|-----------------|
| TP-02 | Institutional EMA | Trend | Needs precise trigger spec |
| SD-02 | London Session H/L | Session | Needs Pre-Session Range Filter |
| SD-03 | Asian Range Fakeout | Session | Needs Liquidity Filter |

## 3. Under Review / Rejected
| ID | Name | Family | Status | Reason |
|----|------|--------|--------|---------|
| SD-01 | Europe Extreme Fail | Session | **REJECTED** | High Correlation with Manipulante |
| VE-01 | RV Shock Break | Volatility | **REVIEW** | Hallucinated Parameters (rv5/rv15) |
| TP-03 | Fibonacci Pullback | Trend | **REVIEW** | Needs ADX/Volume confirmation |

## 4. Deferred (News/Data)
| ID | Name | Family | Requirement |
|----|------|--------|-------------|
| ED-01 | Post-News Stabilization | Event | News Feed Certification |

## 5. Technical Requirements for Skeletons
1. **Module Location**: `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/strategies/`.
2. **Naming Convention**: `strat_mr_01_anchor_elastic.py`.
3. **Indicator Source**: Use `src/v6_utils` or local strategy utils.
4. **Data Contract**: Inputs must be OHLCV dataframes (Train 2015-2024 only).
