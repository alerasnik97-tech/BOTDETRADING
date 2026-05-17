# EURUSD STRATEGY FAMILY MATRIX V1

This document provides a systematic classification of trading concepts tailored specifically for intraday EURUSD. To maintain high portfolio diversity and minimize systemic risk, we categorize strategies into diverse statistical families.

---

## 1. Matrix Overview

| Family ID | Name | Core Trading Class | Entry Focus | suggested Session | Expected Correlation with Manipulante |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LBC** | London Breakout Continuation | Momentum Breakout | High-vol continuation | London | Low |
| **LBF** | London Breakout Failure | Mean Reversion | False breakout fade | London | Low |
| **NOI** | NY Open Impulse Continuation | Momentum / Volatility | Volatility breakout | NY Open | Medium |
| **NER** | NY Open Exhaustion Reversal | Mean Reversion | Session high/low fade | NY Open | Low |
| **VCP** | Volatility Compression Breakout| Volatility Expansion | ATR squeeze expansion | Any | Low |
| **NMR** | Post-News Spike Mean Reversion | Mean Reversion | Volatility spike fade | NY After-News | Low |
| **VMR** | VWAP / Session Mean Reversion | Mean Reversion | Deviation from VWAP | Midday NY | Low |
| **DAF** | Prior Day High/Low Failed Auction| Liquidity Sweep | Daily high/low sweep | London / NY | Low |
| **ARE** | Asian Range Expansion | Volatility Breakout | Asian high/low break | London | Low |
| **ARF** | Asian Range False Breakout | Mean Reversion | Asian sweep fade | London | Low |
| **MDB** | Midday Volatility Compression | Volatility Expansion | Midday compression | Midday NY | Low |
| **LCR** | London Close Rebalancing | Mean Reversion | London close flows | London Close | Low |
| **ADR** | ADR Exhaustion Reversal | Mean Reversion | Average Daily Range fade| Late NY | Low |
| **TVS** | Time-of-day Volatility Regime | Volatility Expansion | Volume session shifts | Session Open | Low |
| **MPC** | Micro Pullback Continuation | Trend following | Retracement inside trend| London / NY | Medium |
| **TDP** | Trend Day Participation | Trend following | Strong expansion buy/sell| London / NY | High |
| **RDF** | Range Day Fade | Mean Reversion | Mean reverting channels | Asian / Midday | Low |
| **WLF** | Weekly Level Failed Auction | Liquidity Sweep | Weekly high/low sweep | Any | Low |
| **PSC** | Previous Session Continuation | Trend following | Post-Asian bias | London | Low |
| **LAN** | Liquidity Alternative No-Manip | Liquidity Sweep | Multi-session sweep | London / NY | Zero (Targeted) |

---

## 2. Family Specifications & Sketches

### 1. London Breakout Continuation (`LBC`)
-   **Market Hypothesis:** The London opening brings institutional volume that establishes a sustainable intraday trend by breaking the Asian range.
-   **Why Edge Might Exist:** Large liquidity injections at the session open create directional momentum that carries through to NY.
-   **Why Edge Might Fail:** High incidence of stop hunts and pre-session fakeouts.
-   **Suggested NY Session:** London / Early NY.
-   **Expected Frequency:** $8-12$ trades per month.
-   **Expected Correlation with Manipulante:** Low.
-   **Implementation Difficulty:** Low.
-   **Overfitting Risk:** Low.
-   **Cost Risk:** Low.
-   **News Dependency Risk:** Low.
-   **Objective Rule Sketch:** Enter long if price breaks above the Asian Range high (00:00 - 07:00 GMT) by $> 0.5$ ATR with an EMA(20) filter pointing upward.
-   **Initial Risk Management:** Stop loss at Asian Range mid-point; 2R fixed target.
-   **Critical Metrics:** Win rate at breakout, average R per trade.
-   **No-Go Conditions:** Tight consolidated Asian range ($< 5$ pips total).
-   **First Batch Candidate:** **YES** (Classic systematic benchmark).

### 2. London Breakout Failure (`LBF`)
-   **Market Hypothesis:** The initial breakout of the Asian range is a stop-run designed to trap breakout traders before reversing.
-   **Why Edge Might Exist:** Institutional liquidity sweeps occur at key levels, matching sell orders with retail stop losses.
-   **Why Edge Might Fail:** On strong trend days, breakouts will continue without pulling back, leading to catastrophic losses.
-   **Suggested NY Session:** London.
-   **Expected Frequency:** $10-15$ trades per month.
-   **Expected Correlation with Manipulante:** Low.
-   **Implementation Difficulty:** Medium.
-   **Overfitting Risk:** Medium.
-   **Cost Risk:** Medium.
-   **News Dependency Risk:** Low.
-   **Objective Rule Sketch:** Enter short if price breaks the Asian Range high by $< 0.5$ ATR and quickly closes back inside the range on a bearish engulfing bar.
-   **Initial Risk Management:** Stop loss at the swing high of the fakeout; 1.5R target.
-   **Critical Metrics:** Profit Factor under high spread conditions.
-   **No-Go Conditions:** Strong pre-session economic releases.
-   **First Batch Candidate:** **YES** (Excellent counter-trend candidate).

### 3. NY Open Impulse Continuation (`NOI`)
-   **Market Hypothesis:** The intersection of London and NY sessions (13:00 - 15:00 GMT) generates massive volume that breaks key session ranges.
-   **Why Edge Might Exist:** US institutional desks enter the market, triggering heavy momentum expansion.
-   **Why Edge Might Fail:** Massive slippage and bid-ask widening during the NY open.
-   **Suggested NY Session:** Early NY.
-   **Expected Frequency:** $6-10$ trades per month.
-   **Expected Correlation with Manipulante:** Medium.
-   **Implementation Difficulty:** Low.
-   **Overfitting Risk:** Low.
-   **Cost Risk:** High.
-   **News Dependency Risk:** High.
-   **Objective Rule Sketch:** Enter long if the 13:30 bar breaks above the NY pre-market high with an ATR expansion $> 1.5$ times the 10-bar average.
-   **Initial Risk Management:** Stop loss at the low of the trigger bar; 2R target.
-   **Critical Metrics:** Slippage sensitivity under high volatility.
-   **No-Go Conditions:** Heavy high-impact US news scheduled within 30 minutes.
-   **First Batch Candidate:** **NO** (Excessive news vulnerability).

### 4. NY Open Exhaustion Reversal (`NER`)
-   **Market Hypothesis:** The initial NY impulse is often an overextended sweep that quickly exhausts once it meets opposite session liquidity.
-   **Why Edge Might Exist:** Reversion occurs as early US morning buyers/sellers take rapid profits before the lunchtime lull.
-   **Why Edge Might Fail:** High trend persistence can blow past exhaustion indicators.
-   **Suggested NY Session:** Mid-day NY (15:00 - 17:00 GMT).
-   **Expected Frequency:** $8-12$ trades per month.
-   **Expected Correlation with Manipulante:** Low.
-   **Implementation Difficulty:** Medium.
-   **Overfitting Risk:** High.
-   **Cost Risk:** Low.
-   **News Dependency Risk:** Medium.
-   **Objective Rule Sketch:** Enter short if price is $> 2.0$ ATR away from its 20-bar VWAP at 15:30 GMT and exhibits a bearish rejection candle.
-   **Initial Risk Management:** Stop loss 1 ATR above the exhaustion high; 1.5R target.
-   **Critical Metrics:** VWAP deviation stability.
-   **No-Go Conditions:** Clear trend day (EMA(20) > EMA(200) on H1).
-   **First Batch Candidate:** **YES** (Ideal lunchtime mean reversion).

### 5. Volatility Compression Breakout (`VCP`)
-   **Market Hypothesis:** Periods of extreme low volatility (ATR compression) must resolve into high volatility expansions.
-   **Why Edge Might Exist:** Market-makers accumulate orders during narrow ranges before marking prices up or down rapidly.
-   **Why Edge Might Fail:** High incidence of false starts and whipsaws in tight brackets.
-   **Suggested NY Session:** Any.
-   **Expected Frequency:** $5-8$ trades per month.
-   **Expected Correlation with Manipulante:** Low.
-   **Implementation Difficulty:** Medium.
-   **Overfitting Risk:** Medium.
-   **Cost Risk:** Low.
-   **News Dependency Risk:** Low.
-   **Objective Rule Sketch:** Identify periods where H1 ATR(14) is in the bottom 10th percentile of its 30-day range. Enter long/short on a M5 breakout of the narrow H1 channel.
-   **Initial Risk Management:** Stop loss at the opposite channel boundary; 2.5R target.
-   **Critical Metrics:** Compression index reliability.
-   **No-Go Conditions:** Pre-holiday sessions.
-   **First Batch Candidate:** **NO** (Requires complex historical percentile calculations).

### 6. Post-News Spike Mean Reversion (`NMR`)
-   **Market Hypothesis:** High-impact economic releases create artificial price spikes that quickly overextend and revert once the news is absorbed.
-   **Why Edge Might Exist:** Liquidity sweeps on news spikes often hit massive institutional bid/ask blocks, forcing a rapid return to fair value.
-   **Why Edge Might Fail:** Severe slippage, spread widening, and extreme risk of lookahead/lookbehind during news feeds.
-   **Suggested NY Session:** NY After-News.
-   **Expected Frequency:** $4-6$ trades per month.
-   **Expected Correlation with Manipulante:** Low.
-   **Implementation Difficulty:** High.
-   **Overfitting Risk:** High.
-   **Cost Risk:** Extreme.
-   **News Dependency Risk:** Critical.
-   **Objective Rule Sketch:** Enter contrarian short if a major US economic news release pushes EURUSD $> 3.0$ ATR within 5 minutes, followed by an immediate 1-minute candle closing back inside the 2-ATR band.
-   **Initial Risk Management:** Wide stops to prevent slippage sweeps; 1R target.
-   **Critical Metrics:** Slip-adjusted fill prices.
-   **No-Go Conditions:** Highly unstable spreads ($> 5$ pips).
-   **First Batch Candidate:** **NO** (Strictly forbidden due to news-filter dependency).

### 7. VWAP/Session Mean Reversion (`VMR`)
-   **Market Hypothesis:** Price oscillates around its intraday volume-weighted average price (VWAP). Large deviations are statistically temporary.
-   **Why Edge Might Exist:** VWAP represents the average institutional fill price; market participants actively rebalance inventory around this anchor.
-   **Why Edge Might Fail:** Strong, news-driven momentum days will ignore VWAP anchors completely, dragging contrarian accounts into deep drawdown.
-   **Suggested NY Session:** Midday NY.
-   **Expected Frequency:** $12-16$ trades per month.
-   **Expected Correlation with Manipulante:** Low.
-   **Implementation Difficulty:** Low.
-   **Overfitting Risk:** Medium.
-   **Cost Risk:** Low.
-   **News Dependency Risk:** Low.
-   **Objective Rule Sketch:** Enter long if price drops $> 1.5$ ATR below the daily VWAP between 15:00 and 19:00 GMT, with the RSI(14) on M5 crossing above 30.
-   **Initial Risk Management:** Stop loss at 2.2 ATR below VWAP; target is a touch of VWAP.
-   **Critical Metrics:** Mean reversion speed, maximum adverse excursion (MAE).
-   **No-Go Conditions:** Major central bank announcements (FOMC, ECB).
-   **First Batch Candidate:** **YES** (Standard institutional mean reversion concept).

### 8. Prior Day High/Low Failed Auction (`DAF`)
-   **Market Hypothesis:** Daily highs and lows are highly visible liquidity zones filled with resting stop-loss orders. Sweeping these levels without continuation triggers immediate reversals.
-   **Why Edge Might Exist:** Institutional desks sweep retail stops to accumulate large orders before pushing the market in the opposite direction.
-   **Why Edge Might Fail:** Clean breakout trend days will continue past daily levels, causing heavy losses.
-   **Suggested NY Session:** London / NY.
-   **Expected Frequency:** $5-9$ trades per month.
-   **Expected Correlation with Manipulante:** Low.
-   **Implementation Difficulty:** Medium.
-   **Overfitting Risk:** Low.
-   **Cost Risk:** Low.
-   **News Dependency Risk:** Medium.
-   **Objective Rule Sketch:** Enter short if price crosses above the prior day's high by $< 3.0$ pips, then closes back below the level on the M15 timeframe.
-   **Initial Risk Management:** Stop loss at the swing high of the sweep (+ 2 pips buffer); 2R target.
-   **Critical Metrics:** False breakout conversion rate.
-   **No-Go Conditions:** High ATR trend expansion day.
-   **First Batch Candidate:** **YES** (High mathematical edge, clean daily levels).

### 9. Asian Range Expansion (`ARE`)
-   **Market Hypothesis:** The Asian session (22:00 - 06:00 GMT) is a consolidation phase. A breakout of this range indicates strong session momentum.
-   **Why Edge Might Exist:** Directional volume from European institutions pushes the market out of the overnight bracket.
-   **Why Edge Might Fail:** High fakeout rate on EURUSD.
-   **Suggested NY Session:** London.
-   **Expected Frequency:** $8-12$ trades per month.
-   **Expected Correlation with Manipulante:** Low.
-   **Implementation Difficulty:** Low.
-   **Overfitting Risk:** Low.
-   **Cost Risk:** Low.
-   **News Dependency Risk:** Low.
-   **Objective Rule Sketch:** Enter long if price closes above the Asian Range high by at least 1 pip on M30, with volume $> 1.2\times$ the 10-bar average.
-   **Initial Risk Management:** Stop loss at the middle of the Asian Range; 2R target.
-   **Critical Metrics:** Breakout continuation ratio.
-   **No-Go Conditions:** Asian range is extremely wide ($> 25$ pips).
-   **First Batch Candidate:** **NO** (Superseded by more robust LBC breakout continuations).

### 10. Asian Range False Breakout (`ARF`)
-   **Market Hypothesis:** The first breakout of the Asian range is a liquidity sweep that traps retail participants before reverting.
-   **Why Edge Might Exist:** Captures stop hunts during early European pre-market hours.
-   **Why Edge Might Fail:** Strong news-driven trends will run straight through the breakout point.
-   **Suggested NY Session:** London.
-   **Expected Frequency:** $10-14$ trades per month.
-   **Expected Correlation with Manipulante:** Low.
-   **Implementation Difficulty:** Medium.
-   **Overfitting Risk:** Medium.
-   **Cost Risk:** Medium.
-   **News Dependency Risk:** Low.
-   **Objective Rule Sketch:** Enter contrarian long if price drops below the Asian Range low by $< 5$ pips and immediately prints a bullish rejection bar on M5.
-   **Initial Risk Management:** Stop loss 1 pip below the fakeout low; 1.5R target.
-   **Critical Metrics:** Fill rate and slippage.
-   **No-Go Conditions:** Asian session high-impact news (e.g., Bank of Japan, Australian employment).
-   **First Batch Candidate:** **NO** (Superseded by LBF which uses standardized ATR-based filters).

### 11. Midday Volatility Compression Breakout (`MDB`)
-   **Market Hypothesis:** The NY lunchtime period (16:30 - 18:30 GMT) compresses volume and volatility, setting up a sharp late-afternoon expansion.
-   **Why Edge Might Exist:** Rebalancing flows and market close institutional setups trigger quick breakouts.
-   **Why Edge Might Fail:** Extremely low volume can lead to directionless drift.
-   **Suggested NY Session:** Late NY.
-   **Expected Frequency:** $4-7$ trades per month.
-   **Expected Correlation with Manipulante:** Low.
-   **Implementation Difficulty:** Medium.
-   **Overfitting Risk:** High.
-   **Cost Risk:** High (due to spread widening near the NY close).
-   **News Dependency Risk:** Low.
-   **Objective Rule Sketch:** Identify range between 16:30 and 18:00 GMT. Enter long/short on breakout of this bracket on M5.
-   **Initial Risk Management:** Stop loss at the opposite range boundary; 2R target.
-   **Critical Metrics:** Spread degradation impact.
-   **No-Go Conditions:** Fridays or days before major US holidays.
-   **First Batch Candidate:** **NO** (Extreme spread risk during rollover hours).

### 12. London Close Rebalancing (`LCR`)
-   **Market Hypothesis:** The London fix (16:00 GMT / 11:00 EST) triggers massive institutional rebalancing flows. Once the fix is complete, price reverts to its pre-fix mean.
-   **Why Edge Might Exist:** Passive fund manager rebalancing orders are executed aggressively, temporarily pushing prices away from equilibrium.
-   **Why Edge Might Fail:** Rebalancing direction is highly variable and can sometimes trend strongly past the fix.
-   **Suggested NY Session:** Late NY (16:00 - 17:30 GMT).
-   **Expected Frequency:** $6-8$ trades per month.
-   **Expected Correlation with Manipulante:** Low.
-   **Implementation Difficulty:** Medium.
-   **Overfitting Risk:** Medium.
-   **Cost Risk:** Low.
-   **News Dependency Risk:** Low.
-   **Objective Rule Sketch:** If price moves $> 1.5$ ATR in a single direction in the 30 minutes leading to 16:00 GMT, enter contrarian at 16:05 GMT.
-   **Initial Risk Management:** Stop loss at the fix extreme swing high/low; 1R target.
-   **Critical Metrics:** Reversion velocity post-fix.
-   **No-Go Conditions:** End-of-quarter or end-of-year fixes where flows are massive and highly persistent.
-   **First Batch Candidate:** **YES** (Clean structural anomaly, highly researched in FX).

### 13. ADR Exhaustion Reversal (`ADR`)
-   **Market Hypothesis:** EURUSD has a stable Average Daily Range. Once the 20-day ADR is exhausted ($> 95\%$), the market is highly likely to exhaust and consolidate.
-   **Why Edge Might Exist:** Retail and automated momentum systems hit profit targets, reducing directional volume and causing mean reversion.
-   **Why Edge Might Fail:** High volatility trend days can exceed the daily ATR by $> 150\%$.
-   **Suggested NY Session:** Late NY.
-   **Expected Frequency:** $3-5$ trades per month.
-   **Expected Correlation with Manipulante:** Low.
-   **Implementation Difficulty:** Low.
-   **Overfitting Risk:** Low.
-   **Cost Risk:** High (due to rollover spread widening).
-   **News Dependency Risk:** Medium.
-   **Objective Rule Sketch:** If intraday range exceeds $100\%$ of the 20-day ADR, enter contrarian at the next M15 rejection bar, provided it occurs after 16:00 GMT.
-   **Initial Risk Management:** Stop loss 0.2 ADR above/below execution price; 1.5R target.
-   **Critical Metrics:** Frequency of ADR expansion days.
-   **No-Go Conditions:** Heavy volatility expansion days.
-   **First Batch Candidate:** **NO** (Extremely low sample size and high rollover risk).

### 14. Time-of-day Volatility Regime Switch (`TVS`)
-   **Market Hypothesis:** Intraday volatility behaves in highly predictable time brackets. Transitions between these brackets trigger major momentum breakouts.
-   **Why Edge Might Exist:** Volatility regimes are structurally driven by bank operating hours.
-   **Why Edge Might Fail:** Shift times can vary slightly during daylight saving transitions.
-   **Suggested NY Session:** Session Opens.
-   **Expected Frequency:** $6-8$ trades per month.
-   **Expected Correlation with Manipulante:** Low.
-   **Implementation Difficulty:** Medium.
-   **Overfitting Risk:** High.
-   **Cost Risk:** Low.
-   **News Dependency Risk:** Low.
-   **Objective Rule Sketch:** Monitor the ratio of M5 ATR to H1 ATR. Enter breakout when ratio exceeds the 90th percentile during the 08:00 - 09:00 GMT transition.
-   **Initial Risk Management:** ATR-based trailing stop; 3R target.
-   **Critical Metrics:** Regime boundary reliability.
-   **No-Go Conditions:** Market holiday periods.
-   **First Batch Candidate:** **NO** (Requires complex multi-timeframe regime analysis).

### 15. Micro Pullback Continuation after Expansion (`MPC`)
-   **Market Hypothesis:** Strong volatility expansions are followed by brief micro pullbacks that offer high-probability entry points in the trend direction.
-   **Why Edge Might Exist:** Institutional traders accumulate orders on minor retracements during strong momentum campaigns.
-   **Why Edge Might Fail:** Micro pullbacks can easily turn into deep reversals if the initial expansion was a fakeout.
-   **Suggested NY Session:** London / NY.
-   **Expected Frequency:** $15-20$ trades per month.
-   **Expected Correlation with Manipulante:** Medium.
-   **Implementation Difficulty:** High.
-   **Overfitting Risk:** High.
-   **Cost Risk:** Low.
-   **News Dependency Risk:** Low.
-   **Objective Rule Sketch:** After price breaks a H1 swing level by $> 1.0$ ATR, wait for a 2-bar pullback on M5 that closes near the EMA(20). Enter long/short on the first bar that resumes the breakout direction.
-   **Initial Risk Management:** Stop loss below the swing low of the pullback; 2R target.
-   **Critical Metrics:** Continuation ratio, average pullback depth.
-   **No-Go Conditions:** Low overall daily ATR.
-   **First Batch Candidate:** **YES** (Clean momentum concept, highly standard).

### 16. Trend Day Participation (`TDP`)
-   **Market Hypothesis:** True "Trend Days" exhibit clean, low-drawdown linear drift that continues in one direction for $> 80\%$ of the daily range.
-   **Why Edge Might Exist:** Large-scale institutional asset reallocation creates persistent order flow that cannot be easily absorbed.
-   **Why Edge Might Fail:** Trend days are rare ($< 15\%$ of all days), leading to high false-entry drawdowns on range days.
-   **Suggested NY Session:** London / NY.
-   **Expected Frequency:** $4-6$ trades per month.
-   **Expected Correlation with Manipulante:** High.
-   **Implementation Difficulty:** Medium.
-   **Overfitting Risk:** Low.
-   **Cost Risk:** Low.
-   **News Dependency Risk:** Medium.
-   **Objective Rule Sketch:** If price closes above the London session high by $> 1.5$ ATR at 14:00 GMT, enter long on M15 with a wide trailing stop.
-   **Initial Risk Management:** Wide 2.5 ATR stop loss; trail using swing lows.
-   **Critical Metrics:** Trend day identification filter.
-   **No-Go Conditions:** Tight rangy market regimes.
-   **First Batch Candidate:** **NO** (High correlation with pre-existing trend portfolios).

### 17. Range Day Fade (`RDF`)
-   **Market Hypothesis:** The market spends $\approx 80\%$ of its time in consolidations. Selling the boundaries of these ranges provides a highly consistent statistical edge.
-   **Why Edge Might Exist:** Price channels provide clean, visible brackets where contrarian liquidity absorbs weak momentum attempts.
-   **Why Edge Might Fail:** When a range day turns into a trend day, losses accumulate rapidly.
-   **Suggested NY Session:** Asian / Midday NY.
-   **Expected Frequency:** $15-22$ trades per month.
-   **Expected Correlation with Manipulante:** Low.
-   **Implementation Difficulty:** Low.
-   **Overfitting Risk:** Low.
-   **Cost Risk:** Low.
-   **News Dependency Risk:** Low.
-   **Objective Rule Sketch:** Identify range boundaries using a Bollinger Band (20, 2) on M15. Enter short/long when price touches the outer band and RSI(14) diverges from price.
-   **Initial Risk Management:** Stop loss 5 pips outside the band; 1.5R target.
-   **Critical Metrics:** Range survival duration.
-   **No-Go Conditions:** High daily ATR expansions.
-   **First Batch Candidate:** **YES** (Excellent diversification against trend models).

### 18. Weekly Level Failed Auction (`WLF`)
-   **Market Hypothesis:** Weekly highs and lows are major institutional reference points. Sweeping these levels triggers long-term reversals.
-   **Why Edge Might Exist:** Weekly levels accumulate the highest concentration of resting buy/sell stop liquidity.
-   **Why Edge Might Fail:** Reversals can take several sessions to materialize, leading to high capital costs or premature stops.
-   **Suggested NY Session:** Any.
-   **Expected Frequency:** $2-4$ trades per month.
-   **Expected Correlation with Manipulante:** Low.
-   **Implementation Difficulty:** High.
-   **Overfitting Risk:** Low.
-   **Cost Risk:** Low.
-   **News Dependency Risk:** Medium.
-   **Objective Rule Sketch:** Enter short if price breaks the previous week's high, then prints a bearish weekly breakout failure on the H4 timeframe.
-   **Initial Risk Management:** Wide 1.5 daily ATR stop loss; 3R target.
-   **Critical Metrics:** Weekly sweep clean conversion.
-   **No-Go Conditions:** Clear weekly momentum continuation.
-   **First Batch Candidate:** **NO** (Extremely low sample size).

### 19. Previous Session Continuation (`PSC`)
-   **Market Hypothesis:** The momentum established during the Asian and early European pre-market sessions continues through the European morning.
-   **Why Edge Might Exist:** Captures the early institutional trend bias before US players enter.
-   **Why Edge Might Fail:** Reversals at the NY open can wipe out early session gains.
-   **Suggested NY Session:** London.
-   **Expected Frequency:** $6-9$ trades per month.
-   **Expected Correlation with Manipulante:** Low.
-   **Implementation Difficulty:** Low.
-   **Overfitting Risk:** Low.
-   **Cost Risk:** Low.
-   **News Dependency Risk:** Low.
-   **Objective Rule Sketch:** If the 07:00 GMT close is above the 00:00 GMT close, enter long on the first M15 pullback to the 8-period EMA.
-   **Initial Risk Management:** Stop loss below the Asian low; 1.5R target.
-   **Critical Metrics:** Intraday trend persistence.
-   **No-Go Conditions:** Major European economic releases at 08:00 GMT.
-   **First Batch Candidate:** **NO** (Superseded by standard London breakout continuations).

### 20. Liquidity Alternative No-Manipulante (`LAN`)
-   **Market Hypothesis:** By monitoring multi-session liquidity brackets (excluding those used by the main trend models), we can capture clean swings away from high-density retail zones.
-   **Why Edge Might Exist:** Targets ignored structural liquidity brackets where institutional resting orders are placed.
-   **Why Edge Might Fail:** Extreme low volatility can lead to directionless grinding.
-   **Suggested NY Session:** London / NY.
-   **Expected Frequency:** $8-12$ trades per month.
-   **Expected Correlation with Manipulante:** Zero (Designed to target completely disjoint setups).
-   **Implementation Difficulty:** Medium.
-   **Overfitting Risk:** Low.
-   **Cost Risk:** Low.
-   **News Dependency Risk:** Low.
-   **Objective Rule Sketch:** Map the H4 swing high/low points. Enter contrarian long/short on a M15 false sweep of these levels, provided the setup does not align with the standard intraday time-of-day entry triggers.
-   **Initial Risk Management:** Stop loss 10 pips below the sweep low; 2.5R target.
-   **Critical Metrics:** Disjoint trade correlation coefficient.
-   **No-Go Conditions:** Highly correlated intraday trend runs.
-   **First Batch Candidate:** **YES** (Critical portfolio diversifier).
