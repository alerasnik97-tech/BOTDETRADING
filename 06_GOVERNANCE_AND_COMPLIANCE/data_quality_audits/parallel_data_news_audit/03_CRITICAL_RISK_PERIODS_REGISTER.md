# INSTITUTIONAL CRITICAL RISK PERIODS REGISTER
**Scope:** Strategy Execution Safety Layers & Slippage Stress Calibration  
**Target Mechanism:** Downside Protection & Realistic Fill Assertions (Manipulante 3.0 Integration)  
**Execution Type:** Read-Only Parallel Agent Risk Register (Zero-Interference Lockdown Protocol)

---

## 1. Structural Market-Clearing Risks (Rollover Zone)

Empirical spread analysis extracted from certified vault snapshots demonstrates systematic, repeatable market degradation periods independent of macroeconomic news releases. Strategy logic engines must natively bypass these windows to prevent catastrophic simulated fill assumptions.

### Daily Rollover Exclusion Zone Specification
- **Critical Temporal Boundary:** `16:55:00` to `17:15:00` (America/New_York local exchange time).
- **Physical Root Cause:** Interbank liquidity withdrawal during the transition between the New York session close and Asian market open.
- **Empirical Spread Degradation:** While the long-term baseline median spread holds at `0.30` pips, rollover dropouts systematically trigger transient spread spikes exceeding **$P_{99}$ bounds ($1.40+$ pips)**, with localized microstructural extremes widening up to **$13.30$ pips**.
- **Mandatory Action:** Enforce an automated execution-freeze layer during this 20-minute daily cycle. Existing positions should not attempt market-order exiting within this window unless explicitly structured to absorb massive slippage markdowns.

---

## 2. Macroeconomic News Shock Buffers (Tier-1 Anchors)

High-impact releases inject near-instantaneous volatility expansions accompanied by order book thinning. Simulations assuming instantaneous standard fills at the exact minute mark introduce severe backtest illusion and look-ahead bias.

### Dynamic Release-Shock Buffer Table

| Anchor Release Family | Release Time Context (NY) | Mandatory Exclusion Buffer | Volatility Profile & Fill Danger | Buffer Implementation Policy |
| :--- | :--- | :--- | :--- | :--- |
| **Non-Farm Payrolls** | `08:30` (First Friday) | `[-1 min, +5 min]` | Extreme localized order book gapping. | Hard signal suppression; halt pending entry orders. |
| **Core / Standard CPI** | `08:30` | `[-1 min, +5 min]` | Acute two-way directional sweeps. | Hard signal suppression; wide trailing stop evaluation. |
| **FOMC Rate Decision** | `14:00` | `[-2 min, +10 min]` | Extended institutional pre-positioning whipsaws. | Absolute entry block; dynamic spread validation threshold. |
| **FOMC Press Conf.** | `14:30` | `[0 min, +45 min]` | Sustained directional regime uncertainty. | Restrict sizing parameters; transition to secondary rules. |
| **ECB Press Conf.** | `08:45` / `09:45` | `[-1 min, +15 min]` | Multi-currency cross-rate contagion. | Monitor interbank spread; suppress secondary setups. |

> [!WARNING]  
> Executing automated entry logic exactly at `T_0` during Tier-1 events represents an architectural violation. Engine developers must verify that simulation loops evaluate dynamic spread checks before executing fills.

---

## 3. Slippage Stress Calibration Profiles

The forensic closure of legacy models (e.g., Manipulante 2.0 Gate 6 Mini) revealed acute performance sensitivity to execution friction. A strategy exhibiting apparent baseline profitability can rapidly cross into terminal drawdown under realistic live transaction costs.

### Historical Friction Sensitivity Curve (Gate 6 Audit Reference)
- **Zero Friction Ideal ($0.0$ pip slippage):** Net Profit Factor `0.9734` (Marginal/Failing).
- **Minimal Friction ($0.1$ pip slippage):** Net Profit Factor `0.7809` (Severe degradation).
- **Moderate Friction ($0.2$ pip slippage):** Net Profit Factor `0.7033` (Unviable profile).
- **Institutional Stress ($0.5$ pip slippage):** Net Profit Factor `0.4781` (Complete logic failure).

### Mandatory Sweeping Specification for Manipulante 3.0
To achieve baseline institutional clearance, any candidate variant of **Manipulante 3.0** must undergo automated multi-tier slippage evaluation across the continuous dataset (`2015_01` to `2026_04`). 

**Minimum Approval Threshold:** Candidates must maintain a net Profit Factor strictly greater than **$1.15$** under a sustained **$0.2$ pip** asymmetric slippage penalty applied across all execution legs.

---

## 4. Operational Sign-Off

This register serves as the formal read-only risk control layer for forward quantitative modeling. 
- **Codebase Access:** Read-only compliance mode active.
- **Production Status:** Zero interference verified.
- **Audit File Verification:** File generated directly inside authorized boundaries.
