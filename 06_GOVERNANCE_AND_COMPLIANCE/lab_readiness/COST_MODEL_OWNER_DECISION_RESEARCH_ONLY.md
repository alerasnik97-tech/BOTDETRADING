# COST MODEL OWNER DECISION — RESEARCH ONLY

**DATE**: 2026-05-17
**SCOPE**: research / train-only laboratory evaluation
**INPUT BRANCH**: `fix/institutional-cost-profile-routing-20260517` @ `4b9e799e`
**RECORDED BY**: Cost Model Governance Officer (owner decision delegated and recorded this phase)

---

## 1. Status

`COST_MODEL_OWNER_DECISION_RECORDED_RESEARCH_ONLY`

---

## 2. Owner Decision

**D1 — Conservative tier ratified (research/train-only):**
- `conservative_spread_multiplier = 1.20`
- `conservative_slippage_multiplier = 1.30`

**D2 — Stress tier unchanged (pre-existing institutional values, not invented):**
- `stress_spread_multiplier = 1.35`
- `stress_slippage_multiplier = 1.60`

**D3 — Commissions unchanged:**
- `commission_per_lot_roundturn_usd = 7.0` USD (current `DEFAULT_COMMISSION_ROUNDTURN_USD`); no per-profile commission changes.

**D4 — Institutional ordering enforced:**
- `base < conservative < stress` (strict monotonicity in both spread and slippage).
- Effective entry multipliers: base ×1.00/×1.00 · conservative ×1.20/×1.30 · stress ×1.35/×1.60.

**D5 — This is NOT approval for:** real money · FTMO · demo · production · incubation · deployment.

**D6 — Any future change requires a separate phase** with broker/FTMO/spread evidence. These magnitudes are laboratory defaults, not market-calibrated values.

---

## 3. Rationale

- **base**: normal conservative cost (existing default spread 1.2 pips / slippage 0.2 pips, no profile multiplier). Not optimistic.
- **conservative**: moderate penalty above base — realistic stress for serious retail/funded evaluation, but strictly below the existing stress ceiling.
- **stress**: strong penalty, already institutionalized (×1.35 spread / ×1.60 slippage). **Not invented, not modified.**
- conservative kept **below** stress so the three tiers are monotone and non-degenerate; 1.20/1.30 sits cleanly between base (×1.00) and stress (×1.35/×1.60). A larger conservative spread (e.g. ×1.50) would exceed the existing stress spread (×1.35) and break ordering — rejected.
- Commissions untouched to preserve the existing institutional convention and isolate the cost decision to spread/slippage.

---

## 4. Safety

- backtest: NO
- strategy run: NO
- optimization / sweep / validation / walk-forward: NO
- holdout: NO
- 2025/2026: NO
- news / high precision: NO
- data mutation: NO
- engine signal logic: untouched

This document only **records a governance decision** over already-implemented, already-tested config parameters (`fix/institutional-cost-profile-routing-20260517`, 56/56 tests green). It introduces no execution and no data access.
