# COST MODEL OWNER DECISION — RESEARCH ONLY

**DATE**: 2026-05-17
**INPUT BRANCH**: `fix/institutional-cost-profile-routing-20260517` @ `4b9e799e`
**INFRA BRANCH**: `infra/formal-runner-cost-gates-20260517`
**RECORDED BY**: Cost Model Governance Officer (owner decision delegated and recorded this phase)

---

## 1. Status

`COST_MODEL_OWNER_DECISION_RECORDED_RESEARCH_ONLY`

---

## 2. Decision

Owner ratifies, for **research / train-only** use:

- `conservative_spread_multiplier = 1.20`
- `conservative_slippage_multiplier = 1.30`

Owner keeps unchanged (pre-existing institutional values — not invented):

- `stress_spread_multiplier = 1.35`
- `stress_slippage_multiplier = 1.60`

Commissions: **no changes** (`commission_per_lot_roundturn_usd = 7.0` USD remains the default; no per-profile commission changes).

---

## 3. Scope

**Approved only for:** research · train-only · internal comparison · laboratory dossiers.

**NOT approved for:** real · FTMO · demo · production · incubation · deployment.

Revisable in a future phase with real broker/spread/slippage evidence.

---

## 4. Rationale

- **base** = normal conservative cost (existing default spread 1.2 pips / slippage 0.2 pips, no profile multiplier; not optimistic).
- **conservative** = moderate penalty above base, realistic for serious retail/funded evaluation.
- **stress** = strong, pre-existing institutional penalty (×1.35 / ×1.60) — kept, not modified.
- conservative sits strictly **below** stress; a larger conservative spread (e.g. ×1.50) would exceed the existing stress spread (×1.35) and break ordering — rejected.
- Strict monotonicity enforced: `base < conservative < stress` (spread and slippage).
- Commissions untouched to isolate the cost decision to spread/slippage.
- Magnitudes are laboratory defaults; not market-calibrated; revisable with future broker/FTMO evidence (separate phase).

---

## 5. Safety

- backtest: NO
- strategy run: NO
- optimization / sweep / validation / walk-forward: NO
- holdout: NO
- 2025/2026: NO
- news / high precision: NO
- data mutation: NO
- engine signal logic: untouched

This document only **records a governance decision** over already-implemented, already-tested config parameters (`fix/institutional-cost-profile-routing-20260517`). No execution, no data access.
