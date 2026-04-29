# PHASE31 PROP FIRM SURVIVAL SIMULATOR REPORT

## Objective
Compare Phase25 and TP1.4_BE0.5_BF70 under configurable prop-firm rules. No real trading.

## Rules
- FTMO Challenge: 10% target, 5% max daily loss, 10% max loss, 4 minimum trading days, unlimited period.
- FTMO Verification: 5% target, same loss limits.
- Funded: no profit target in simulation, same loss limits.

## Monte Carlo
- Paths per cell: 10000
- Bootstrap: monthly block bootstrap.

## Risk Recommendation
{
  "PHASE25": {
    "strategy": "PHASE25",
    "challenge_conservative_risk_pct": 0.75,
    "challenge_balanced_risk_pct": 0.75,
    "challenge_aggressive_warning_pct": 1.0,
    "verification_risk_pct": 0.75,
    "funded_account_risk_pct": 0.75,
    "max_risk_not_to_exceed_pct": 0.75,
    "note": "Do not use a risk level that creates material breach probability in Monte Carlo or funded rolling windows."
  },
  "TP1.4_BE0.5_BF70": {
    "strategy": "TP1.4_BE0.5_BF70",
    "challenge_conservative_risk_pct": 0.75,
    "challenge_balanced_risk_pct": 0.75,
    "challenge_aggressive_warning_pct": 1.0,
    "verification_risk_pct": 0.75,
    "funded_account_risk_pct": 0.75,
    "max_risk_not_to_exceed_pct": 0.75,
    "note": "Do not use a risk level that creates material breach probability in Monte Carlo or funded rolling windows."
  }
}

## Strategy Comparison
{
  "challenge_best": "TP1.4_BE0.5_BF70",
  "verification_best": "TP1.4_BE0.5_BF70",
  "funded_best": "TP1.4_BE0.5_BF70",
  "authority_recommendation": "PHASE25 remains authority; candidate remains shadow.",
  "psychological_operability": "Candidate has lower non-win streak and higher WR; Phase25 has higher PF."
}

## Verdict
PHASE31_PROP_FIRM_READY_CONSERVATIVE_RISK

## Next Step
Use the simulator for paper evaluation planning only; do not trade real or MT5.
