# NEXT PROMPT RETURN TO FINAL APPROVED PRIORITY A SKELETONS

Branch hygiene has been applied before skeleton implementation.

Use skeleton implementation only if `BRANCH_HYGIENE_FINAL_APPLY_REPORT.md` says one of the following:

- `BRANCH_HYGIENE_FINAL_CLEAN`
- `BRANCH_HYGIENE_PARTIAL_OWNER_REVIEW_REQUIRED` with no operational skeleton blocker accepted by owner

Current canonical branch:

- `governance/claude-strategy-intake-audit-20260516`

Current canonical head at hygiene time:

- `519e3054c47dfa8ce1cbee4b3cbd2b19527517a4`

Do not use:

- `clean-sync-branch`
- deleted legacy governance branches
- historical V50B/F06 branches as implementation base
- any open F06 PR branch as a skeleton base

Approved Priority A skeleton list:

1. MR-01 Anchor Elastic
2. MR-02 VWAP Stretch Reversion
3. TP-01 London-NY Momentum Pullback, reformulated
4. VE-ORB Volatility Expansion

Explicit exclusions:

- VE-01 remains REVIEW because of phantom parameters.
- SD-01 remains REJECTED because of high correlation with Manipulante.
- ED-01 remains DEFERRED because news data is not certified.

Safety contract for next phase:

- no backtest
- no strategy run
- no optimization
- no sweep
- no validation
- no holdout
- no 2025/2026
- no F06 real
- no F06 adapter
- no engine change
- no data mutation
- no main touch
- no force push
- no ZIP workflow
- no root files

Skeleton output must be limited to institutional research-lab code/docs for the four approved Priority A candidates, with explicit tests and governance report.
