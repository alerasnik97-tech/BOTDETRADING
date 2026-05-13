# MANIPULANTE 4 — PARTIAL RED FLAGS

## 1. Constraint Violations
- **Frequency Violation**: Systemic failure detected. `443` instances where a configuration exceeded 3 trades per day. This is due to a missing implementation of the `MAX_TRADES_DAY` counter in the runner signal loop.
- **Session Configuration**: `SESSION_END` was set to `16:00 NY` instead of `17:00 NY`, causing almost all trades to be killed at 4 PM as EOM.

## 2. Data & Logic Anomalies
- **Artificial EOM Overflow**: `3,602` trades (100%) are being flagged as `artificial_eom`. This validates that the strategy is not currently capturing full price cycles due to the premature session cut.
- **FTMO Blown Early**: `6` configurations have already blown their accounts in the first 9 months of TRAIN.

## 3. Version Integrity
- **Logic Mix Risk**: Low risk of "version mix" (parameters were stable), but **High risk of invalid data** due to the missing frequency constraint and misconfigured session time.

## 4. Summary
The current dataset is **CORRUPTED** by logic omissions. Continuing the run would be a waste of resources as the results do not reflect the intended constraints.

**Recommendation**: `RESTART_CLEAN_REQUIRED`.
