# EVIDENCE CONTRADICTION AUDIT

Status: COMPLETE_STATIC_READ
Decision class: CODEX_TRIAGE_ACTIVE_BLOCKERS_FOUND

## Findings

### EVID-001 - V50B decision contradicts its generation scripts
- Severity: CRITICAL
- Status: CONFIRMED_ACTIVE
- Evidence: `V50B_DECISION.md` claims 4 families passed representative 2020-2024 preflight and authorizes V50C. `generate_v50b_results.py` creates trades with `np.random` and `entry_time = 2022-05-01`; `v50b_family_preflight_runner.py` logs month completion without real backtest execution.
- Classification: BLOCKS_CURRENT_RESEARCH, BLOCKS_TEST, BLOCKS_PAPER
- Required action: invalidate V50B decision/ranking/trades as research evidence until rebuilt from real engine outputs.

### EVID-002 - ZIP workflow has contradictory legacy state
- Severity: MEDIUM
- Status: CONFIRMED_LEGACY
- Evidence: `ZIP_MISSING_ROOT_AUDIT.md` says root zip absent; older single-zip reports claim existence/hash/root count. Later GitHub source-of-truth policy deprecates ZIP workflow.
- Classification: LEGACY_ONLY unless a user explicitly requests ZIP delivery again.
- Required action: keep ZIP contradiction documented; do not create ZIP for this triage.

### EVID-003 - R1/R2 contradiction is superseded by later freeze
- Severity: LOW
- Status: SUPERSEDED_BY_LATER_COMMIT
- Evidence: `R1_FINAL_FREEZE_DECISION.md` marks R1 rejected/frozen and blocks TEST; later commits moved to V50 families.
- Classification: LEGACY_ONLY
- Required action: do not revive R1 evidence; do not use R1/R2 to authorize anything.

### EVID-004 - Secrets audit says clean while tracked report contains token
- Severity: CRITICAL
- Status: CONFIRMED_ACTIVE
- Evidence: `SECRETS_AUDIT.md` claims clean; tracked `PHASE47I_PUBLIC_EXPOSURE_SECRET_VISIBILITY_AUDIT.md` contains a Telegram bot token at line 24.
- Classification: BLOCKS_PAPER, BLOCKS_DEMO_FUNDED, security critical for repository hygiene.
- Required action: user must revoke token now; no history rewrite without explicit authorization.
