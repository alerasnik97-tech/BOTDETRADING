# MANIPULANTE 4 — PRE-COMMITMENT PROTOCOL

Frozen BEFORE any code execution. Immutable.

## Hypothesis
Sweep quality discrimination + displacement gate produce measurable edge vs undiscriminated M3.

## Temporal Partitions
- TRAIN: 2020-01 to 2021-12
- VAL: 2022-01 to 2023-12
- TEST: 2024-01 to 2026-04 (SEALED until VAL approved)

## Configs
- Maximum: 150
- Sampling: stratified with RANDOM_SEED = 20260513

## Variables Permitted
- level_type: Asia HL, PDH/PDL, PWH/PWL (3 values)
- sweep_depth_atr_min: 0.05, 0.10, 0.20 (3 values)
- reclaim_max_minutes: 5, 15, 30 (3 values)
- displacement_body_atr_min: 0.75, 1.00, 1.25 (3 values)
- displacement_closes_beyond_structure: True (fixed)
- entry_type: stop_confirmation, fvg_50pct (2 values)
- SL: sweep_extreme+1.5p, structure_extreme+1.5p (2 values)
- TP: 2.0R, 2.5R (2 values)
- BE: none, 1.25R (2 values)
- session: 07:00-11:00 NY, 08:00-11:00 NY (2 values)

Theoretical max: 3×3×3×3×2×2×2×2×2 = 2592
Sampled to ≤150 using seed 20260513.

## Selection Criteria (immutable)
- Only VAL metrics used for candidate selection
- TEST used only as pass/fail after VAL candidate frozen
- Metric: net_r (post commission + slippage)

## Kill Criteria (immutable)
- Best TRAIN PF_net < 1.0 → M4_MICRO_RED
- Best VAL PF_net (slip 0.2) < 1.05 → M4_MICRO_RED
- FTMO blown in majority → M4_MICRO_RED
- EOM artificial > 0 in metrics → BLOCKED
- N_val < 40 for best candidate → INCONCLUSIVE
- Profit concentration > 60% in ≤3 trades → M4_MICRO_RED
- Slippage 0.2 collapses edge completely → M4_MICRO_RED

## Expansion Criteria (immutable)
- PF_val_net (slip 0.2) >= 1.15
- PF_test_net (slip 0.2) >= 1.00
- N_val >= 40, N_test >= 40
- FTMO not blown early in VAL
- EOM artificial in metrics = 0
- Independent verify match = YES

## Costs
- Commission: $5.00/lot round-turn FTMO
- Slippage: 0.0 and 0.2 mandatory, 0.5 optional
- News: AM Fortress v3, fail-close
- Rollover: 16:55-17:15 NY blocked
- Tier-1 buffers: standard -1/+5 min, FOMC -2/+10 min

## EOM Rules
- Artificial EOM trades excluded from metrics
- Count must be reported in EOM_AUDIT.csv
- If artificial_eom_in_metrics > 0: BLOCKED
