# PHASE 41 - MANIPULANTE HYBRID REPLAY REPORT

## Status
**HYBRID_REPLAY_PASS_HIGH_CONFIDENCE**

## Metrics
- **Expected Signals**: 77
- **Replay Signals**: 77
- **Signal Match**: 100%
- **PF Bruto (Replay)**: 2.12
- **Scorecard**: 84/100

## Differences Detected
- Outcome mismatches due to SL buffer precision (0.5 pips fixed in bot vs variable/unknown in old baseline).
- Outcome mismatches due to forced daily/weekend closing logic (Protective overlay).

## Technical Note
- Replay script: `BOT_V2_DAYTIME_LAB/src/phase41_manipulante_hybrid_replay.py`
- Source: `Phase27 Historical Trades` + `Certified M3 Data`.
