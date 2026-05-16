# NEXT PROMPT — CLAUDE FINAL PRE-LAB AUDIT (FINAL GATE)

> Created because this phase met its preconditions (no critical conflicts,
> root/governance intact, F06 119/119, no secrets/data staged, blockers list
> explicit). **This is the LAST gate — do NOT run it until ALL entry
> conditions below are TRUE. It does not itself authorize the lab.**

## Entry conditions (ALL must be TRUE first — verify, do not assume)
- [ ] B2 IMPORT cleared: `import research_lab.light_runner` works (config
      `DEFAULT_NEWS_FILE` symbol drift fixed) on the canonical line.
- [ ] B3 PATH cleared: no active refs to relocated root `scripts/`.
- [ ] B4 DATA cleared: `forex_factory_cache.csv`, `news_eurusd_v2_utc.csv`,
      hi-precision dukascopy present with recorded provenance;
      `canonical_anchor_events.csv` provenance audited/accepted.
- [ ] B5 TEST: `research_lab/tests` green (or every residual failure has an
      owner-accepted waiver — never faked).
- [ ] B6 token-docs `.gitignore` policy decided by owner.
- [ ] B7 clean-sync 50+ commits triaged (engine/research/cloud dispositioned).
- [ ] Branch state: canonical line clean, in sync, no unrelated-history merge.

## Objective
Single, honest, read-only **READY_FOR_LAB vs NOT_READY** determination with an
evidence-backed score (reuse the Phase E rubric: root / git / imports-tests /
F06 / data / safety, ≥90 and zero critical blockers required). No false green
light.

## Scope
1. Re-run the full Phase E audit blocks on the then-canonical branch.
2. Verify each entry-condition box with commands, not memory.
3. Safe tests only: `import research_lab` (+ `light_runner`),
   F06 (expect ≥119), `research_lab/tests` (classify honestly).
4. Produce `PHASE_F_LAB_AUTHORIZATION_DECISION.md`: `READY_FOR_LAB` only if
   score ≥ 90 AND zero critical blockers AND no missing required data AND no
   secrets — otherwise enumerate exact remaining blockers.

## Hard rules
- No backtest / strategy / F06-real / optimization / sweep / validation /
  holdout / 2025-2026 analysis. No engine/trading-logic edits. No data
  regeneration/download. No ZIP. No main. No force push. No unrelated-history
  merge. Lab stays unauthorized until this gate explicitly passes.
