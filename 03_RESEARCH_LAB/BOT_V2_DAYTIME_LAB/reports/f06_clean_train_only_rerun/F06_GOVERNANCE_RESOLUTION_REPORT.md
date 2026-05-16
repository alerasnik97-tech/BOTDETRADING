# F06 GOVERNANCE RESOLUTION REPORT

Generated: 2026-05-15 — READ-ONLY governance analysis. No engine/F06/backtest run. No adapter code.
PR #6 head: e61d36ae4f679e3f6f8dd9952d07496bb917ac2a

## 1. Status
F06_GOVERNANCE_READY_FOR_OWNER_DECISION

## 2. Executive Summary
Discovery proves "F06" has **no safe code definition** in the current repo. `STRATEGY_REGISTRY`
(`research_lab/strategies/__init__.py`) contains 60+ strategies keyed by `NAME`; **none is "F06"**.
`F06_EVIDENCE_REBUILD_FOUNDATION_REPORT.md` states explicitly that **F06/F08/F12 are NOT-CERTIFIED
families whose V50B evidence was invalidated**. Therefore F06 is a *V50B-era family label*, not a
strategy module — and reconstructing the F06→strategy mapping from V50B artifacts is FORBIDDEN
(invalidated/quarantined source of truth). F06 cannot be defined from code; it requires an explicit
owner governance decision. Adapter implementation stays BLOCKED.

## 3. F06 Candidate Discovery
| candidate_id | path | strategy_name | entrypoint | tracked | safe_path | evidence_for_F06 | evidence_against_F06 | safe_to_select | decision |
|---|---|---|---|---|---|---|---|---|---|
| REG-* | `research_lab/strategies/*` (60+ via `STRATEGY_REGISTRY`) | e.g. `bollinger_mean_reversion_simple`, `ema_trend_pullback`, `eurusd_c4_ict_align`, … | `generate_signal(frame,i,params)` + `NAME`/`WARMUP_BARS` | YES | YES | none — registry is NAME-keyed, no family tag | `rg "\bF06\b"` over `research_lab/` = 0 matches | NO (no evidence any is F06) | owner must designate |
| V50B-EVID | `reports/.../v50b_*`, `pre_claude_blocker_remediation` | — | — | mixed | NO (FORBIDDEN) | historically F06 was a V50B family | V50B evidence INVALIDATED (foundation report §3) | NO | forbidden source |
| BACKUP/CANON | `research_lab/*_BACKUP_*`, `reports/canonical_*/research_lab/**` | — | — | mixed | NO | — | stale/non-authoritative | NO | forbidden |
Result: **F06_GOVERNANCE_DECISION_REQUIRED** (= F06_MODULE_NOT_FOUND_CONFIRMED; reconstruction from V50B is unsafe).

## 4. Recommended F06 Definition
Claude RECOMMENDS (does NOT impose — owner must choose the concrete strategy):
- **F06 := exactly ONE owner-selected, tracked strategy `NAME` from `STRATEGY_REGISTRY`**, evaluated
  with **ONE canonical, pre-registered, immutable config** (`config_id = F06_PHASE3_CANONICAL_001`).
- Do NOT recover F06 from V50B (invalidated/forbidden). Treat Phase 3 as a clean, single, named
  baseline — not a family rebuild.
- recommended_option: hybrid of Option A + Option C (single strategy + single canonical config; the
  "family" stays a conceptual label, ranking is a one-row summary — see §5).
- required_owner_decision: WHICH `STRATEGY_REGISTRY` NAME is F06. Claude cannot infer this — zero
  code evidence links any registry entry to "F06".
- what_blocks_implementation: absence of the F06→NAME decision + config taxonomy.
- what_can_proceed_read_only: loader design, EngineConfig ruling, atr provenance, validator pinning
  (all done here); NOT adapter code.

## 5. Ranking vs No-Sweep Ruling
Contract requires `ranking/RANKING.csv` with `(family_id, config_id, N_train, PF_train,
Total_R_train, WR_train)`. Phase 3 forbids sweep/optimization/open search.
- **Recommended ruling: single-row ranking.** Exactly ONE `config_id`
  (`F06_PHASE3_CANONICAL_001`). RANKING.csv is a *summary*, not a search output. This eliminates
  sweep risk and satisfies `ranking_schema.json` (its degeneracy rule hard-fails only if
  single-unique AND `configs > 1`; with one config there is no degeneracy violation).
- Multi-config is acceptable ONLY under Option B: an owner-pre-registered, frozen, hash-stamped
  immutable config list created BEFORE the run, with NO parameter selection/tuning during/after the
  run. Recommended to DEFER multi-config to a later, separately-audited phase.
- p-hacking guard: no config may be added/edited after data is seen; config set + hashes recorded
  in MANIFEST before run_backtest; net metrics only (no validation columns).
Result: **RANKING_GOVERNANCE_NEEDS_OWNER_DECISION** (recommended = single-row; owner confirms).

## 6. Config Taxonomy Proposal
- `family_id = "F06"` (fixed string; satisfies `manifest.families == ["F06"]`).
- `config_id = "F06_PHASE3_CANONICAL_001"` (single, pre-registered).
- `parameter_hash` = sha256 of the canonical params dict (recorded pre-run).
- `result_signature` = sha256 of the produced ledger summary (post-run, integrity only).
- The canonical params dict (the exact `params` passed to `run_backtest`) MUST be owner-approved and
  frozen in `CONFIG_USED.yaml` + MANIFEST before any run.

## 7. Decisions Required From Owner
1. The exact `STRATEGY_REGISTRY` NAME that IS F06 (single, tracked).
2. Confirm single-config single-row ranking (recommended) vs pre-registered frozen multi-config.
3. The canonical F06 params dict (frozen) and session/risk params consistent with Phase 3.
4. Cost ruling: explicit `assumed_spread_pips` value vs an audited `cost_profile` (see continuation report §4).
5. Resolution for ledger `gross_r` / `sl_pips` which the engine trade record does NOT emit (see continuation report §5): contract clarification or owner-approved derivation (NO engine core modification).

## 8. Adapter Impact
Until §7 is resolved: the adapter cannot select a strategy, cannot populate `family_id/config_id`,
cannot build a schema-valid ledger/ranking, and cannot pass the validator. Implementation BLOCKED.
All mechanical surfaces (entrypoint, loader, EngineConfig, atr, validator) are pinned and ready to
consume the owner decisions.

## 9. Final Recommendation
Proceed to an **owner governance decision round** (see `NEXT_PROMPT_F06_GOVERNANCE_OWNER_DECISION.md`).
Do NOT author an implementation prompt. Do NOT reconstruct F06 from V50B. Keep F06 NOT CERTIFIED.
