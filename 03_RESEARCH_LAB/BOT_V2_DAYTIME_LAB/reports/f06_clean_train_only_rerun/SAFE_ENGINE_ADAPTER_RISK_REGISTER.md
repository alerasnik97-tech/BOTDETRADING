# SAFE ENGINE ADAPTER — RISK REGISTER

Generated: 2026-05-15 — DESIGN ONLY. PR #6 head: d988c03ad7a60c0080a767baf740161346b1222c
Severity scale: LOW / MED / HIGH / CRITICAL. Likelihood: LOW / MED / HIGH.
"blocking_before_implementation" = must be designed-out before adapter code is written.
"blocking_before_real_run" = must be proven by tests/guards before any real F06 run.

| risk_id | risk | severity | likelihood | mitigation | blocking_before_implementation | blocking_before_real_run |
|---|---|---|---|---|---|---|
| R01 | Leakage of 2025/2026 into train data | CRITICAL | MED | Pre-load temporal guard in adapter loader; assert months ⊆ 5 train months; manifest `allow_2025/2026:false`; reject on any 2025/2026 date in trades/ranking | YES | YES |
| R02 | Accidental validation evaluation | CRITICAL | MED | Static import allowlist (adapter MUST NOT import `validation.py`); config `validation_enabled:false` asserted; `no_validation_columns` enforced in RANKING.csv | YES | YES |
| R03 | Accidental holdout/WFA/OOS use | CRITICAL | MED | Forbid importing `wfa.py`/OOS paths; assert `holdout_enabled:false`; manifest `holdout_touched:false` | YES | YES |
| R04 | Dependency on old/quarantined outputs | HIGH | MED | Forbidden-token guard (`QUARANTINED`,`DO_NOT_USE`,`v50b_limited...`,`V50B_RERUN_*`); inputs hashed & referenced in manifest | YES | YES |
| R05 | Partial/!atomic output publish | HIGH | MED | Build in temp dir; contract-validate; single atomic rename; cleanup temp on failure; final dir only via rename | YES | YES |
| R06 | Cost model mismatch (missing spread/slippage/commission) | HIGH | MED | Pin cost config; assert all three from per-`Position` fields; `COST_REPORT.json` must assert spread+slippage+commission_round_turn_usd | YES | YES |
| R07 | Session/timezone mismatch (not NY 07:00–17:00) | HIGH | MED | Adapter injects & asserts NY 07:00–17:00 via params/SessionConfig/session_cutoff; record in manifest | YES | YES |
| R08 | `max_trades_per_day==3` not enforced (engine has no cap) | HIGH | HIGH | Adapter enforces & asserts the 3/day cap post-signal; test coverage; manifest records cap | YES | YES |
| R09 | Engine API ambiguity (F06 family selection via `strategy_module`) | HIGH | HIGH | Dedicated read-only F06 discovery pass to pin the exact F06 module(s); adapter refuses any other family | YES | YES |
| R10 | Data loader ambiguity (`prepared_m5_bid` vs dukascopy precision) | MED | HIGH | Discovery pass pins the canonical train loader & data source; record `data_source_used` + dataset sha/reference | YES | YES |
| R11 | Manifest incomplete (missing engine/adapter/config hash) | HIGH | MED | Mandatory manifest field list (spec §8); validator rejects incomplete manifest | YES | YES |
| R12 | Validator bypass (results read before validation) | CRITICAL | MED | Mandatory `validate_rebuild_outputs` must print READY_FOR_CLAUDE_AUDIT before any interpretation; fail-closed otherwise | YES | YES |
| R13 | Ranking degeneracy (too few strategies / trivial ranking) | MED | MED | Sample-size + family floors; sanity checks on ranking; documented | NO | YES |
| R14 | Sample size insufficient (<100/family, <10/month report) | HIGH | MED | Post-run count gate; fail-closed below floor | NO | YES |
| R15 | Path traversal in output/emit | HIGH | LOW | Reuse PR #6 hardened guards (`assert_no_forbidden_path_tokens`, commonpath containment, allowed subtree, atomic reserve) | YES | YES |
| R16 | Repo hygiene: historical ZIP/locks/backups/canonical copies | MED | MED | Adapter binds only to canonical `research_lab/engine.py`; forbid `*_BACKUP_*`, `reports/canonical_*`, `000_PARA_CHATGPT.zip`; no ZIP delivery | YES | NO |
| R17 | Raw/tick/parquet data mutation | CRITICAL | LOW | Loader opens data read-only; never writes under data dirs; manifest `raw_data_mutated:false`; test asserts no writes outside allowed subtree | YES | YES |
| R18 | Non-determinism (seed/mode/news drift) | HIGH | MED | Pin `DEFAULT_SEED`, execution_mode, cost_profile, intrabar_policy, news settings; record all in manifest; reproducibility test | YES | YES |
| R19 | Scope escalation (adapter triggers real run/optimization/sweep) | CRITICAL | LOW | Adapter does only one F06 train-only run; no optimization/sweep code path; implementation ≠ real run (separate gate) | YES | YES |
| R20 | Engine writes to its native results dir / CWD pollution | MED | MED | Force engine outputs through adapter into Phase 3 subtree; never use `results/research_lab_robust`; cwd-independent paths | YES | NO |

## Top Blocking Risks (must be neutralized in design before implementation)
R01 (2025/2026 leakage), R02 (validation), R03 (holdout/WFA), R09 (F06 selection ambiguity), R12 (validator bypass), R19 (scope escalation), R17 (raw data mutation).

## Notes
- R09 and R10 are the primary reasons IMPLEMENTATION is not authorized yet: they require a dedicated read-only F06 engine-discovery pass (no execution) before the adapter can bind safely.
- All "blocking_before_implementation: YES" rows must have a corresponding designed guard in SAFE_ENGINE_ADAPTER_DESIGN_SPEC.md and a planned test in its Section 11.
