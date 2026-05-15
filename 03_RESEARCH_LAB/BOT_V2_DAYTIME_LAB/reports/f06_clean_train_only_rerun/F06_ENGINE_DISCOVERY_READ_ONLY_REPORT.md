# F06 ENGINE DISCOVERY READ-ONLY REPORT

Generated: 2026-05-15 — READ-ONLY. No engine executed, no backtest, no adapter code, no data mutated.
PR #6 head: 4d60109aec034907ef264cf22c017c0aa37e9863

## 1. Status
F06_ENGINE_DISCOVERY_PARTIAL_NEEDS_FIX

## 2. Executive Summary
The mechanical engine surface is now pinned (entrypoint, frame schema, EngineConfig fields, loader
module, validator + output schemas). However discovery uncovered **decisive, governance-level
blockers** that code discovery alone cannot resolve:

1. **"F06" does not exist in the engine.** `rg "\bF06\b"` over `research_lab/` returns zero matches.
   F06 appears ONLY inside the PR pipeline (configs/fixtures/schemas/docs). `STRATEGY_REGISTRY`
   (`research_lab/strategies/`) is keyed by strategy NAME (e.g. `strategy_vse`), not by family.
   There is no F06→strategy(s) mapping anywhere. The exact F06 strategy module(s) cannot be pinned
   from code — this is a human/governance definition gap.
2. **The output contract is a multi-config family ranking; the engine is a single-run backtest.**
   `RANKING.csv` requires rows per `(family_id, config_id)` with `N_train/PF_train/Total_R_train/
   WR_train`; `TRADES.csv` requires `family_id, config_id`. `run_backtest` runs ONE strategy with
   ONE params dict. Producing a ranked multi-config table is conceptually a parameter
   sweep/screening — which the standing rules explicitly FORBID. This is a contract-vs-mandate
   contradiction that must be resolved by governance, not by the adapter.
3. **Engine trade schema ≠ ledger schema.** Engine trades expose `pnl_usd, pnl_r, session_date,
   exit_price, exit_reason, lots, ...`; the ledger requires `run_id, family_id, config_id,
   signal_time, side, gross_r, sl_pips, net_r, month`. No mapping is defined.

Conclusion: NOT ready for adapter implementation. Next step is a governance/spec decision +
continuation discovery, NOT code.

## 3. Governance Confirmation
- head/origin: `4d60109aec034907ef264cf22c017c0aa37e9863`; branch `research/f06-clean-train-only-rerun-20260515`; not main; no tracked changes; no python processes.
- adapter implemented: NO · adapters/ created: NO · F06 real: NO · backtest: NO · validation/holdout/2025/2026: NO · F06 certified: NO.

## 4. F06 Module Discovery — **F06_MODULE_NOT_FOUND**
| candidate_path | tracked? | safe_to_use? | reason | entrypoint | required_params | blockers |
|---|---|---|---|---|---|---|
| `research_lab/strategies/STRATEGY_REGISTRY` (via `research_lab/strategies/__init__.py`) | YES | mechanism only | registry maps strategy NAME→module; engine calls `module.generate_signal(frame,i,params)` w/ `module.NAME`,`module.WARMUP_BARS` | `generate_signal(frame,i,params)` | per-strategy params | registry has NO "F06" key; F06 is a family label absent from engine |
| any `research_lab/strategies/*.py` (~100 modules) | YES | UNKNOWN | none is named/declared F06; no family tag | `generate_signal` | varies | which module(s) = F06 is undefined |
| `research_lab/validation.py`, `wfa.py` | YES | NO (FORBIDDEN) | validation/holdout surface | — | — | forbidden |
| `reports/canonical_*/research_lab/**`, `*_BACKUP_*` | mixed | NO (FORBIDDEN) | stale snapshots/backups | — | — | not source of truth |
Decision: **F06_MODULE_NOT_FOUND** — requires governance to define F06 → exact strategy module(s) + config taxonomy.

## 5. Train Loader Discovery — **TRAIN_LOADER_NEEDS_DESIGN**
| item | finding |
|---|---|
| loader | `research_lab/data_loader.py::load_backtest_data_bundle(...)` → `BacktestDataBundle`; `load_prepared_ohlcv(pair,data_dirs,timeframe)` reads CSV `pd.read_csv(index_col=0)`, merges, `validate_price_frame` |
| index/timezone | `parse_prepared_index`: `to_datetime(utc=True, ISO8601).tz_convert(NY_TZ)` → tz-aware **America/New_York** |
| data source | prepared OHLCV CSV (per-pair canonical dirs `config.canonical_prepared_data_dirs`); high-precision package optional |
| date filtering | **NO built-in train-month / 2025-2026 filter in the loader.** `research_lab/main.py` defaults `--end 2025-12-31` (leakage default) and uses WFA/OOS (`evaluate_strategy`→`wfa_res.oos_stats`) — main.py path is FORBIDDEN |
| atr14 / range_atr | producer NOT pinned (loader exposes ema/macd/rsi helpers; atr14/range_atr origin not located) |
Decision: loader exists and is usable in principle, but the adapter must (a) bypass `main.py`/WFA, (b) implement an explicit pre-load train-month filter + 2025/2026 hard block, (c) pin the atr14/range_atr producer. **TRAIN_LOADER_NEEDS_DESIGN**.

## 6. EngineConfig Mapping — **ENGINECONFIG_MAPPING_NEEDS_FIX**
`research_lab/config.py` `@dataclass(frozen=True) EngineConfig` (l.180-213).
| EngineConfig field | default (config.py) | Phase 3 required | risk if default used | adapter enforcement |
|---|---|---|---|---|
| pair | EURUSD | EURUSD | none | assert |
| risk_pct | 0.5 | pin explicitly | silent default | YES |
| max_trades_per_day | **2** (l.206; enforced engine.py:979) | **3** | wrong evidence (cap 2≠3) | **YES — must set 3 & assert** |
| max_open_positions | 1 | pin (1) | low | YES |
| assumed_spread_pips | **None** (l.184) → PAIR_META fallback | pin explicit spread OR audited profile | silent spread source | **YES — pin** |
| max_spread_pips | 3.0 | pin | guard divergence | YES |
| commission_per_lot_roundturn_usd | 7.0 | pin (round-turn) | silent default | YES |
| slippage_pips (+ multipliers) | 0.2 (+many) | pin or audited `cost_profile` | silent slippage | YES |
| cost_profile | "auto" | pin explicit (base/precision audited) | non-deterministic cost | YES |
| execution_mode | "normal_mode" | pin | nondeterminism | YES |
| intrabar_policy / intrabar_exit_priority | auto / stop_first | pin | nondeterminism | YES |
| price_source | "bid" | pin | ambiguity | YES |
| session_cutoff | None | pin to clamp to 17:00 NY | wrong session | YES |
| SessionConfig (separate, frozen) | entry 11:00–19:00, force_close 19:00 | NY 07:00–17:00 via `params`→`session_window_from_params` + `session_cutoff` | **wrong session window** | **YES — pin via params+cutoff** |
Note: `run_backtest` builds `SessionConfig()` with hardcoded defaults; the entry window comes from `params` (`session_window_from_params`), NOT from EngineConfig. The adapter must inject the 07:00–17:00 window through `params` and clamp with `session_cutoff`. Decision: **ENGINECONFIG_MAPPING_NEEDS_FIX** (fields known; many silent defaults must be explicitly overridden & asserted).

## 7. Frame Schema — **FRAME_SCHEMA_PINNED (with indicator-provenance gap)**
| aspect | finding |
|---|---|
| index | `pd.DatetimeIndex`, tz-aware; engine converts to NY (`engine.py:612`); loader yields NY (`parse_prepared_index`) |
| required columns | `open, high, low, close, atr14, range_atr` (engine.py:620-625) |
| optional/precision | `precision_package: dict[str,DataFrame]` for `high_precision_mode` (bid/ask); else `prepared_m5_bid` |
| bid/ask | not required in normal mode (bid price source); required only for high-precision mode |
| atr14 / range_atr | REQUIRED by engine; **producer not pinned** (compute step location unknown) |
Decision: schema columns/index/timezone PINNED; the canonical producer of `atr14`/`range_atr` must still be pinned in continuation discovery.

## 8. BacktestResult → Output Mapping — **OUTPUT_MAPPING_BLOCKED**
- Engine `BacktestResult{strategy_name, trades:DataFrame, equity_curve:DataFrame, params, news_filter_used}`; trade rows ~`{exit_price, exit_reason, pnl_usd, pnl_r, lots, session_date, ...}` (engine.py:960-974). No `run_id/family_id/config_id/side/gross_r/sl_pips/net_r/month`.
- Ledger required (`fixtures/output_good/ledger/TRADES_good.csv` + `schemas/ledger_schema.json`): `run_id,family_id,config_id,signal_time,side,gross_r,sl_pips,net_r,month`; single run_id; no 2025/2026; month ∈ exact_months; ≥100 rows/family.
- Ranking required (`RANKING_good.csv` + `schemas/ranking_schema.json`): `family_id,config_id,[parameter_hash,result_signature,]N_train,PF_train,Total_R_train,WR_train`; no validation cols; degeneracy rule (min_unique_ratio 0.5; hard-fail single-unique if configs>1; `deduplicated` flag).
- Cost (`schemas/cost_report_schema.json`): `scenarios`(≥3 each name/spread_pips/slippage_pips/commission_round_turn_usd), `components_applied`(spread/slippage/round_turn all true), `input_ledger_run_id`(≥4), `input_is_quarantined_path:false`.
Blockers: (a) F06 family/config taxonomy undefined; (b) multi-config ranking ≈ forbidden sweep/optimization; (c) no defined transform engine→ledger (gross_r/net_r/sl_pips/month derivation); (d) `config_id`/`parameter_hash`/`result_signature` semantics undefined. Decision: **OUTPUT_MAPPING_BLOCKED**.

## 9. Validator Compatibility — **VALIDATOR_COMPAT_PINNED**
Authoritative validator: `scripts/validate_rebuild_outputs.py` → delegates to `f06_rebuild_pipeline.py::validate_output_dir` → PASS ⇒ `READY_FOR_CLAUDE_AUDIT`, FAIL ⇒ `BLOCKED_GUARD_FAILED` (exit 0/2).
| validator expectation | source | adapter responsibility | impl risk |
|---|---|---|---|
| manifest 29 required fields | `_MANIFEST_REQUIRED` (pipeline l.582-590) incl. `generator_pid, script_path, script_is_tracked, input_is_quarantined_path` | emit ALL | HIGH (spec omitted some) |
| manifest constants | `_MANIFEST_CONST` (script_is_tracked=True, input_is_quarantined_path=False, train_only=True, validation_evaluated=False, holdout_touched=False, allow_2025/2026=False) | exact values | HIGH |
| manifest.status enum | `_STATUS_ENUM = [DRY_RUN_SCHEMA_VALIDATED, BLOCKED_GUARD_FAILED, READY_FOR_CLEAN_TRAIN_RERUN, NOT_READY]` | use enum only | HIGH (spec proposed non-enum) |
| families/symbol | must == `["F06"]` / `EURUSD` | enforce | MED |
| exact_months | regex `20(20..24)-MM`, no 2025/2026 | enforce 5 months | MED |
| artifact declaration | manifest keys `ledger|ledger_csv|trades|trades_csv`, `ranking|ranking_csv|master_ranking`, `cost_report|cost|cost_report_json` or `output_hashes` | declare keys | HIGH |
| ledger schema | `validate_ledger_schema` + single run_id + run_id==manifest.run_id + no 2025/2026 + sample floor + trade_count==rows | produce exact ledger | HIGH |
| ranking schema | `validate_ranking_schema(train_only)` no validation cols + degeneracy | produce exact ranking | HIGH |
| cost report | `validate_cost_report_schema` + `input_ledger_run_id`==run_id | produce exact cost report | HIGH |
| hashes | `output_hashes` sha256 match disk; script/config sha match | hash all artifacts | MED |
| paths | no quarantined token anywhere in manifest strings | sanitize | MED |
Decision: validator EXPECTATIONS fully pinned; adapter's ability to satisfy is gated by §4/§8 blockers.

## 10. Forbidden Surfaces — **FORBIDDEN_SURFACES_PINNED**
`research_lab/validation.py`, `research_lab/wfa.py`, `research_lab/main.py` WFA/OOS path, `research_lab/*_BACKUP_*.py`, `reports/canonical_*/research_lab/**`, `mt5_demo_executor_lab/**`, any `QUARANTINED`/`DO_NOT_USE` path, `v50b_limited_real_gauntlet_rerun_sw`, `V50B_RERUN_TRADES.csv`, `V50B_RERUN_MASTER_RANKING.csv`, any 2025/2026 data, `--end 2025-12-31` default, non-EURUSD, non-F06, `000_PARA_CHATGPT.zip`, engine native `results/research_lab_robust`.

## 11. Blocking Ambiguities Remaining
| ambiguity | severity | blocks_adapter_implementation | required_fix |
|---|---|---|---|
| F06 = which strategy module(s)? (absent from engine) | CRITICAL | YES | Governance defines F06→exact module(s) + import |
| F06 config taxonomy (CFG_001…, parameter_hash, result_signature) | CRITICAL | YES | Governance defines the config set & semantics |
| Multi-config RANKING vs "no sweep/optimization" rule | CRITICAL | YES | Governance ruling: authorized fixed-config evaluation vs forbidden search; written exception or contract change |
| Engine trades → ledger transform (gross_r/net_r/sl_pips/month) | HIGH | YES | Define exact column derivation |
| Pre-load 2025/2026 + train-month filter (not in loader) | HIGH | YES | Design explicit adapter pre-load guard |
| atr14/range_atr producer not pinned | HIGH | YES | Pin canonical indicator producer |
| EngineConfig silent defaults (max_trades=2, spread None, cost_profile auto, session) | HIGH | YES | Adapter must set & assert all Phase-3 values |
| manifest required-fields/status-enum/artifact-keys vs spec | HIGH | YES | Reconcile spec to validator (this report §9) |

## 12. Adapter Implementation Readiness
**BLOCKED** (governance decisions required: F06 definition, config taxonomy, sweep-vs-contract contradiction). Not implementable; not even resolvable by further pure code discovery alone.

## 13. Safety Verification
- adapter_implemented: NO
- real_f06_run: NO
- backtest_run: NO
- validation_touched: NO
- holdout_touched: NO
- 2025_touched: NO
- 2026_touched: NO
- raw_data_mutated: NO
- old_outputs_used: NO

## 14. Next Step
FIX_DISCOVERY_GAPS — governance/spec decision + continuation discovery (see `NEXT_PROMPT_F06_ENGINE_DISCOVERY_CONTINUATION.md`). NO adapter implementation prompt is created because CRITICAL blockers remain.

## 15. Copy-Paste Summary for ChatGPT
F06 engine discovery (read-only, PR #6 head 4d60109a) = **PARTIAL_NEEDS_FIX**. Pinned: run_backtest entrypoint, frame schema (open/high/low/close/atr14/range_atr, tz-aware NY), full EngineConfig field map, loader (`data_loader.load_backtest_data_bundle`), and ALL validator/output schemas (ledger/ranking/cost JSON + 29 manifest fields + status enum). CRITICAL blockers: (1) "F06" is absent from the engine — no F06→strategy mapping exists (STRATEGY_REGISTRY is name-keyed); (2) output contract requires a multi-config family RANKING while sweep/optimization is forbidden — a governance contradiction; (3) engine trade schema ≠ ledger schema, no transform defined; (4) loader has no train-month/2025-2026 filter (main.py defaults to --end 2025-12-31 + uses WFA). Adapter implementation: **BLOCKED**. NO real F06/backtest/validation/holdout/2025/2026 touched; F06 NOT certified. Next: governance defines F06 module(s)+config taxonomy and rules on the ranking-vs-sweep contradiction, then continuation discovery; only afterwards can an implementation prompt exist.
