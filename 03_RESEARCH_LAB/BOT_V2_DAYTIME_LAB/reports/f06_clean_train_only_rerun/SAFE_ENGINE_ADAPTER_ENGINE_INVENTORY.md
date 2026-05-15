# SAFE ENGINE ADAPTER — ENGINE INVENTORY

Generated: 2026-05-15 — READ-ONLY inventory. Nothing executed. No engine code modified.
PR #6 head: d988c03ad7a60c0080a767baf740161346b1222c
Constraints honored: no adapter implementation, no F06 real run, no backtest, no validation/holdout, no 2025/2026, no core modification.

## 1. Status
ENGINE_INVENTORY_COMPLETE_READ_ONLY

Correction vs. prior docs: the engine surface is NOT inside the PR #6 pipeline tree. It is a **repo-root tracked package** `research_lab/` (209 tracked files). Earlier `PHASE3_ENGINE_INTERFACE_AUDIT.md` correctly named `research_lab/engine.py` / `research_lab.engine.run_backtest(...)` / `research_lab/config.py`; this inventory grounds that with the real signature and surfaces.

## 2. Files Inspected (read-only)
| path | purpose | safe_to_use | reason |
|---|---|---|---|
| `research_lab/engine.py` (998 lines, tracked, last commit cefdd8db) | Bar/precision backtest execution engine; public `run_backtest(...)` | YES (as design target only) | Canonical engine surface; tracked in this repo |
| `research_lab/config.py` (322 lines, tracked) | `EngineConfig`/`SessionConfig`, cost/data defaults | YES (as design target only) | Canonical config surface |
| `research_lab/engine_BACKUP_20260421_1535.py` | timestamped backup | NO | Backup; not source-of-truth |
| `research_lab/config_BACKUP_20260421_1535.py` | timestamped backup | NO | Backup; not source-of-truth |
| `research_lab/validation.py` | validation/WFA logic | NO (FORBIDDEN) | Validation surface — forbidden for train-only Phase 3 (not opened) |
| `research_lab/wfa.py` | walk-forward analysis | NO (FORBIDDEN) | Holdout/OOS surface — forbidden (not opened) |
| `reports/canonical_*/research_lab/engine.py` (+ copies) | old report snapshots | NO (FORBIDDEN) | Stale snapshot copies; not canonical |
| `BOT_V2_DAYTIME_LAB/src/phase*_engine.py` | legacy phase engines | NO | Superseded; not the Phase 3 target |
| `mt5_demo_executor_lab/mt5_risk_engine.py` | live/demo execution | NO (FORBIDDEN) | Live execution surface; out of scope |
| `pipelines/f06_evidence_rebuild/scripts/f06_rebuild_pipeline.py` | PR #6 fail-closed runner | YES | Caller side of the future adapter |

## 3. Engine Entry Points Found
- **`run_backtest(strategy_module, frame, params, engine_config, news_block, news_filter_used, *, precision_package=None, data_source_used=None, news_events=None, news_settings=None) -> BacktestResult`** — `research_lab/engine.py:573`.
  - input expectations: `frame` is a tz-aware OHLC DataFrame with columns `open, high, low, close, atr14, range_atr`; `strategy_module` carries `NAME`/`WARMUP_BARS`; `params` carries the session window; `engine_config` is an `EngineConfig`.
  - output behavior: returns `BacktestResult{ strategy_name, trades: DataFrame, equity_curve: DataFrame, params: dict, news_filter_used: bool }`. No native manifest/hash/cost-report/ranking.
  - risk: engine does NOT load data and does NOT enforce Phase 3 governance (family, months, per-day cap, output subtree). All governance must live in the adapter.
- Dataclasses: `Position` (engine.py:23, per-trade incl. `entry_commission_usd`, `entry_spread_pips`, `entry_slippage_pips`, `execution_mode_used`, `cost_profile_used`), `BacktestResult` (engine.py:55).

## 4. Data Loading Surface
- `run_backtest` does NOT read disk — the caller supplies `frame` (and optional `precision_package`).
- Config defaults (`config.py`): `DEFAULT_DATA_DIRS`, `DEFAULT_HIGH_PRECISION_RAW_DIR=data_precision_raw/dukascopy`, `DEFAULT_HIGH_PRECISION_PREPARED_DIR=data_precision/dukascopy`, news import dirs, obsolete news files.
- date filtering available: only if the adapter's loader slices to the 5 train months BEFORE building `frame`.
- whether 2025/2026 can be blocked before load: YES — must be enforced by the adapter loader (engine has no such guard).
- risks: leakage if the loader is broad; data-source ambiguity (`prepared_m5_bid` vs `dukascopy_m1_bid_ask_full`).

## 5. Cost Model Surface
- spread: `configured_spread_pips`, `estimate_spread_pips`, `spread_guard_allows`, `actual_spread_pips`; default `DEFAULT_SPREAD_PIPS=1.2`.
- slippage: `estimate_slippage_pips`, `slippage_price`; default `DEFAULT_SLIPPAGE_PIPS=0.2`.
- commission: `DEFAULT_COMMISSION_ROUNDTURN_USD=7.0`; per-trade `Position.entry_commission_usd`.
- Phase 3 enforceable: YES — all three components exist and are recorded per `Position`; the adapter must pin them, assert all three applied, and emit `COST_REPORT.json`.

## 6. Session / Risk Surface
- session: `SessionConfig` (config.py:155), `session_window_from_params(params)`, `session_cost_bucket`, `engine_config.session_cutoff`; timezone default `NY_TZ="America/New_York"` (matches Phase 3 07:00–17:00 NY).
- max trades/day: NO explicit engine-level per-day cap constant; `risk.max_trades_per_day==3` must be enforced by the adapter (via params/post-filter) and asserted — engine will not enforce it.
- timezone assumptions: engine converts the frame index to NY with DST handling.
- risks: if the adapter does not inject/assert the exact NY 07:00–17:00 window and the 3/day cap, the engine will silently use strategy/param defaults.

## 7. Output Surface
- engine emits: `BacktestResult.trades` (DataFrame), `BacktestResult.equity_curve` (DataFrame). Native results dir `DEFAULT_RESULTS_DIR=results/research_lab_robust`; `VISIBLE_CHATGPT_ARCHIVE=000_PARA_CHATGPT.zip`.
- trades output: derivable from `trades`. ranking output: NOT native — adapter must compute `RANKING.csv` (train-only, no validation columns). cost report: NOT native — adapter derives `COST_REPORT.json` from per-`Position` cost fields. manifest/hash: NOT native — adapter must generate `MANIFEST.json` + `HASHES.txt`.
- gaps: engine output ≠ Phase 3 output contract; the adapter is fully responsible for contract production, atomic publish, and the mandatory post-run validator.

## 8. Forbidden Surfaces (must NOT be used by the adapter)
- `research_lab/validation.py`, `research_lab/wfa.py`, any OOS/holdout/WFA path (`DEFAULT_WFA_IS_MONTHS`, `DEFAULT_WFA_OOS_MONTHS`).
- `reports/canonical_*/research_lab/**` snapshot copies; `*_BACKUP_*.py`.
- `mt5_demo_executor_lab/**` (live/demo execution).
- Engine native output dir `results/research_lab_robust`, `000_PARA_CHATGPT.zip`, any path containing `QUARANTINED`, `DO_NOT_USE`, `v50b_limited_real_gauntlet_rerun_sw`, `V50B_RERUN_TRADES.csv`, `V50B_RERUN_MASTER_RANKING.csv`.
- Any 2025/2026 data; any non-EURUSD symbol; any family other than F06.

## 9. Adapter Readiness
NEEDS_MORE_ENGINE_DISCOVERY (for IMPLEMENTATION) / READY_FOR_ADAPTER_DESIGN (for DESIGN-ONLY)

Rationale: the public `run_backtest` interface is concrete enough to design a fail-closed adapter contract NOW. Implementation remains gated because the F06 strategy-module selection path, the canonical train-data loader contract, and the exact `EngineConfig` field set for Phase 3 still require a dedicated read-only discovery pass against `research_lab/` (without executing the engine). Design proceeds; implementation does not.
