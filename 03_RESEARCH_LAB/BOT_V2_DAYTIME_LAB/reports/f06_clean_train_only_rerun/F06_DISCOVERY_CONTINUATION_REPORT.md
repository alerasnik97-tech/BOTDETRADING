# F06 DISCOVERY CONTINUATION REPORT

Generated: 2026-05-15 — READ-ONLY. No engine/F06/backtest run. No adapter code. No core modified.
PR #6 head: e61d36ae4f679e3f6f8dd9952d07496bb917ac2a

## 1. Status
DISCOVERY_CONTINUATION_PARTIAL_NEEDS_OWNER_DECISION

Mechanical surfaces are pinned. Two governance decisions + one engine-field gap keep adapter
implementation BLOCKED.

## 2. Train Loader Design — TRAIN_LOADER_DESIGN_PINNED (adapter wrapper required)
- selected_loader_path: `research_lab/data_loader.py`
- selected_loader_function: `load_prepared_ohlcv(pair, data_dirs, timeframe)` (and/or
  `load_backtest_data_bundle`), then the loader's indicator-enrichment that adds `atr14`/`range_atr`.
- FORBIDDEN: `research_lab/main.py` orchestration (uses `evaluate_strategy`→WFA/OOS and defaults
  `--end 2025-12-31`). The adapter must NOT call main.py.
- allowed_data_roots: `research_lab.config.canonical_prepared_data_dirs("EURUSD")` /
  `PAIR_CANONICAL_DATA_DIRS["EURUSD"]` (prepared OHLCV CSV; read-only).
- forbidden_data_roots: any path with `QUARANTINED`/`DO_NOT_USE`/`v50b_limited`, `data_precision_raw`
  unless explicitly approved, news obsolete files, any 2025/2026 partition.
- exact_month_filter_plan: after load, slice index to EXACTLY the 5 train months
  (2020-03, 2021-08, 2022-05, 2023-01, 2024-04); reject if any other month present.
- 2025/2026_guard_plan: hard assert (fail-closed) that `frame.index.year` ∈ {2020..2024} AND month
  set ⊆ the 5 train months, BEFORE `run_backtest`; abort `BLOCKED_TEST_LEAKAGE_RISK` otherwise.
- required_frame_schema: tz-aware DatetimeIndex America/New_York; columns `open, high, low, close,
  atr14, range_atr` (engine.py:620-625); precision package only if high_precision_mode.
- validation_checks: `validate_price_frame` + adapter-side schema/month/tz asserts; assert
  `atr14`/`range_atr` present and non-degenerate.
- blockers: the canonical EURUSD prepared-data dirs must be confirmed to contain train-range data
  and must be filtered strictly (loader has NO built-in date filter).

## 3. ATR14 / RANGE_ATR Provenance — ATR_RANGE_PROVENANCE_PINNED
- producer_path: `research_lab/data_loader.py` — `atr()` (l.217-227) + enrichment
  `frame["atr14"] = atr(frame,14)` (l.513), `frame["range_atr"] = bar_range / atr14` (l.515),
  `bar_range = high - low` (l.511).
- formula: `prev_close=close.shift(1)`; `TR=max(high-low, |high-prev_close|, |low-prev_close|)`;
  `ATR = TR.ewm(alpha=1/period, adjust=False, min_periods=period).mean()`.
- causal: **YES** — uses `shift(1)` (prior bar only) + causal EWM; `range_atr` divides current-bar
  range by causal atr14. No future bars.
- rolling_window: EWM(alpha=1/14), min_periods=14 (warmup ⇒ NaN until 14 bars).
- uses_future_bars: NO. timezone_issues: index already NY tz-aware (parse_prepared_index).
- safe_to_use: YES.
- required_test_before_implementation: a causality/no-lookahead regression test (atr14 at bar i
  unchanged when bars > i are mutated) + warmup-NaN handling test.

## 4. EngineConfig Phase 3 Ruling — ENGINECONFIG_PHASE3_RULING_PINNED (1 owner choice)
`research_lab/config.py` `@dataclass(frozen=True) EngineConfig` (l.180-213); `INITIAL_CAPITAL=100000`,
`DEFAULT_SEED=42` are module globals (not EngineConfig fields).
| field | default | Phase3_required | must_override | reason | source |
|---|---|---|---|---|---|
| pair | EURUSD | EURUSD | no | matches | config.py:181 |
| risk_pct | 0.5 | pin explicit (record) | YES | no silent default | :182 |
| max_trades_per_day | **2** | **3** | **YES** | engine enforces (engine.py:979); 2≠3 ⇒ wrong evidence | :206 |
| max_open_positions | 1 | 1 (pin) | YES | determinism | :207 |
| assumed_spread_pips | **None** | explicit pips OR audited `cost_profile` | **YES (owner choice)** | None ⇒ silent PAIR_META fallback | :184 |
| max_spread_pips | 3.0 | pin | YES | guard determinism | :185 |
| commission_per_lot_roundturn_usd | 7.0 | pin (round-turn) | YES | cost completeness | :186 |
| slippage_pips (+ multipliers) | 0.2 (+many) | pin or audited profile | YES | cost determinism | :187-204 |
| cost_profile | "auto" | pin explicit (base/precision audited) | YES | "auto" non-deterministic | :210 |
| execution_mode | "normal_mode" | pin | YES | determinism | :209 |
| intrabar_policy / intrabar_exit_priority | auto / stop_first | pin | YES | determinism | :205,211 |
| price_source | "bid" | pin | YES | ambiguity | :208 |
| session_cutoff | None | set to clamp 17:00 NY | YES | wrong session otherwise | :212 |
| enforce_hard_stop | True | True | no | safety | :213 |
| SessionConfig (separate frozen) | entry 11:00–19:00 | entry 07:00–17:00 via `params`→`session_window_from_params` + `session_cutoff` | YES | engine builds `SessionConfig()` default; entry window comes from `params` | config.py:154-159 / engine.py:596-604 |
| INITIAL_CAPITAL (global) | 100000 | pin & record in manifest | n/a | reproducibility | config.py |
| DEFAULT_SEED (global) | 42 | pin & record | n/a | reproducibility | config.py |
Owner choice: explicit `assumed_spread_pips` value vs an audited `cost_profile` (both satisfy the
cost contract; must be decided & frozen).

## 5. Output Mapping Design — OUTPUT_MAPPING_NEEDS_ENGINE_FIELD_DISCOVERY (blockers)
Engine trade dict (engine.py:960-974) = `{strategy_name, direction, signal_time, signal_price,
entry_time, entry_price, exit_time, exit_price, exit_reason, pnl_usd, pnl_r, lots, session_date}`.
`pnl_r = pnl_usd/risk_usd` and `pnl_usd` is **cost-inclusive** (spread/slippage in exec prices;
commission split entry/exit, deducted) ⇒ `pnl_r` is a NET R.

| ledger_column | source_from_engine | transform | required? | blocker? |
|---|---|---|---|---|
| run_id | adapter | inject same run_id all rows | yes | no |
| family_id | governance | "F06" (owner) | yes | yes (F06 undefined) |
| config_id | governance | `F06_PHASE3_CANONICAL_001` (owner) | yes | yes |
| signal_time | `signal_time` | ISO-8601 | yes | no |
| side | `direction` | map → BUY/SELL (confirm vocab) | yes | low |
| net_r | `pnl_r` | identity (cost-inclusive) | yes | no |
| gross_r | **NOT EMITTED** | engine exposes no pre-cost R | yes | **YES** |
| sl_pips | **NOT in trade dict** | `Position.sl`/`initial_risk_distance` not appended | yes | **YES** |
| month | `signal_time` | `YYYY-MM` | yes | no |
Ranking: adapter computes `N_train`(row count), `PF_train`(Σwin/Σ|loss|), `Total_R_train`(Σnet_r),
`WR_train`(win rate) per (family_id, config_id) — designable once F06/config fixed.
Cost report: `components_applied`{spread,slippage,round_turn}=true; `input_ledger_run_id`=run_id;
`scenarios`≥3 (owner must define the 3 cost scenarios — schema requires ≥3).
Manifest: all 29 `_MANIFEST_REQUIRED` (incl. `generator_pid, script_path, script_is_tracked,
input_is_quarantined_path`); `status`∈`_STATUS_ENUM` (use `READY_FOR_CLEAN_TRAIN_RERUN`);
artifact-declaration keys (`ledger/ledger_csv/trades/trades_csv`, `ranking/...`, `cost_report/...`);
`output_hashes` sha256 == disk; `script_sha256/config_sha256` == disk; add `adapter_sha256`,
`engine_sha256` as extras.
**Blockers:** `gross_r` and `sl_pips` are not produced by the engine trade record; core
modification is FORBIDDEN ⇒ requires an owner/contract decision (derive vs change schema/contract);
plus the F06/config governance gap.

## 6. Validator Compatibility Update — VALIDATOR_COMPAT_PINNED
Authoritative validator: `scripts/validate_rebuild_outputs.py` → `f06_rebuild_pipeline.py::
validate_output_dir` (PASS⇒`READY_FOR_CLAUDE_AUDIT`). Schemas: `schemas/ledger_schema.json`
(required cols incl. `gross_r,sl_pips,net_r,month`; single run_id; ≥100 rows/family; no 2025/2026;
month∈exact_months), `ranking_schema.json` (no validation cols; degeneracy rule), `cost_report_schema.json`
(`scenarios`≥3, `components_applied`, `input_ledger_run_id`), `_MANIFEST_REQUIRED`/`_MANIFEST_CONST`/
`_STATUS_ENUM`. The ledger schema itself REQUIRES `gross_r` and `sl_pips` — reinforcing the §5 blocker:
the contract demands fields the engine does not emit.

## 7. Remaining Blockers
| blocker | severity | blocks_adapter_implementation | required_fix |
|---|---|---|---|
| F06 = which strategy NAME | CRITICAL | YES | owner decision |
| config taxonomy / single vs frozen multi-config ranking | CRITICAL | YES | owner decision |
| `gross_r` not emitted by engine (ledger requires it) | CRITICAL | YES | owner/contract decision (no core mod) |
| `sl_pips` not in engine trade dict (ledger requires it) | CRITICAL | YES | owner/contract decision (no core mod) |
| spread method: explicit pips vs audited cost_profile | HIGH | YES | owner decision |
| cost report ≥3 scenarios definition | HIGH | YES | owner decision |
| EngineConfig silent defaults (max_trades=2 etc.) | HIGH | designed-out | adapter sets & asserts (pinned here) |
| pre-load 2025/2026 + exact-month filter | HIGH | designed-out | adapter wrapper (pinned here) |
| atr14/range_atr provenance/causality | RESOLVED | no | pinned (data_loader.py, causal) |

## 8. Next Step
Owner governance decision round — see `NEXT_PROMPT_F06_GOVERNANCE_OWNER_DECISION.md`. NO adapter
implementation prompt is created (CRITICAL blockers remain). Implementation ≠ real run; both remain
forbidden until blockers close and a fresh Claude audit approves.
