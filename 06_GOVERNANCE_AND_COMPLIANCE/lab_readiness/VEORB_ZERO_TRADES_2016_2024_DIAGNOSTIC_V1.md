# VEORB ZERO TRADES 2016–2024 DIAGNOSTIC V1

> Read-only, lightweight, skeptical diagnostic. No code, data, runner, engine or
> strategy was modified. No backtest, validation, holdout, 2025/2026,
> optimization, sweep or walk-forward was executed. No ZIP. No `git add .`.
> This diagnostic does NOT approve VE-ORB for validation, holdout, demo, real,
> FTMO, incubation, portfolio or champion.

---

## 1. Diagnostic Status

**DIAGNOSTIC_CONFIRMS_WATCHLIST_BUG_INVESTIGATION_ONLY**

The data-coverage / data-corruption / missing-column / reporting hypotheses are
**refuted**. The strategy source contains **no hardcoded date and no
deterministic post-2015 kill switch**. The post-2015-02-02 silence is best
explained by an **extremely restrictive filter conjunction that only aligned
during the January-2015 SNB/ECB ultra-volatility window**, but the *extremeness*
of "literally zero signals for ~9.9 years from regime-relative filters", combined
with an O(N²) opening-range scan and a median-cadence inference that both operate
on the full growing multi-year frame, leaves a credible runtime
implementation-interaction question that **static reading alone cannot close**.

## 2. Executive Verdict

VE-ORB is **economically dead and statistically void** (15 trades / 10 years,
all inside 2015-01-06 → 2015-02-02; self-flagged `insufficient_sample`). The
run was technically faithful: the data vault contains complete, uncorrupted
EURUSD data for **every month 2015-01 → 2024-12** (274M ticks, M5 = 729,382
bars), and the equity curve (728,997 rows, 2015→2024) proves the engine
iterated the **entire** train timeline. The strategy simply produced **zero
qualifying setups after 2015-02-02**.

The strategy code (`ve_orb_volatility_expansion.py`, 224 lines) is clean: no
year/date hardcode, no off-switch, all gates are *regime-relative* (rolling ATR
percentile, OR width normalised by ATR). That makes a clean "confirmed bug"
declaration unsupported — **but** a regime-relative p65 ATR filter yielding
exactly zero signals for ~2,470 trading days is implausible enough, and the
`_opening_range` / `_infer_cadence_minutes` full-prefix logic suspicious enough,
that "pure regime obsolescence" cannot be cleanly certified either. Verdict:
**insufficient evidence to confirm a bug; insufficient evidence to certify pure
regime death; strategy non-viable regardless.** Stays
`WATCHLIST_BUG_INVESTIGATION_ONLY`.

## 3. Scope

| Item | Value |
|---|---|
| run_id (output token) | `VEORB_OFFICIAL_RUNNER_RUN_2015_2024_20260517_165407` |
| run_id (manifest) | `ve_orb_volatility_expansion_FORMAL` |
| base branch | `audit/veorb-regenerated-dossier-v1-20260517` @ `7bcb6387a64a7a2fc534abd1579e0df66aeffa5c` |
| diagnostic branch | `diagnostic/veorb-zero-trades-2016-2024-20260517` |
| strategy code | `03_RESEARCH_LAB/research_lab/strategies/ve_orb_volatility_expansion.py` (read-only) |
| data manifest | `05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared/prepared_data_manifest.json` (read-only) |
| trades/equity | `.../VEORB_OFFICIAL_RUNNER_RUN_..._165407/local_outputs_do_not_commit/{base,conservative,stress}/` (read-only) |
| artifacts | RUN_MANIFEST.json, 3× ENGINE_CONFIG, 3× summary.json, yearly/monthly, cost summary, post-run report, external audit V1 (carried from prior audit; re-confirmed) |
| **no rerun** | **Confirmed.** No backtest / runner `--execute` / validation / holdout / optimization / sweep was run. Only `git`, `Read`, `Glob`, and read-only `wc/head/tail/cut/sort/uniq/awk` on local files. |

## 4. Safety Verification

| Control | Result |
|---|---|
| code modified? | **NO** |
| data modified? | **NO** |
| backtest run? | **NO** |
| validation run? | **NO** |
| holdout used? | **NO** |
| 2025/2026 used? | **NO** (manifest explicitly excludes 16 2025/2026 source files; `excluded_2025_2026_by_filename: true`) |
| optimization / sweep / walk-forward? | **NO** |
| heavy outputs committed? | **NO** (trades/equity git-ignored via `*_DO_NOT_COMMIT*`; only this report staged) |
| `git add .` used? | **NO** (explicit single-file add) |
| force push / merge / rebase / reset --hard / clean / stash? | **NO** |
| preexisting dirty tree touched? | **NO** (unrelated `external_research_20260516/` files left untouched & unstaged) |

## 5. Activity Confirmation

| Profile | Trades | First entry (NY) | Last entry (NY) | Years with trades |
|---|---|---|---|---|
| base | 15 | 2015-01-06 08:00:00 | 2015-02-02 11:35:00 | 2015 only |
| conservative | 15 | 2015-01-06 08:00:00 | 2015-02-02 11:35:00 | 2015 only |
| stress | 15 | 2015-01-06 08:00:00 | 2015-02-02 11:35:00 | 2015 only |

- Trades by year: **15 in 2015, 0 in 2016–2024.** Monthly split (from prior
  audit, reconciled): 2015-01 = 14, 2015-02 = 1. Last signal **2015-02-02**.
- Profile consistency: all three profiles share the **identical signal set**
  (same timestamps/directions); only PnL differs by cost profile → cost/spread
  does not gate signal selection. No duplicates, no out-of-range timestamps.
- `zero_trade_period`: **2015-02-03 → 2024-12-31 (~118.9 months, ~2,470 trading
  days, 0 trades).**

## 6. Equity / Data Coverage Confirmation

- **Equity curve (base):** 728,997 rows; first `2015-01-02 07:00:00`
  (equity 100000.0); last `2024-12-31 17:00:00` (equity 100137.55); distinct
  years = **2015,2016,2017,2018,2019,2020,2021,2022,2023,2024**.
- **Prepared data manifest** (`status: BUILT_OK`): included tick files for
  **every month 2015-01 → 2024-12** (120 monthly files, `included_min_period
  2015-01`, `included_max_period 2024-12`); `raw_rows_kept 274,002,003`;
  `min_raw_timestamp_utc 2015-01-01 22:00:01`, `max_raw_timestamp_utc
  2024-12-31 21:59:58`. Prepared **M5 = 729,382 rows** (2015-01-01 →
  2024-12-31), M15 = 243,169, M1 = 3,634,609, H1 = 60,800; schema
  `[open,high,low,close,volume]` present for all; `price_synthesized: false`;
  `empty_bars_fabricated: false`; `gaps_forward_filled: false`;
  `excluded_2025_2026_by_filename: true`.
- Equity rows (728,997) ≈ prepared M5 bars (729,382), Δ≈385 (~0.05%, consistent
  with warmup/edge trimming) → the engine fed and iterated the **full M5 series
  across all years**.

→ **`EVIDENCE_PIPELINE_PROCESSED_FULL_TRAIN_BUT_STRATEGY_DID_NOT_TRIGGER`**
→ **`PASS_DATA_AFTER_2015_EXISTS`** (data for 2016–2024 exists, complete,
uncorrupted). No `BLOCKER_EQUITY_COVERAGE_INCOMPLETE`. No
`BLOCKER_DATA_AFTER_2015_MISSING`.

## 7. Strategy Logic Audit

`ve_orb_volatility_expansion.py` — `NAME=ve_orb_volatility_expansion`,
`WARMUP_BARS=220`, `EXPLICIT_TIMEFRAME="M1_OR_M5"`. `DEFAULT_PARAMS` ==
`parameter_set_used` in summary.json (no hidden override). Pure functional
`signal(frame, i, params)`; **no I/O, no date literals, no global state.**

| filter / condition | purpose | evidence (line) | could block 2016–2024? | severity | note |
|---|---|---|---|---|---|
| OHLC columns present & `i>1` | input guard | 172–174 | NO (schema constant all years) | PASS | data manifest schema constant |
| time gate `or_end ≤ minute < entry_end` (08:00–12:00) | restrict entries to session | 176–178 | NO by itself (fires every trading day; NY-local, DST-safe) | LOW | matches observed `entry_time_ny` 08:00–11:35 |
| warmup `i < lookback+atr_period` (214) | ATR history | 182–183 | NO (consumed in 1st Jan-2015 day; cliff is Feb) | LOW | `WARMUP_BARS=220` aligned |
| ATR-percentile `current_atr > p65(last 200 ATR)` | volatility expansion | 185–192 | **NO as cliff cause** — rolling/relative, ~35% pass every regime | LOW (cliff) / MED (selectivity) | does not decay over time |
| OR build `idx.date()==ts.date()` & 07:00–08:00, `frame.iloc[:i]` scan | opening range | 120–138 | possible *runtime* interaction (O(N²) full-prefix scan) | LOW–MED | correctness OK statically; perf pathological |
| OR completeness `≥ ceil(12·0.9)=11` unique bars, first ≤07:05, last ≥07:55 | OR integrity | 87–117 | unlikely (normal M5 days satisfy all years) | LOW–MED | depends on cadence inference |
| cadence `_infer_cadence_minutes` median of **all** prefix deltas ∈{1,2,3,5,10,15} | timeframe inference | 73–84 | possible *runtime* interaction on growing multi-year frame | LOW–MED | static: median stays 5.0 (robust to 789 gaps) |
| OR-width band `0.40 ≤ (or_high−or_low)/ATR ≤ 3.00` | filter chop / blowoff | 198–200 | NO alone (regime-relative) | MED | part of degenerate conjunction |
| breakout `prev_close ≤ or_high < close` / `≥ or_low > close` | entry trigger | 207–222 | NO alone (regime-relative) | MED–HIGH | strict 2-close breakout; rare in conjunction |
| spread / cost / liquidity guard | — | **absent in strategy** | N/A | PASS | costs applied by engine post-selection |
| date / year hardcode | — | **none found** | N/A | PASS | no off-switch anywhere |

`suspected_primary_cause`: **degenerate selectivity** — the *conjunction*
`[ATR>p65] ∧ [OR-width 0.4–3.0×ATR] ∧ [strict 2-close OR breakout] ∧ [≥11/12 OR
bars] ∧ [08:00–12:00 window]` aligned during the Jan-2015 SNB-floor-removal
(2015-01-15) + ECB-QE (2015-01-22) ultra-volatility regime and effectively never
recurred — **with a residual, unresolved suspicion** on the full-prefix
`_opening_range`/`_infer_cadence_minutes` runtime behaviour.

## 8. Failure Mode Analysis (A–J)

| # | Hypothesis | Evidence for | Evidence against | Missing | Likelihood |
|---|---|---|---|---|---|
| A | Data issue (post-2015 missing/corrupt) | none | manifest BUILT_OK; 120 monthly files 2015-2024; M5 729,382 rows; equity all years | — | **LOW (refuted)** |
| B | Timestamp/timezone/session bug | none static | NY-local window stable & DST-safe; trades at expected NY times; no date hardcode | runtime index the engine passes signal() | **LOW–MED** |
| C | Missing/changed columns | none | schema `[o,h,l,c,v]` constant all years; `required_columns_present:true` | — | **LOW (refuted)** |
| D | Spread/cost guard kills later years | none | no spread/cost logic in strategy; identical signals across cost profiles | — | **LOW (refuted)** |
| E | ATR-percentile too restrictive | tight gate | rolling/relative → ~35% pass every regime; cannot cause a 2015-only cliff | — | **LOW (cliff) / MED (selectivity)** |
| F | OR-coverage too restrictive | full-prefix cadence/coverage logic | normal M5 days meet 11/12 across all years; median cadence stays 5.0 | runtime cadence on engine-fed frame | **LOW–MED** |
| G | Breakout/expansion too restrictive | strict 2-close breakout + OR-width band | regime-relative; should occasionally fire any year | — | **MED (in conjunction)** |
| H | Correct impl, regime disappeared | trades cluster exactly on SNB/ECB Jan-2015 shock; all gates regime-relative; data complete; no kill switch | a relative-filter set yielding 0 signals for ~9.9 yrs is extreme | counterfactual signal rate post-2015 (would need run) | **MED–HIGH (dominant economic story, not cleanly certifiable)** |
| I | Summary/reporting issue | none | summary↔yearly↔monthly↔trades↔equity reconcile to float/cent (prior audit) | — | **LOW (refuted)** |
| J | Evidence insufficient to split H vs runtime impl interaction | O(N²) `_opening_range` + median-cadence over growing multi-year frame; unconfirmed runtime tz/index | static code has no deterministic off-switch | read-only engine/runner data-feed + signal-call review | **HIGH** |

## 9. Bug vs Regime Obsolescence Verdict

- **Does it look like a bug?** Not a *data/column/reporting/spread* bug — those
  are refuted. No deterministic off-switch or date hardcode in the strategy. A
  *runtime implementation interaction* (O(N²) full-prefix `_opening_range`
  scan + median-cadence inference over the growing multi-year frame, plus the
  unverified engine-fed index/timezone) **remains possible but is not
  demonstrable from static reading**.
- **Does it look like a data issue?** **No.** Data is complete and uncorrupted
  for 2016–2024; the engine processed the full timeline.
- **Does it look like an over-restrictive filter?** **Yes — primary explanation.**
  The filter *conjunction* is so tight it effectively only fired in the
  Jan-2015 SNB/ECB extreme-volatility regime.
- **Does it look like regime obsolescence?** **Plausible and dominant**, but a
  regime-*relative* filter set producing exactly **zero** signals for ~2,470
  trading days is extreme enough that "pure regime death" cannot be cleanly
  certified without ruling out the runtime implementation interaction.
- **What evidence is missing?** The counterfactual signal rate 2016–2024 and the
  exact frame/index/cadence the engine passes `signal()` — obtainable only via a
  separate **read-only** engine/runner data-feed + signal-invocation review
  (NOT a rerun, NOT validation, NOT holdout). Out of scope for this light pass.

**Net:** insufficient evidence to declare a confirmed bug; insufficient evidence
to cleanly certify pure regime obsolescence. **Either way the strategy is
non-viable.**

## 10. Institutional Decision

Applying the Block-8 no-go rule (15 trades / 1 active month / 0 trades
2016–2024 / marginal PF / irrelevant return):

- **NO validation.**
- **NO holdout / sealed_holdout.**
- **NO demo / real / FTMO.**
- **NO champion.**
- **NO portfolio candidate.**
- **NO incubation.**
- **NO edge approved. NOT profitable. NOT robust.**

Status: **`DIAGNOSTIC_CONFIRMS_WATCHLIST_BUG_INVESTIGATION_ONLY`**.

## 11. Allowed Next Step

**D) Strong-enough implementation-interaction question → propose a SEPARATE
future, strictly read-only follow-up; do NOT implement it now.**

Proposed (not executed) future read-only scope: review the engine/runner
data-feed and the exact `signal(frame, i, params)` invocation contract — the
frame index timezone, the cadence actually inferred over the multi-year frame,
and whether the O(N²) `_opening_range` full-prefix scan / any iteration bound
interacts with post-2015 signal generation. **No rerun, no `--execute`, no
validation, no holdout, no optimization/sweep.** If that review confirms the
data is fed correctly and the filter genuinely finds no setups by correct
design → reclassify to `VEORB_PRELIMINARY_REJECTED_LOW_EDGE_REGIME_OBSOLETE`
and archive. Option A (immediate archive/reject as low-edge) is also acceptable
if the lab declines further effort — the strategy is non-viable under either.

## 12. Findings Table

| id | severity | category | finding | evidence | implication | required_action |
|---|---|---|---|---|---|---|
| D1 | PASS | data-coverage | Complete uncorrupted EURUSD data 2015-01→2024-12 | prepared_data_manifest.json (120 monthly files, M5 729,382 rows, BUILT_OK) | Not a data issue | None |
| D2 | PASS | pipeline | Engine iterated full train timeline | equity 728,997 rows, 2015→2024, ≈ M5 bars | Run faithful; refutes truncation | None |
| D3 | PASS | strategy-source | No date/year hardcode, no off-switch | full read of strategy (224 lines) | No deterministic post-2015 kill switch | None |
| D4 | PASS | reporting | Activity consistent across all artifacts | trades/yearly/monthly/summary/equity reconcile | Not a reporting artifact | None |
| D5 | PASS | scope-safety | No 2025/2026; train-only | manifest excludes 16 2025/2026 files | No leakage | None |
| D6 | WARN | strategy-design | Filter conjunction degenerately selective (E+F+G+H) | 15 trades clustered on SNB/ECB Jan-2015 shock | Strategy economically dead | Reject as investable |
| D7 | WARN | implementation | `_opening_range` O(N²) full-prefix scan + median-cadence over growing multi-year frame | code lines 73–138 | Possible runtime interaction; unconfirmable statically | Propose separate read-only engine review (D) |
| D8 | WARN | sample/edge | n=15, `insufficient_sample`, PF within noise, ~0.014%/yr | summaries (prior audit) | No edge; statistically void | No validation/holdout |
| D9 | WARN | traceability | summary.timeframe=M15 vs strategy `M1_OR_M5` / trades `prepared_m5_bid` | summary vs strategy vs trades | Minor labeling inconsistency | Document timeframe inference |
| D10 | WARN | repo-hygiene (out of scope) | preexisting unrelated dirty tree under `external_research_20260516/` | `git status` | Not caused by this work; untouched | Repo owner only |
| — | BLOCKER | — | **NONE** | — | — | — |

## 13. Final Institutional Verdict

VE-ORB is dead as a tradeable strategy: 15 trades in a decade, all in a single
4-week SNB/ECB shock window of 2015, self-flagged insufficient sample, return
economically nil. The run was faithful — data is complete 2015–2024 and the
engine traversed the entire timeline. The strategy code has no date hardcode
and no off-switch; the silence is driven by a degenerately selective filter
conjunction (most likely genuine extreme regime selectivity). Because a
regime-*relative* filter set producing zero signals for ~9.9 years is extreme,
and because the `_opening_range`/cadence logic runs O(N²) over the full
multi-year frame, a residual runtime implementation-interaction question remains
that static reading cannot close. Evidence is insufficient to confirm a bug and
insufficient to cleanly certify pure regime death. Decision:
`DIAGNOSTIC_CONFIRMS_WATCHLIST_BUG_INVESTIGATION_ONLY`. No validation, no
holdout, no production, no incubation, no edge. Permitted forward motion is at
most one **separate, future, read-only** engine/runner data-feed review — or
outright archival. Do not spend further effort trying to save this strategy.

---

*Read-only diagnostic. No code/data/runner/engine/strategy modification, no
rerun, no validation/holdout/2025-2026, no optimization/sweep, no ZIP, no heavy
output committed, no `git add .`, no destructive git.*
