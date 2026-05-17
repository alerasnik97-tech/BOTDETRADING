# VEORB REGENERATED DOSSIER EXTERNAL AUDIT V1

> External, read-only, institutional quant audit of the VE-ORB regenerated dossier.
> No code, data, runner, engine, holdout, validation, optimization, sweep, rerun or
> production was touched. No heavy outputs were committed. No `git add .`.
> This audit does NOT approve VE-ORB for real, demo, FTMO, production or incubation.

---

## 1. Audit Status

**AUDIT_PASS_BUT_STRATEGY_WATCHLIST_BUG_INVESTIGATION_ONLY**

The run is technically correct, faithfully executed, fully reconciled and safe.
The **strategy** is rejected as non-investable. It is retained on a strict
*diagnostic-only* watchlist solely to close one open question with a read-only
review (see §11–§15). No BLOCKERS were found.

## 2. Executive Verdict

VE-ORB was executed correctly via the official fail-closed formal runner on
train-only EURUSD data (2015-01-01 → 2024-12-31). Every lightweight artifact
(manifest, three engine configs, three summaries, yearly/monthly tables, cost
profile summary) is internally consistent and reconciles **to full float
precision**, including against the heavy local `trades.csv` and `equity_curve.csv`
(equity endpoints match summary returns to the cent).

However, the strategy is **statistically dead**:

- **15 trades in 10 years.** The producing system itself stamps
  `insufficient_sample: true` and `sample_penalty_applied: true` on all three profiles.
- **All 15 trades fall in a single 4-week window: 2015-01-06 → 2015-02-02.**
  Zero trades for the remaining ~118.9 months of the train window. This is *worse*
  than the declared "0 trades since 2016" — the last signal was **2015-02-02**.
- Profit factor 1.03–1.06 with n=15 is **inside statistical noise**; expectancy
  +0.02–0.04 R is not an edge.
- Total return over the full 10-year window is **0.0199%–0.1376%**
  (~0.002%–0.014% per year) — economically irrelevant.

The data-pipeline bug hypothesis is **refuted**: the equity curve proves the
engine iterated all 728,997 five-minute bars across every year 2015–2024. The
data was present and fully processed; only the **entry logic** produced no
qualifying setup after 2015-02-02. Whether that is genuine degenerate
selectivity / regime death **or** an over-restrictive strategy-logic/parameter
defect cannot be settled by this read-only artifact audit — it requires a
single, bounded, **read-only** code/diagnostic review (no rerun, no validation,
no holdout). Either way the strategy is **not investable** and must **not**
advance to validation or holdout.

The prior classification `VEORB_PRELIMINARY_INTERESTING_NEEDS_AUDIT` is judged
**too optimistic** and is downgraded (see §12).

## 3. Scope Audited

| Item | Value |
|---|---|
| Project | `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo` |
| Base (authorizing) branch | `audit/mr01-regenerated-dossier-v1-20260517` |
| Base (authorizing) commit | `11ea9d1b68d022d8bfb76f7db75bf2214e43ee30` |
| Research branch | `research/veorb-official-runner-run-20260517` |
| Research commit audited | `937376fd0e0d7433a1018b80e241a0de34173175` |
| Audit branch created | `audit/veorb-regenerated-dossier-v1-20260517` |
| Run ID (manifest) | `ve_orb_volatility_expansion_FORMAL` |
| Run ID (output token) | `VEORB_OFFICIAL_RUNNER_RUN_2015_2024_20260517_165407` |
| Output dir | `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/formal_train_only/ve_orb_volatility_expansion/VEORB_OFFICIAL_RUNNER_RUN_2015_2024_20260517_165407` |
| Strategy | `ve_orb_volatility_expansion` |
| Runner | `research_lab.runners.formal_train_runner` |
| Data path | `05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared` |

**Files inspected (read-only):** `manifests/RUN_MANIFEST.json`;
`configs/{base,conservative,stress}_ENGINE_CONFIG.json`;
`profile_reports/{base,conservative,stress}/summary.json`;
`profile_reports/{base,conservative,stress}/tables/{yearly,monthly}.csv`;
`tables/VEORB_COST_PROFILE_SUMMARY.csv`;
`local_outputs_do_not_commit/{base,conservative,stress}/trades.csv` (15 rows each, full read);
`local_outputs_do_not_commit/{base,conservative,stress}/equity_curve.csv`
(row count, header, first/last row, distinct years — light read only);
`06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/VEORB_POST_RUN_RECONCILIATION_REPORT.md`.
Git: `status`, `branch`, `rev-parse`, `fetch --prune`, `log`, `remote`,
`diff`, `show --name-only --stat`, `ls-files`, `check-ignore`.

**What was NOT done (by mandate):** no backtest rerun; no validation; no
holdout / sealed_holdout; no 2025/2026 data; no optimization; no sweep; no grid
search; no walk-forward; no second strategy; no code/data/runner/engine/strategy
modification; no destructive git; no `git add .`; no ZIP; no Explorer; no
production / CORE_PRODUCTION access; heavy `equity_curve.csv` not fully parsed.

## 4. Safety Verification

| Control | Result | Evidence |
|---|---|---|
| main used? | **NO** | Worked on `research/...` then `audit/veorb-regenerated-dossier-v1-20260517` |
| force push? | **NO** | Only a normal `push -u` of the audit branch |
| merge / rebase / reset --hard / clean / stash? | **NO** | None executed |
| `git add .`? | **NO** | Only the audit report (+ findings CSV) staged explicitly |
| code modified? | **NO** | Commit `937376fd` contains 0 `.py`/`.ipynb`; audit branch zero-diff vs `origin/research/...` |
| data modified? | **NO** | No write to `05_MARKET_DATA_VAULT`; no data files in commit |
| runner / engine / strategy / report.py / data_loader / metric_reconciliation / cost_profiles modified? | **NO** | No code paths in commit; nothing edited |
| heavy outputs committed? | **NO** | `trades.csv`/`equity_curve.csv` git-ignored via `.gitignore:121 *_DO_NOT_COMMIT*` (confirmed by `git check-ignore`); absent from commit `937376fd` |
| holdout / sealed_holdout used? | **NO** | `manifest.holdout_used=false`; data path = `prepared_train_2015_2024`; no holdout token in artifacts |
| 2025 / 2026 used? | **NO** | `manifest.max_timestamp=2024-12-31`; trades/equity all ≤2024-12-31; "2025/2026" hits are numeric false-positives inside equity floats |
| validation run? | **NO** | `manifest.validation_run=false` |
| optimization / sweep / walk-forward? | **NO** | `manifest.optimization_run=false`, `sweep_run=false`; no sweep artifacts |
| news / high_precision used? | **NO** | `manifest.news_used=false`, `high_precision_used=false`; `news_filter_used=false`; every trade `blocked_by_news=False`. (See W7: report prose wording defect only.) |
| preexisting dirty tree touched? | **NO** | 6 modified + 5 untracked files under `external_research_20260516/` are unrelated, pre-existing, **left untouched and unstaged** |

## 5. Manifest Audit

`RUN_MANIFEST.json` (PASS on scope, WARN on traceability):

- `strategy = ve_orb_volatility_expansion` ✓; `branch = research/veorb-official-runner-run-20260517` ✓.
- `commit = 11ea9d1b68d022d8bfb76f7db75bf2214e43ee30` = the **authorizing base
  commit** the code ran against (expected; the run predates the artifact commit
  `937376fd`). Provenance is consistent with the declared authorization chain.
- `train_only=true`; `min_timestamp=2015-01-01`; `max_timestamp=2024-12-31` ✓
  (matches declared range).
- `holdout_used=false`, `validation_run=false`, `optimization_run=false`,
  `sweep_run=false`, `news_used=false`, `high_precision_used=false` — all clean
  negative declarations.
- `profiles_run=[base,conservative,stress]` ✓; `reconciliation_required=true`.
- **WARN (W9):** `run_id="ve_orb_volatility_expansion_FORMAL"` does **not** embed
  the timestamped output token `..._165407`; the manifest carries **no
  timestamp** and **no reconciliation result/sealed/exit-code field** (only
  `reconciliation_required: true`). Sufficient to verify scope/safety; **not
  sufficient** to independently verify the formal gate seal or to bind the
  manifest tightly to this specific output folder
  (`FINDING_MANIFEST_INSUFFICIENT_FOR_FULL_AUDIT`, limited to those two points).

## 6. Config Audit

`base/conservative/stress _ENGINE_CONFIG.json`:

- **PASS_CONFIG_SCOPE:** no forbidden-scope fields in any config — no holdout,
  no validation, no 2025/2026, no news, no high_precision, no optimization/sweep.
- `pair=EURUSD`; `max_trades_per_day=2` (≤3 lab limit) ✓; `max_open_positions=1`;
  `max_spread_pips=3.0`; `risk_pct=0.5`; `commission_per_lot_roundturn_usd=7.0`;
  `slippage_pips=0.2`.
- Profiles are genuinely differentiated by `cost_profile`
  (base/conservative/stress) and `execution_mode`
  (`normal_mode`/`conservative_mode`/`stress_mode`); `intrabar_policy` is
  `standard`/`conservative`/`conservative`. The numeric multipliers
  (`conservative_*`=1.2/1.3, `stress_*`=1.35/1.6) are a shared schema in all
  three files; differentiation is **mode-driven**, not per-file numeric.
  Real economic differentiation is **confirmed downstream** by strict monotonic
  metric degradation (§7, §10) and at the per-trade level in `trades.csv`.
  → effective **PASS**; not a cost-profile-not-real / monotonicity blocker.
- **WARN (W10):** configs contain no strategy name, date range or dataset path;
  scope is only verifiable via manifest + summary
  (`WARN_CONFIG_FIELD_MISSING`) — acceptable but reduces config-level auditability.

## 7. Metric Reconciliation Audit

`summary.json` ↔ `VEORB_COST_PROFILE_SUMMARY.csv` ↔ declared snapshot:
**identical to full float precision** for all three profiles
(`PASS_METRIC_RECONCILIATION`).

| Profile | Trades | PF | Expectancy R | Return % | Max DD % | Win % |
|---|---|---|---|---|---|---|
| base | 15 | 1.0620173301588234 | 0.036140338562152145 | 0.1375515937120264 | 1.5212770745123854 | 46.666… |
| conservative | 15 | 1.0450107231486905 | 0.02724755552201962 | 0.07340633393906337 | 1.5506737306529248 | 46.666… |
| stress | 15 | 1.0308440125884266 | 0.019830442289099158 | 0.019876606772384342 | 1.5757416842959064 | 46.666… |

- Base is the best, stress the worst, on PF / expectancy / return; max DD worsens
  base→stress. No metric inconsistency, no impossible metric, no unexplained
  contradiction at the machine-artifact level.
- Identical trade count and win rate across profiles is **logically explained**:
  costs change R magnitude, not the win/loss classification of these fixed
  2R-target / 1R-stop trades (confirmed trade-by-trade in `trades.csv`).
- **WARN_LOW_SAMPLE_SIZE / WARN_LOW_FREQUENCY / WARN_MARGINAL_EDGE:** n=15;
  `insufficient_sample=true` and `sample_penalty_applied=true` on all three
  profiles (the system self-rejects the sample); `avg_trades_per_month=0.125`;
  PF inside noise; return economically irrelevant.

## 8. Yearly/Monthly Consistency Audit

- `yearly.csv` (all 3 profiles) contains **exactly one row: 2015**
  (15 trades, 7 W / 8 L, 0 breakevens, win 46.67%). Years 2016–2024 are
  **absent entirely** (not even zero-filled).
- `monthly.csv` (all 3 profiles) contains **exactly two rows**: `2015-01`
  (14 trades, 6 W / 8 L) and `2015-02` (1 trade, 1 W, PF=`inf`).
- Aggregation ties out exactly: 14 + 1 = 15 trades; 7 W / 8 L; monthly
  `total_pnl_r`/`total_pnl_usd` sum to the yearly row and to `summary.json`
  for every profile. **No `BLOCKER_METRIC_INCONSISTENCY`.**
- **WARN (W3):** `positive_years=1 / negative_years=0` is **misleading** —
  there is exactly **one active year**; "0 negative years" is vacuous because
  2016–2024 had no trades. It must not be read as robustness/consistency.

## 9. Trades / Escalation Audit

Local `trades.csv` **were read** (read-only; not staged; not modified;
git-ignored). 15 data rows per profile.

- **Entry dates (all profiles, identical set):** 2015-01-06, -01-08, -01-09
  (×2), -01-12, -01-16 (×2), -01-19, -01-20 (×2), -01-21, -01-22, -01-27,
  -01-29, **2015-02-02**. → `CONFIRMED_ACTIVITY_ONLY_2015`; precisely
  **2015-01-06 → 2015-02-02 only**. Last signal **2015-02-02**.
- Per-day count never exceeds 2 (consistent with `max_trades_per_day=2` and the
  lab's ≤3/day rule).
- Outcomes per profile: 7 W / 8 L (matches yearly/monthly/summary). Loss legs
  ≈ −1.0 R (hard stop), win legs ≈ +1.99 R (`target_rr=2.0`), with fractional
  R on `forced_session_close` — economically coherent for an ORB design.
- All three profiles share the **same signal set**, differing only in
  cost/execution magnitude; per-trade PnL strictly worsens
  base→conservative→stress (monotonic at trade granularity).
- Structurally valid: realistic 2015 EURUSD prices (~1.13–1.19),
  `data_source_used=prepared_m5_bid`, `blocked_by_news=False`,
  `gap_exit_flag=False`, no duplicates, no out-of-range/future timestamps.
  **No `BLOCKER_SUMMARY_TRADES_MISMATCH`.**
- **WARN (W11):** `summary.timeframe=M15` while trades show
  `data_source_used=prepared_m5_bid` with `allow_inferred_timeframe=true` —
  likely intentional (M5 source, M15 inferred) but undocumented.

## 10. Cost Profile Audit

Quality degrades strictly and in the **economically correct direction**
(`base ≥ conservative ≥ stress`):

| Metric | base | conservative | stress | Direction |
|---|---|---|---|---|
| Profit factor | 1.0620 | 1.0450 | 1.0308 | strictly ↓ ✓ |
| Expectancy (R) | 0.0361 | 0.0272 | 0.0198 | strictly ↓ ✓ |
| Total return % | 0.1376 | 0.0734 | 0.0199 | strictly ↓ ✓ |
| Max drawdown % | 1.5213 | 1.5507 | 1.5757 | strictly ↑ ✓ |

Drawdown does **not** improve under worse costs; confirmed at per-trade level
(trade 1: −1.0319 R base → −1.0365 conservative → −1.0410 stress). Monotonicity
is real and correct. **No `BLOCKER_COST_MONOTONICITY_BROKEN`.**

## 11. Regime Obsolescence Analysis

- **15 trades total; 1 active month-block (2015-01) + 1 trade (2015-02-02);
  ~118.9 months with zero activity; PF/expectancy marginal; return irrelevant.**
- **Hypothesis A (data/pipeline/iteration bug): REFUTED.** All three
  `equity_curve.csv` = **728,997 rows**, first `2015-01-02 07:00`, last
  `2024-12-31 17:00`, distinct years = **2015..2024 (all ten)**. Ending equity
  reconciles to the cent with each summary return
  (base 100137.55→0.13755%, conservative 100073.41→0.0734%,
  stress 100019.88→0.01988%). The engine demonstrably loaded and iterated the
  full declared timeline; there is no truncation, no premature stop, no
  data-window artifact.
- **Hypothesis B (no qualifying setups post-2015-02-02): SUPPORTED, with a
  caveat.** Because the data was fully processed, the zero-activity is produced
  by the **entry logic**, not the data layer. The pattern — a healthy ~daily
  cadence for ~4 weeks then a hard cliff for ~10 years — is **not** the gradual
  taper typical of organic regime decay. It is equally consistent with an
  **over-restrictive strategy-logic/parameter condition** (e.g. the
  `atr_percentile=65` / `atr_percentile_lookback=200` gate, `min_or_atr=0.4`,
  `min_or_coverage_pct=0.9`, the 07:00–08:00 OR window) that early-2015 EURUSD
  microstructure satisfied and the post-2015 regime essentially never did.
- **Conclusion:** the *run* is sound; the data-pipeline bug is closed. The
  remaining open question — **genuine degenerate selectivity / regime death
  vs. a strategy-logic-or-parameter over-restriction defect** — is *not*
  resolvable from artifacts alone and requires a strictly **read-only**
  review of the VE-ORB entry-filter code. It does **not** require rerun,
  validation, holdout, optimization or sweep. The strategy is **economically
  dead either way**.

## 12. Classification Review

- Prior: **`VEORB_PRELIMINARY_INTERESTING_NEEDS_AUDIT`** — judged **too
  optimistic**. Leading with "INTERESTING" for a strategy the producing system
  itself flags `insufficient_sample` (n=15, 4 active weeks in 10 years, return
  ~0.0138%/yr, PF within noise) overstates merit, even though the report does
  caveat it. This is classification inflation (W8).
- Auditor classification: **`VEORB_PRELIMINARY_WATCHLIST_BUG_INVESTIGATION_ONLY`**.
  Rationale: the run is correct, safe and fully reconciled (incl. heavy equity
  endpoints); the data-pipeline bug is refuted; but the cause of post-Feb-2015
  zero-setups (genuine selectivity/regime death vs. an over-restrictive
  strategy-logic/parameter defect) is unresolved and closeable **only** by a
  bounded read-only diagnostic. The strategy is **non-investable** regardless.
- Conditional path: if a future read-only diagnostic confirms the data is
  complete (already shown) **and** the entry filter is correctly implemented and
  genuinely finds no setups by design → reclassify to
  **`VEORB_PRELIMINARY_REJECTED_LOW_EDGE_REGIME_OBSOLETE`**. There is no path
  from this evidence to "interesting", "edge", "candidate" or "validation".

## 13. Professional Decision

Explicitly and without qualification:

- **NO validation.**
- **NO holdout / sealed_holdout.**
- **NO 2025 / 2026 data.**
- **NO production / CORE_PRODUCTION.**
- **NO incubation.**
- **NO FTMO / demo / real.**
- **NO champion / portfolio candidate.**
- **NO edge declared. NO profitability declared. NOT robust.**

VE-ORB, as configured, is statistically and economically dead (15 trades / 10
years, self-flagged insufficient sample, marginal PF within noise, irrelevant
return). It must not consume validation or holdout budget. Its only permitted
forward motion is a single, bounded, **read-only** diagnostic (see §15).

## 14. Findings Table

| id | severity | category | finding | evidence | implication | required_action |
|---|---|---|---|---|---|---|
| P1 | PASS | git/safety | Run commit = 16 lightweight files only; no code/data/heavy/zip | `git show --name-only 937376fd` count=16, code/heavy/data check NONE, all paths in run dir/lab_readiness | Run did not modify code or data | None |
| P2 | PASS | output-policy | VE-ORB trades/equity properly git-ignored | `git check-ignore` → `.gitignore:121 *_DO_NOT_COMMIT*`; absent from commit | No heavy outputs committed | None |
| P3 | PASS | scope/manifest | Manifest train-only, all forbidden flags false, range 2015–2024 | `RUN_MANIFEST.json` | Scope clean | None |
| P4 | PASS | scope/config | No forbidden-scope fields; max_trades/day=2 | 3 `_ENGINE_CONFIG.json` | Configs within lab rules | None |
| P5 | PASS | reconciliation | summary ↔ cost CSV ↔ declared identical to float precision | `summary.json` ×3, `VEORB_COST_PROFILE_SUMMARY.csv` | Numbers trustworthy | None |
| P6 | PASS | reconciliation | monthly→yearly→summary tie out exactly | yearly/monthly CSV ×3 | Aggregation correct | None |
| P7 | PASS | cost-realism | Strict monotonic degradation base→stress incl. per-trade | summaries + `trades.csv` | Cost model coherent | None |
| P8 | PASS | trades | 15 trades structurally valid, ≤2/day, no news/gap/dupes | `trades.csv` ×3 | Trade ledger sound | None |
| P9 | PASS | reconciliation | Equity endpoints match summary returns to the cent | `equity_curve.csv` ×3 last row | Heavy↔light reconciled | None |
| P10 | PASS | data-integrity | Equity = 728,997 rows spanning all years 2015–2024 | `wc -l`, first/last, distinct years | Data-pipeline bug refuted | None |
| P11 | PASS | scope/scan | No real forbidden-scope usage; tokens = negatives/false-positives | no-ignore grep + direct reads | Scope confirmed clean | None |
| W1 | WARN | sample-size | n=15; `insufficient_sample=true`, `sample_penalty_applied=true` | summaries ×3 | Statistically dead; no edge | Reject as investable |
| W2 | WARN | activity | All trades 2015-01-06→2015-02-02; ~118.9 empty months | `trades.csv`, yearly/monthly | Strategy inactive ~10y | Diagnostic-only |
| W3 | WARN | misleading-metric | `positive_years=1/negative_years=0` vacuous (1 active year) | yearly.csv (only 2015 row) | False robustness impression | Correct narrative |
| W4 | WARN | materiality | 10y return 0.0199–0.1376% (~0.002–0.014%/yr) | summaries | Economically irrelevant | Reject as investable |
| W5 | WARN | reconciliation-evidence | Formal seal (sealed/exit0/violations none) prose-only; no machine artifact; manifest has only `reconciliation_required` | run dir tree; manifest; report §7 | Gate seal declared, not evidenced (data recon itself verified) | Emit a sealed/violations artifact in future runs |
| W6 | WARN | reconciliation | Report §9 conservative \$161.43 / stress \$98.42 ≠ canonical \$176.58 / \$121.09 | post-run report vs yearly/monthly | Report USD column unreliable | Correct report USD figures |
| W7 | WARN | output-policy/wording | Report §6 labels stress "High precision mode" (forbidden lab term); actual `execution_mode=stress_mode`, `high_precision_used=false` | report §6 vs config/manifest | Misleading prose only (no real usage) | Reword to "stress mode" |
| W8 | WARN | classification | "INTERESTING" inflates a statistically dead strategy | report §10 | Risk of self-deception | Downgrade (see §12) |
| W9 | WARN | traceability | Manifest run_id lacks timestamped token; no timestamp/recon-result fields | `RUN_MANIFEST.json` | Weak manifest↔folder binding | Add token/timestamp/result to manifest |
| W10 | WARN | config-auditability | Configs lack strategy/date/dataset; scope only via manifest+summary | 3 configs | Configs not self-contained | Embed scope in engine config |
| W11 | WARN | timeframe | summary M15 vs trades `prepared_m5_bid` + `allow_inferred_timeframe` | summary vs trades | Likely intentional, undocumented | Document timeframe inference |
| W12 | WARN | repo-hygiene (out of scope) | Pre-existing tracked heavy CSVs/backup under `05_MARKET_DATA_VAULT/derived_data` & `07_BACKUPS/legacy_archive_2026` (incl. `000_PARA_CHATGPT_*.zipbak`) | `git ls-files` | Not caused by this run; untouched | Repo owner review (separate from VE-ORB) |
| — | BLOCKER | — | **NONE** | — | — | — |

## 15. Allowed Next Step

**B) One lightweight diagnostic audit only — strictly read-only — to determine
why 2015-02-03 → 2024-12-31 produced 0 trades, without validation, holdout,
rerun, optimization or sweep.**

Concretely and within safety rules: a read-only review of the
`ve_orb_volatility_expansion` entry-filter code and its ATR-percentile /
opening-range coverage gating, cross-referenced against the (already verified)
fully-processed 2015–2024 timeline, to classify the post-2015 inactivity as
*correct-by-design degenerate selectivity / regime death* vs *strategy-logic or
parameter over-restriction*. No engine/strategy/data/runner modification. No
rerun. Outcome either confirms reject-as-regime-obsolete or surfaces a
lab-process logic defect — neither path reopens validation or holdout.

(Option A — immediate reject-and-archive as low-edge — is also defensible if the
lab declines to spend diagnostic effort; the strategy is non-investable under
either option. Option C is not applicable: no inconsistency / output-policy /
scope blocker was found.)

## 16. Final Recommendation

Accept the **run** as correctly executed, faithfully reconciled and safe.
**Reject the strategy as a trading edge.** Do not advance VE-ORB to validation
or holdout. Downgrade the classification from
`VEORB_PRELIMINARY_INTERESTING_NEEDS_AUDIT` to
`VEORB_PRELIMINARY_WATCHLIST_BUG_INVESTIGATION_ONLY`, with the explicit
understanding that the watchlist status exists **only** to permit one bounded
read-only diagnostic of the entry logic — not because the strategy has any
demonstrated merit. The data-pipeline bug is refuted; the surviving question is
strategy-logic selectivity vs. regime death, and both answers end in rejection
as investable. No edge. No profitability. No candidate. No production.

---

*Audit performed read-only. Artifacts and lightweight inspection commands only.
No rerun, no validation, no holdout, no 2025/2026, no optimization, no sweep,
no code/data change, no heavy-output commit, no `git add .`, no destructive git.*
