# PROJECT EXTREME READONLY AUDIT V1

## 1. Audit Status

- Audit type: extreme, destructive-intent, read-only.
- Activation gate: owner authorization phrase present as an autonomous declaration. Gate PASSED.
- Read-only contract honored: no code modified, no tests modified, no data modified, no Python executed, no scripts executed, no backtest run, no real market data loaded, no market CSV content read (names/paths/sizes only).
- Base branch at audit start: `research/bo01-phase-a-execution-prompt-warning-patch-v1-20260518` @ `3ae0fde3682ecd764ab1b80dea237b34c184a587` (local == origin).
- Audit branch: `audit/project-extreme-readonly-audit-v1-20260518` (created from the above HEAD).
- Method: direct read of safety-critical code/tests/prompts + two read-only research sub-agents (governance corpus; architecture/security) + read-only git inspection.
- One safety-positive deviation from the literal BLOQUE 0 protocol: the worktree-drift snapshot was performed in-memory (no `TEMP_*` files written/deleted) instead of writing temp files; redundant `git switch`/`git pull --ff-only` were skipped because local HEAD already equalled origin HEAD exactly. Both reduce filesystem/state mutation while preserving the same safety detection.

## 2. Executive Verdict

**PROJECT_EXTREME_AUDIT_PASS_WITH_WARNINGS**

No BLOCKER was found. The core safety architecture is sound and internally consistent:

- The audited backtest runner has no file I/O, no data-vault access, no top-level execution, no optimization surface, no broker/demo/real/FTMO/Telegram logic; it applies costs, enters at t+1 open, resolves same-bar as STOP_FIRST, and hard-blocks 2025/2026 by inspecting actual index timestamps.
- BO01 and MR02 strategies are causal in their own frame computations; an explicit no-future-poisoning test proves the strategy output does not change when all future rows are corrupted.
- The decision chain M0 → M1 → M2 → BO01 runner → real-data protocol design → Phase A prompt is contiguous and git-verifiable, fails closed where required, and carries full per-phase metadata.
- The W-01 / W-02 / W-03 Phase A prompt warning patch is correctly applied and was independently re-verified two ways (governance sub-agent git-verification + direct grep of the prompt draft). This extreme audit therefore subsumes a patch-specific audit of that warning patch.
- The Phase A execution prompt is tightly scoped (train-only, window 2015-01-05→2015-01-09, path-anchored to the gitignored train partition, SHA256 pinning required), owner-gated by an explicit activation phrase, with comprehensive abort conditions and an explicit prohibition on auto-advancing to Phase B and on edge/profitability claims.
- No live secret is present in the working tree; the `.gitignore` secrets block is effective (0 secret-like files tracked).

The warnings are real and are listed below. The two most material are HIGH and concern the **data-preparation / data-loading surface that is not covered by the audited runner**, not the runner itself.

## 3. Scope

In scope (read-only): full git/branch/worktree topology; `.gitignore` and ignore behavior; tracked-file inventory and risky-pattern classification; repository architecture; `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness` governance + prompt corpus; `BO01Strategy.py`, `MR02Strategy.py`, `bo01_backtest_runner.py`; the five BO01/MR02 test files; the real-data protocol design; the Phase A execution prompt draft + warning-patch report; data-vault structure (names/paths/sizes only); secrets/dangerous-command scan.

Out of scope by owner instruction: executing anything; loading or reading market-data CSV content; validation/holdout/2025/2026 data; optimization/sweep; FTMO/demo/real; declaring edge or profitability; modifying any code/test/data; the causal correctness of the data-preparation pipeline that produces `ema_m15_200`/`atr14` (flagged as a finding, not audited here).

## 4. Safety Verification

| Control | Result |
|---|---|
| Code modified by audit | NO |
| Tests modified by audit | NO |
| Data modified by audit | NO |
| Data loaded by audit | NO |
| Market CSV content read | NO (names/paths/sizes only) |
| Python executed by audit | NO |
| Scripts executed by audit | NO |
| Real-data backtest run | NO |
| Formal train run | NO |
| Validation run | NO |
| Holdout used | NO |
| 2025/2026 data used | NO |
| Optimization / sweep | NO |
| `git add .` used | NO (explicit per-file staging only) |
| `reset --hard` used | NO |
| `rebase` used | NO |
| `git clean` used | NO |
| `git stash` used | NO |
| force push | NO |
| Branch correctness | audit branch created from verified base HEAD |
| Active research Python process | NONE detected |
| Worktree drift during audit | NONE (in-memory A/B snapshot identical) |

## 5. Git / Repository Audit

- Base HEAD verified: local == `origin/research/bo01-phase-a-execution-prompt-warning-patch-v1-20260518` == `3ae0fde3682ecd764ab1b80dea237b34c184a587`. The W-01 runner-audit commit `5bdb4bed…` is present in the log as "audit: review bo01 backtest runner warning patch".
- Commit topology shows a disciplined alternating `research:` → `audit:` pattern across M0/M1/M2/BO01 stages.
- 4,653 tracked files. No `.zip` tracked. No secret-like file tracked. `local_outputs_do_not_commit` has 0 tracked files.
- **M-03 (MEDIUM):** 9 legacy USDJPY market CSVs (`05_MARKET_DATA_VAULT/legacy_data/data_usdjpy_*/prepared/USDJPY_{M5,M15,H1}.csv`) are tracked in git. `.gitignore` has no blanket vault or `*.csv` rule; only the EURUSD train and sealed-holdout subpaths are ignored.
- **M-04 (MEDIUM):** 744 `trades.csv`/`equity_curve.csv`-class backtest output artifacts are tracked (mostly `07_BACKUPS/legacy_archive_2026/**`, plus `05_MARKET_DATA_VAULT/derived_data/**` checkpoints, `summary.json`, `signal_log.csv`).
- **M-10 (MEDIUM):** a stale linked worktree is registered: `BOT_RESEARCH_WORKTREES/eurusd-daytime-strategy-01` (`research/eurusd-daytime-strategy-01`, marked `prunable`). It is not an active writer (no Python running), but registered stale worktrees are a drift/confusion hazard.
- **L-09 (LOW):** ~135 local branches plus worktree sprawl; the Telegram-token security remediation is reachable only from two `research/v50b-*` branches and is not consolidated to `main`/audit HEAD.
- **L-10 (LOW):** 5 untracked external-research intake files present in the working tree (`03_RESEARCH_LAB/strategy_research_intake/external_research_20260516/**`); a `desktop.ini` is tracked under `05_MARKET_DATA_VAULT/manual_data/DATA MANUAL/`.

Classification: HIGH_GIT_HYGIENE_RISK for data/outputs-in-git (M-03/M-04); MEDIUM/LOW for the rest. No BLOCKER_GIT_SECURITY.

## 6. Architecture Audit

- Clean 8-domain root separation (production / incubation / research / infrastructure / data-vault / governance / backups / cloud-run). No loose root scripts, no tracked notebooks.
- Research-lab purity is enforced by an in-repo static safety test, not only by convention.
- `scratch/` and `local_outputs_do_not_commit/` exist on disk with 0 tracked files.
- **L-08 (LOW):** `*_BACKUP_20260421_1535.py` copies of `config.py`/`engine.py`/`strategies/__init__.py` are committed beside the live modules in the active research package; core modules are also snapshot-duplicated under `03_RESEARCH_LAB/reports/canonical_*`. Source-of-truth ambiguity / refactor hazard.

Classification: PASS for separation; LOW_ORGANIZATION_NOTE for backup/snapshot duplication. No BLOCKER_ARCHITECTURE.

## 7. Governance Chain Audit

- Chain is contiguous and git-verifiable; the M2-blocked report fails closed (`BLOCKED_M2_RUNNER_NOT_AUDITED`) and forces the runner build/audit detour instead of skipping it.
- Each phase carries status/branch/commit/scope/safety/decision/allowed-next/forbidden-next blocks.
- No "no execution / no data" vs "execution / data" contradictions: design docs declare `data_loaded: NO`; only the M1/M2 execution reports declare data loaded, which is correct for their phase.
- BO01 = 638 valid structural signals and MR02 = 5 are stated consistently across all relevant reports.
- **M-07 (MEDIUM):** recurring inflated language ("executed flawlessly", "Sealed (fully guarded)", "100% compliant", "perfect execution contract compliance", "certified as structurally safe / robust", "fully secured", "perfectly sealed") across M1/M2/protocol/runner-patch reports, contradicting the project's own sober-language mandate. Mostly self-flagged but uncorrected.
- **L-01 (LOW):** `M2_TRAIN_ONLY_STRUCTURAL_RETRY_EXECUTION_REPORT_V1.md:63` mislabels its provenance "Commit SHA" as the base runner-patch SHA `f01bdf77…` when the true execution HEAD is `76a27c8a…` (self-flagged F-02).
- **L-02 (LOW):** stale/off-by-line evidence anchors in `BO01_PHASE_A_EXECUTION_PROMPT_DRAFT_EXTERNAL_AUDIT_V1.md:172-174` (cites #L111/#L182/#L248; actual 100/164-165/229) and the M2 retry audit F-03 line citation.

Classification: PASS on logic/coherence; MEDIUM_GOVERNANCE_GAP for systemic language inflation; LOW for provenance/anchor defects. No BLOCKER_GOVERNANCE_CONTRADICTION.

## 8. Quant Methodology Audit

Positives: train/validation/holdout conceptually separated; 2025/2026 protected at multiple layers; no parameter sweep present (the `parameter_grid`/`parameter_space` functions are inert — they return the single default config); entry policy fixed to `ENTRY_NEXT_CANDLE_OPEN`; same-bar resolved STOP_FIRST; 3 mandatory non-zero cost profiles (base/conservative/stress) with an explicit prohibition on qualitative model selection; MR02 excluded for low signal count; explicit negative disclaimers (no edge / profitability / readiness).

Open methodology risks:

- **H-01 (HIGH, data-leakage):** BO01 consumes precomputed columns `ema_m15_200` (M15 EMA200 mapped onto M5) and optionally `atr14`. The strategy itself is causal (`iloc[:i]` for the Asian range, `iloc[:i+1]` for inline ATR/EMA, `iat[i]` for the current bar; the no-future-poisoning test confirms it). But the causal correctness of those **precomputed** columns is determined by the data-preparation pipeline, which is NOT audited here and is NOT covered by the no-future-poisoning test (it only corrupts rows after `i`, while these columns are read at `iat[i]`). If `ema_m15_200` were aligned to an in-progress/future M15 bar, structural statistics would be lookahead-contaminated.
- **M-01 (MEDIUM):** the runner's validation/holdout guard only triggers on a literal `partition`/`split`/`dataset_split`/`data_split` column containing "validation"/"holdout", or on 2025/2026 by year. Validation/holdout data passed without such a label column and within 2015–2024 would not be detected by the runner; true protection depends on path-anchoring + hash in the loader.
- **M-02 (MEDIUM):** the protocol §7 specifies a "Max Spread Guard" (3.0/4.0 pips) that does not exist in the audited runner's `compute_cost_r`. The control is specified but unimplemented in audited code.
- Selection-by-result risk: BO01's fixed parameters (min Asian range 8 pips, ATR mult 0.5, RR 2.0, EMA20 / EMA-M15-200, sessions) must be evidenced as pre-registered before any backtest result was seen. The governance corpus presents them as design inputs and there is no sweep, but a future structural-evidence audit should explicitly confirm parameter pre-registration before any edge interpretation.
- Strict Asian-session completeness (exactly 79 bars, exact timestamp-set match) silently suppresses signals on any day with a missing M5 bar; over a 5-day Phase A window this can bias which days trade. This is conservative behavior but must be considered when interpreting Phase B.

Classification: HIGH_DATA_LEAKAGE_RISK (H-01, pre-Phase-B critical); MEDIUM_METHOD_GAP (M-01/M-02); no BLOCKER_METHODOLOGY for Phase A because Phase A is plumbing-only with no edge claims and the runner hard-blocks 2025/2026 by timestamp.

## 9. BO01 Audit

- Causal: Asian range from `frame.iloc[:i]` (strictly before `i`); ATR/EMA from `iloc[:i+1]`; current bar via `iat[i]`. No future index access.
- Contract: `_build_signal` emits `signal/direction/stop_price/target_rr` (+ extras) matching the runner's `validate_signal_contract`; rejects stop on the wrong side of price.
- Self-limits to 1 trade/day and no concurrent position (returns `None` if `daily_trade_count > 0` or `has_active_position`), duplicating the runner guard (defense in depth).
- Residual risk: dependence on precomputed `ema_m15_200`/`atr14` (see H-01).

Classification: PASS for strategy causality/contract; HIGH risk inherited from H-01 (upstream feature construction).

## 10. MR02 Audit

- Causal and contract-correct; does NOT depend on `ema_m15_200` (required columns are OHLC only) — cleaner dependency surface than BO01.
- Very low frequency by design (5 structural signals over a quarter). Correctly excluded from the Phase A protocol and deferred.
- Risk is misuse, not code: 5 signals is not statistically interpretable; MR02 must remain paused and out of Phase A, and must not be advanced to backtest/edge interpretation on current evidence.

Classification: PASS (code); MEDIUM_MR02_GAP only if any future prompt reintroduces MR02 without additional evidence — currently governance correctly excludes it.

## 11. Backtest Runner Audit (static, no execution)

PASS items: no `open`/`read_csv`/`to_csv`/`Path`/`os`/`pathlib`/`sys` imports or calls; imports limited to numpy/pandas/typing; no module-level execution or `__main__`; constants correct (`ENTRY_NEXT_CANDLE_OPEN`, `STOP_FIRST`, `MAX_TRADES_PER_DAY=1`, `MAX_ACTIVE_POSITIONS=1`); 2025/2026 blocked by `frame.index.year.isin([2025,2026])` (timestamp-level, not just endpoints); signal contract fails closed (TypeError/ValueError caught and counted); entry strictly at `idx+1` open with t+1-existence guard; STOP_FIRST same-bar returns −1R; single active position enforced; costs subtracted (`net_r = gross_r − cost_r`); metrics in-memory only; no optimization; no broker/demo/real/FTMO/Telegram; no output writes.

Gaps:

- **M-01 (MEDIUM):** validation/holdout guard is bypassable (label-column only); no positive train-window enforcement (only negative 2025/2026 block).
- **L-03 (LOW):** `resolve_trade_exit` supports `timeout_bars` but `run_bo01_backtest_on_frame` never passes it; `timeout_count` is therefore always 0 (dead metric vs protocol §8).
- **L-04 (LOW):** `winrate` is computed on `gross_r > 0` while streaks/expectancy use `net_r`; mixing gross and net is internally inconsistent.
- **L-05 (LOW):** `except Exception` around `strategy.signal` is broad; a systematically failing strategy would yield 0 trades and "look safe" while being broken (counted as `exception_count`, not surfaced).
- **L-07 (LOW):** `max_trades_per_day` is a caller-overridable parameter (default 1); module constant `MAX_TRADES_PER_DAY` is decorative and not enforced in the loop. The per-day boundary is the index-tz calendar date; for a London-Breakout strategy on UTC data the "day" is UTC-calendar, a session-alignment note.
- Open-position trades reaching end-of-frame are marked-to-close and their unrealized R is included in headline `net_R`/metrics; over a 5-day Phase A window this can distort aggregate R (counted via `open_end_count`).

Classification: PASS on all hard-safety items; MEDIUM_RUNNER_GAP (M-01); LOW_RUNNER_NOTE for the rest. No BLOCKER_RUNNER.

## 12. Tests Audit (static, no execution)

- All five test files are synthetic, build in-memory tz-aware frames, do not read real data, and do not write outputs. A self-test asserts no external data access.
- Coverage present: import/constants; frame validation success and 7 failure modes; 2025/2026 blocked at endpoints AND internal positions; validation/holdout partition blocked (`partition` and `dataset_split` columns); signal-contract success/failure (None/non-dict → TypeError, missing keys/invalid values → ValueError); entry t+1 open; max 1 trade/day; ignore signals while position open; STOP_FIRST long & short; clean stop/target long & short; no-t+1 abort; costs reduce net R with explicit arithmetic; commission R standard-lot scaling; skipped-active-position counter; non-dict signal fails closed; BO01 contract incl. **no-future-poisoning**, missing/duplicate/wrong-cadence Asian endpoint fail-closed, file-access-during-signal patched to raise, warmup gate, fail-closed on missing column/tz-naive/NaN/state, forbidden-scope token scan.
- **L-06 (LOW):** the runner safety static-scan asserts absence of `read_csv`/`to_csv`/`Path(` but not `open(`/`os.`/`subprocess`/`requests`/`socket`/`urllib`; the runner's no-file-I/O property is enforced by a brittle string scan, not a behavioral test that patches `open` (the strategy contract test does patch `open`, the runner test does not). A future edit adding `open(...)` to the runner would not be caught.
- Gap: no test asserts the precomputed-feature causality (H-01) — by nature this cannot be caught at the strategy/runner test layer; it belongs to a data-prep audit.

Classification: PASS overall (synthetic, covers the critical negatives); LOW_TEST_NOTE (L-06). No BLOCKER_TESTS.

## 13. Data Policy Audit (no CSV content read)

- `eurusd_data/` has four partitions on disk: `prepared_train_2015_2024` (gitignored ✓), `sealed_holdout_2025_2026` (gitignored ✓), `data_candidates_2022_2025` (NOT gitignored), `data_free_2020` (NOT gitignored). 0 `eurusd_data` files are tracked in git (the train/holdout/candidate/free data is not committed) ✓.
- The Phase A prompt anchors data access to `prepared_train_2015_2024/prepared/EURUSD_{M5,M15}.csv` (the gitignored partition) and requires SHA256 hash logging of loaded CSVs (prompt line 87/207/240).
- **M-05 (MEDIUM):** `data_candidates_2022_2025` (contains 2025) and `data_free_2020` are not in `.gitignore`; they are protected only by a read-only attribute and by not having been added. A future `git add` of those paths would commit 2025 data.
- **M-06 (MEDIUM):** `local_outputs_do_not_commit` is ignored only via the case-insensitive `*_DO_NOT_COMMIT*` rule, which matches only because Windows `core.ignorecase=true`. On a case-sensitive filesystem (Linux/cloud CI) this rule would not match and Phase A local outputs could be committable. The protocol's "outputs blocked by gitignore" guarantee rests on this fragile rule.
- **M-09 (MEDIUM):** the protocol design §5 Data Proof Requirements omit the SHA256 hash that the later Phase A prompt §87 adds. The operative prompt is stricter (good); the design doc was not back-patched (consistency gap).
- **H-02 (HIGH):** the strongest leakage controls (path-anchoring, partition verification, 2025/2026 check, SHA256 pinning, monotonicity/NaN proof) are specified to run in the data-loading/data-proof layer. Per W-02 the execution script is OPTIONAL and is NOT part of the audited runner; that loader code is therefore the main unaudited surface that Phase A will exercise. Backstop: the audited runner independently hard-blocks 2025/2026 by timestamp and raises on NaN/non-monotonic/missing columns, which mitigates the date-leakage subset but not validation/holdout-by-content or feature causality.

Classification: HIGH_DATA_LEAKAGE_RISK (H-02); MEDIUM_DATA_POLICY_GAP (M-05/M-06/M-09). No BLOCKER_DATA_POLICY (train/holdout partitions are gitignored, nothing leakage-critical is committed, runner has a 2025/2026 backstop).

## 14. Security / Secrets Audit

- No live secret in the working tree: no private keys, cloud keys, connection strings, hardcoded credentials, or active Telegram token. Only tracked secret-like file is a safe `.example` DEMO template (no credentials). `.gitignore` secrets block is effective (`.env*`, `*secret*`, `*.key`, `*token*` → 0 tracked).
- Committed `_temp_*.py`/`debug_*.py` archive scripts were reviewed: no destructive shell, no `reset --hard`/`git clean`/`stash`/recursive delete, no external `Invoke-WebRequest`/`curl`/`wget`. The only `Remove-Item -Force` hits delete a single self-created zip path.
- **M-08 (MEDIUM):** a Telegram bot token WAS committed historically, then detected and revoked with a documented remediation dossier (commit `66768383`). No live token remains in the working tree. Residual: the raw token may persist in old git objects / other branches, and the remediation evidence is branch-siloed (not on `main`/audit HEAD). Provider-side revocation should be positively confirmed; history scrub and evidence consolidation are owner decisions.

Classification: MEDIUM (historical incident, mitigated, not currently leaking). No BLOCKER_SECRET_EXPOSURE in the audited tree.

## 15. Outputs Audit

- `local_outputs_do_not_commit` has 0 tracked files; the protocol restricts local outputs to that gitignored folder and lists only 2 committable governance docs.
- **M-04 (MEDIUM):** 744 historical `trades.csv`/`equity_curve.csv`/`signal_log.csv`/checkpoint artifacts are tracked under `07_BACKUPS/legacy_archive_2026/**` and `05_MARKET_DATA_VAULT/derived_data/**`. These are legacy/forensic, not Phase A outputs, but they establish a precedent of committed backtest outputs and bloat the repo; the Phase A pipeline must not extend this pattern.
- No ZIP tracked; no Phase A output committed (Phase A has not run).

Classification: MEDIUM_OUTPUT_GAP (legacy committed outputs); PASS for the Phase A output policy itself.

## 16. Prompts Audit

- Phase A prompt activation phrase (line 11) requires the owner to explicitly state train-only / no validation / no holdout / no 2025-2026 / no optimization-sweep / no demo-real-FTMO / no edge claims.
- Scope, data path (gitignored train partition), window (2015-01-05→2015-01-09 UTC), SHA256 pinning, runner-pin to `5bdb4bed…`, optional temporary script (W-02), W-03 handoff flags, abort conditions, and FORBIDDEN_NEXT_STEPS are all present and consistent.
- No execution is hidden inside a design prompt and no design is smuggled into the execution prompt. No prompt authorizes a dangerous next step.
- W-01/W-02/W-03 verified correctly applied (governance sub-agent git-verification + independent grep).
- **L-02 (LOW):** the draft external-audit findings table cites stale draft line numbers.

Classification: PASS; LOW_PROMPT_NOTE (L-02). No BLOCKER_PROMPT.

## 17. Phase A Readiness Audit

- Does the Phase A warning patch need a separate specific audit before execution? **No.** This extreme audit independently re-verified W-01/W-02/W-03 two ways; a dedicated patch audit would be redundant.
- Are there blockers? **No.**
- Are there warnings that should be addressed or formally accepted before Phase A? **Yes** — primarily H-02 (the data-loader/data-proof surface that Phase A will exercise is the main unaudited code) and H-01 (precomputed-feature causality), plus the data-policy MEDIUMs (M-05/M-06/M-09).
- Mitigation already in place for Phase A specifically: Phase A is explicitly plumbing-only, draws no edge conclusions, is owner-gated, path-anchored to the gitignored train partition, requires SHA256 pinning, and runs on the audited runner whose 2025/2026 timestamp block + NaN/monotonicity validation act as an independent backstop.
- H-01 (feature causality) is principally a **pre-Phase-B** concern: it threatens structural-statistic interpretation, not the Phase A plumbing check, but it must be resolved before any structural/edge interpretation.

Readiness state: **READY_FOR_PHASE_A_AFTER_OWNER_APPROVAL** conditional on the owner formally accepting or remediating H-02 and the data-policy MEDIUMs, and on scheduling an H-01 data-prep causality audit before Phase B.

## 18. Findings Summary

- BLOCKERS: 0
- HIGH: 2 (H-01 precomputed-feature causality unaudited; H-02 data-loader/data-proof surface unaudited)
- MEDIUM: 10 (M-01 … M-10)
- LOW: 10 (L-01 … L-10)
- INFO: 9 (I-01 … I-09)

## 19. Top 20 Most Important Findings

1. H-02 — The data-loading/data-proof layer (path/partition/2025-2026/SHA256/monotonicity) is the OPTIONAL, unaudited script per W-02; it is the main code Phase A will exercise.
2. H-01 — `ema_m15_200`/`atr14` precomputed-feature causal correctness is not audited and not covered by the no-future-poisoning test; pre-Phase-B critical.
3. M-01 — Runner validation/holdout guard is bypassable (label-column / 2025-2026-year only); no positive train-window enforcement in the runner.
4. M-03 — 9 legacy USDJPY market CSVs tracked in git; no blanket vault/`*.csv` ignore rule.
5. M-04 — 744 backtest output artifacts (trades/equity/checkpoints) tracked in git.
6. M-05 — `data_candidates_2022_2025` (incl. 2025) and `data_free_2020` not gitignored.
7. M-06 — `local_outputs_do_not_commit` ignored only via case-insensitive `*_DO_NOT_COMMIT*`; fragile on case-sensitive/cloud CI.
8. M-02 — Protocol "Max Spread Guard" specified but absent from the audited runner.
9. M-09 — Protocol design §5 lacks the SHA256 requirement the Phase A prompt adds (doc inconsistency).
10. M-07 — Systemic inflated language across M1/M2/protocol/runner-patch reports vs the project's own sober-language rule.
11. M-08 — Historical Telegram-token exposure: revoked + documented, but branch-siloed and possibly still in old git objects.
12. M-10 — Stale `prunable` linked worktree registered.
13. L-06 — Runner no-file-I/O enforced by a brittle string scan that misses `open(`/`os.`/`subprocess`/network.
14. L-03 — `timeout_count` metric never produced (timeout never wired) vs protocol §8.
15. L-04 — `winrate` on gross R while streaks/expectancy on net R (metric inconsistency).
16. L-05 — Broad `except Exception` around `strategy.signal` can mask a systematically broken strategy.
17. L-07 — `max_trades_per_day` caller-overridable; per-day boundary is index-tz calendar date (session-alignment note).
18. L-01 — M2 retry execution report mislabels its provenance commit SHA (self-flagged).
19. L-08 — `*_BACKUP_<ts>.py` committed beside live `config.py`/`engine.py`; module snapshot duplication.
20. L-09 — ~135-branch/worktree sprawl; security remediation not consolidated to `main`.

## 20. Required Actions Before Phase A

These are owner decisions; this audit does not execute or authorize them.

1. Decide H-02: either audit the data-loader/data-proof code path before Phase A, OR formally accept that Phase A relies on the audited runner's independent 2025/2026-timestamp + NaN/monotonicity backstop and treats the loader script as throwaway plumbing, recording that acceptance.
2. Formally accept or remediate the data-policy MEDIUMs: add `.gitignore` coverage for `data_candidates_2022_2025`, `data_free_2020`, and a comprehensive case-independent outputs/vault rule (M-05/M-06); reconcile protocol §5 with the prompt's SHA256 requirement (M-09).
3. Schedule (do not yet run) a pre-Phase-B data-prep causality audit of `ema_m15_200`/`atr14` (H-01).
4. Confirm Phase A will run under the exact owner activation phrase and the verified Phase A prompt (no modification).

## 21. Recommended Actions After Phase A

1. Pre-Phase-B: audit precomputed-feature causality (H-01) and parameter pre-registration evidence.
2. Repo-hygiene pass: extract committed market data + 744 output artifacts from tracked history decision (M-03/M-04); prune stale worktree (M-10) and dead branches (L-09).
3. Sober-language remediation pass over M1/M2/protocol/runner-patch reports (M-07); correct provenance SHA and stale anchors (L-01/L-02).
4. Runner LOW cleanups if/when the runner is next legitimately revised (L-03/L-04/L-05/L-06/L-07) — not before Phase A, to preserve the audited runner pin.
5. Consolidate the Telegram-token remediation evidence to `main` and decide on history scrub (M-08).

## 22. Final Decision

**PROJECT_EXTREME_AUDIT_PASS_WITH_WARNINGS.** No blocker. The project does not appear to be self-deceiving on the core safety architecture: the runner is pure, the strategies are causal, the chain is coherent and git-verifiable, the Phase A prompt is correctly patched and owner-gated, and edge/profitability are explicitly disclaimed. The material warnings concern the data-preparation/data-loading surface outside the audited runner and repository data/output hygiene. This audit does NOT authorize FTMO, demo, or real trading, does NOT declare edge or profitability, and does NOT itself authorize Phase A execution.

## 23. Allowed Next Step

Exactly one, chosen by the owner (see the companion NEXT_PROMPT document):

- **C (primary recommendation):** Execute Phase A later — only after the owner explicitly issues the exact Phase A activation phrase and only because zero blockers were found — with H-02 and the data-policy MEDIUMs formally accepted or remediated first, and an H-01 audit scheduled before Phase B.
- **B (alternative):** Patch the data-policy MEDIUMs and decide H-02 (audit the loader) before Phase A.
- A separate Phase-A-warning-patch-specific audit is NOT required (this audit subsumes it).
- **D (fallback):** Freeze and run the repo-hygiene/language remediation pass first.

## 24. Forbidden Next Steps

- No immediate Phase A execution authorized by this audit alone.
- No validation.
- No holdout.
- No 2025/2026 data.
- No optimization / sweep / grid / walk-forward / parameter search.
- No demo / real / FTMO.
- No edge / profitability / rentabilidad / "strategy ready" claims.
- No modification of the audited runner or strategy classes (would break the W-01 pin).
