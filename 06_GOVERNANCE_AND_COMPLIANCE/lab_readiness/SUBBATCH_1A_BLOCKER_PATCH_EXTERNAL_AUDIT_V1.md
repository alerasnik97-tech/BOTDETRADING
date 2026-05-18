# SUBBATCH 1A BLOCKER PATCH EXTERNAL AUDIT V1

## 1. Audit Status

AUDIT_PASS_SUBBATCH_1A_BLOCKER_PATCH_READY_FOR_OWNER_REVIEW

Read-only external audit completed. No code, tests, or data were modified by
this audit. No prohibited execution was performed.

## 2. Executive Verdict

The audited commit corrects the declared Asian-range completeness blockers in
BO01 and MR02 and adds targeted unit/contract coverage that genuinely fails
against the prior count-only implementation. The diff is confined to the eight
authorized files. Lightweight permitted tests were executed and all passed.
The static safety scan found no blockers. This audit makes no statement about
strategy edge, performance, or profitability and authorizes no laboratory
execution. Two non-blocking warnings concern pre-existing repository state that
this patch did not introduce and this audit did not touch.

## 3. Scope Audited

- base branch: `research/subbatch-1a-blocker-patch-v1-20260517`
- audit branch: `audit/subbatch-1a-blocker-patch-v1-20260517`
- audited commit: `fdce9603f28e03ba24f92a64235f5a031e758a14`
- origin ref verified equal to local HEAD and to the audited commit.
- files inspected (full read): the two strategy files, the four BO01/MR02 test
  files, the patch report, and the next-audit prompt.
- diff inspected: `git show`/`git diff` for the audited commit only.
- tests run: lightweight BO01/MR02 unit/contract suites plus three related
  lightweight contract suites.
- no-execution confirmation: no backtest, micro-run, dry-run, validation,
  holdout, 2025/2026 access, optimization, or sweep was run.

## 4. Safety Verification

- code modified by audit? NO
- tests modified by audit? NO
- data modified? NO
- strategy modified by audit? NO
- engine modified? NO
- runner modified? NO
- registry modified? NO
- strategies/__init__.py modified? NO
- backtest run? NO
- micro-run? NO
- dry-run? NO
- validation run? NO
- holdout used? NO
- 2025/2026 used? NO
- optimization/sweep? NO
- git add dot used? NO
- force push? NO

## 5. Diff Scope Audit

`git show --stat fdce9603` reports exactly eight changed files
(387 insertions, 8 deletions):

1. `03_RESEARCH_LAB/research_lab/strategies/BO01Strategy.py` (modified)
2. `03_RESEARCH_LAB/research_lab/strategies/MR02Strategy.py` (modified)
3. `03_RESEARCH_LAB/research_lab/tests/test_strategy_contract_bo01.py` (modified)
4. `03_RESEARCH_LAB/research_lab/tests/test_strategy_tz_bo01.py` (modified)
5. `03_RESEARCH_LAB/research_lab/tests/test_strategy_contract_mr02.py` (modified)
6. `03_RESEARCH_LAB/research_lab/tests/test_strategy_tz_mr02.py` (modified)
7. `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/SUBBATCH_1A_BLOCKER_PATCH_REPORT_V1.md` (added)
8. `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/NEXT_PROMPT_AUDIT_SUBBATCH_1A_BLOCKER_PATCH_V1.md` (added)

This matches the authorized whitelist exactly. No `strategies/__init__.py`,
registry, factory, engine, runner, data, configs, root files, outputs, ZIP,
trades/equity, or MR03/LS01/LS02 files appear in the commit. Files added by
the commit are limited to the two governance markdown documents; the six code
and test files were modified in place.

Result: authorized files only. Unauthorized files: NONE.

## 6. BO01 Range Completeness Audit

`BO01Strategy.py` adds `_expected_asian_timestamps_utc(trade_date, start, end)`
and rewrites the selection loop in `_asian_range`.

- Expected timestamps: built from the UTC trade date at UTC midnight; the
  helper returns `[]` unless `start >= 0`, `end >= start`, and
  `(end - start) % 5 == 0`. With `asian_start="00:00"` and
  `asian_end="06:30"` (`DEFAULT_PARAMS`), `_minute` yields `start=0`,
  `end=390`, producing `range(0, 391, 5)` = 79 timestamps
  (`00:00, 00:05, ... 06:30`). `ASIAN_MIN_BARS = 79`; `_asian_range`
  returns `None` if `len(expected) != ASIAN_MIN_BARS`.
- Endpoints: `00:00` and `06:30` are members of `expected_set`; absence of
  either yields `set(selected) != expected_set` -> `None`.
- Missing interior bar / partial range: same set comparison -> `None`.
- Duplicate timestamp: `stamp in selected` -> `None`.
- Off-grid timestamp within range: `stamp not in expected_set` -> `None`.
- Wrong cadence: denser cadence introduces off-grid stamps; sparser cadence
  drops expected stamps; both fail closed.
- Causality: range is built from `rows = frame.iloc[:i]` (strictly before `i`).
  No `frame.iloc[i+1:]`, no future rows, no global min/max over the full frame.
- Fail closed: tz-naive index at `i` (`_utc_ts` returns `None`), tz-naive
  interior row, missing/duplicate/off-grid stamps, and NaN in window
  `high`/`low` all return `None`.
- Scope: entry, ATR, EMA, stop (midpoint), target (`target_rr`), and
  parameters are unchanged by the commit; no I/O or external data.

Classification: PASS_BO01_RANGE_COMPLETENESS_PATCH

## 7. MR02 Range Completeness Audit

`MR02Strategy.py` applies the identical helper and `_asian_range` rewrite.

- Exact `00:00`-`06:30` GMT timestamps, strict M5 cadence, count 79
  (`ASIAN_MIN_BARS = 79`, `asian_start="00:00"`, `asian_end="06:30"`).
- Missing, duplicate, and off-grid timestamps and wrong cadence all fail
  closed via the same set/dict logic as BO01.
- Causality: `rows = frame.iloc[:i]`; the prior-swing window is
  `frame.iloc[max(0, i - 3):i]` (excludes bar `i`). No rows after `i`.
- Fail closed on tz-naive index/row and NaN window values.
- Fakeout and engulfing logic (`_bearish_engulfing`, `_bullish_engulfing`),
  ATR (`_atr_at`), stop buffer (`fakeout_stop_buffer_pips`), and target
  (`target_rr = 1.5`) are unchanged by the commit; no I/O or external data.

Classification: PASS_MR02_RANGE_COMPLETENESS_PATCH

## 8. BO01 Tests Audit

`test_strategy_contract_bo01.py` adds: missing `06:30` endpoint fail-closed,
duplicate Asian timestamp replacing a missing bar fail-closed, wrong cadence
fail-closed, and a short-side eligible signal contract test. `test_strategy_tz_bo01.py`
adds a timezone test for a missing `06:30` endpoint during a valid entry time.
Pre-existing coverage is retained: future-poisoning invariance, warmup gate,
fail-closed for missing columns / tz-naive / NaN / state, no file access during
`signal`, GMT/DST behavior, and forbidden-token absence.

The missing-endpoint, duplicate, and wrong-cadence fixtures keep the in-window
bar count at 79, so the prior count-only check would have proceeded and emitted
a signal; the patched logic returns `None`. The new tests therefore
discriminate the patched implementation from the prior one and are not
decorative. All fixtures are synthetic; no data vault, internet, runner, or
backtest; no `assertTrue`-only assertions; no unexplained skips/xfails.

Classification: PASS_BO01_TESTS_COVER_BLOCKERS

## 9. MR02 Tests Audit

`test_strategy_contract_mr02.py` adds: missing `06:30` endpoint fail-closed,
duplicate timestamp fail-closed, wrong cadence fail-closed, long-side eligible
fakeout signal contract, and a third-prior-bar breach eligibility test
(swing high placed on bar `i-3`, asserting an eligible short). `test_strategy_tz_mr02.py`
adds the missing `06:30` endpoint timezone test. Pre-existing coverage is
retained (future-poisoning, warmup, fail-closed state, no file access,
GMT/DST, forbidden tokens).

Same count-preserving fixture reasoning as BO01: the new fail-closed tests
fail against the prior count-only logic and pass after the completeness patch.
The third-prior-bar test exercises the causal `frame.iloc[max(0, i-3):i]`
window. All fixtures synthetic; no data vault, internet, runner, or backtest;
no decorative assertions; no unexplained skips/xfails.

Classification: PASS_MR02_TESTS_COVER_BLOCKERS

## 10. Patch Report Audit

`SUBBATCH_1A_BLOCKER_PATCH_REPORT_V1.md` states the correct status, lists the
F-001..F-006 blockers, lists the eight files accurately, lists the added tests,
documents no backtest / micro-run / dry-run / validation / holdout / 2025-2026 /
optimization / sweep, and sets the decision and allowed next step to external
read-only audit only. It explicitly declines any edge, performance, or
profitability claim. No prohibited promotional language ("perfect",
"guaranteed", "100%", "champion", "profitable", "FTMO", "demo", "real")
is present.

Observation (non-blocking): the report's self-reported safety-scan tally
("allowed hits: 29 lines") differs from this audit's independently observed
count of allowed governance/negative-declaration hit-lines. The safety-relevant
conclusion (zero blockers; all hits are negative declarations or governance
wording) is identical and unaffected.

Classification: PASS_REPORT_AND_PROMPT_SAFE (report)

## 11. Future Prompt Audit

`NEXT_PROMPT_AUDIT_SUBBATCH_1A_BLOCKER_PATCH_V1.md` is read-only and
owner-gated: it forbids modifying code/tests/data, forbids micro-run, dry-run,
backtest, validation, holdout, 2025/2026, optimization, and sweep, permits
auditing the diff and running the lightweight tests, permits creating one
markdown report, and explicitly states it must not authorize execution. It is
the prompt that governs this audit.

Classification: PASS_REPORT_AND_PROMPT_SAFE (prompt)

## 12. Test Execution Audit

Environment: `PYTHONPATH=03_RESEARCH_LAB`, Python 3.14.3, `unittest` discover,
all suites synthetic and sub-second.

New/modified suites:

- `test_strategy_contract_bo01.py` — 11 tests, OK
- `test_strategy_tz_bo01.py` — 7 tests, OK
- `test_strategy_contract_mr02.py` — 12 tests, OK
- `test_strategy_tz_mr02.py` — 7 tests, OK

Related lightweight contract suites:

- `test_engine_strategy_contract.py` — 7 tests, OK
- `test_engine_time_contract.py` — 5 tests, OK
- `test_strategy_activity_gates.py` — 6 tests, OK

Total: 55 passed, 0 failed. This matches the patch report. No
`formal_train_runner`, `--execute`, backtest, micro-run, dry-run, validation,
holdout, or optimization/sweep was invoked.

## 13. Static Safety Scan

Scan over the eight patch files for the prohibited token set:

- `BO01Strategy.py`: no matches.
- `MR02Strategy.py`: no matches.
- four BO01/MR02 test files: no matches (forbidden tokens are intentionally
  split in source, e.g. `"20" + "25"`, so no literal forbidden string exists).
- `SUBBATCH_1A_BLOCKER_PATCH_REPORT_V1.md`: ~18 hit-lines, all of the form
  "no micro-run / no dry-run / no backtest / no validation / no holdout /
  no 2025-2026 / no optimization-sweep" — NEGATIVE_DECLARATION_OK.
- `NEXT_PROMPT_AUDIT_SUBBATCH_1A_BLOCKER_PATCH_V1.md`: ~2 hit-lines,
  prohibition/governance wording — GOVERNANCE_TERM_OK.

Blockers: 0. All hits classified NEGATIVE_DECLARATION_OK / GOVERNANCE_TERM_OK.

## 14. Output Policy / Git Audit

- The audited commit added no ZIP, no trades/equity CSV, no local outputs, no
  secrets, no data, and no code/tests outside the BO01/MR02 + four-test +
  two-governance-doc scope. Verified via `git show --name-only` and
  `git show --diff-filter=A`.
- No secret-like tracked files (`.env`, `.pem`, `.key`, `kaggle.json`,
  `.netrc`, `secrets/`, `credentials`) exist anywhere in the index.
- `git status` shows only pre-existing dirty files under
  `03_RESEARCH_LAB/strategy_research_intake/external_research_20260516/`.
  These are unrelated to this patch, were not staged, and were not touched by
  this audit — WARN_PREEXISTING_DIRTY_NOT_TOUCHED.
- `git ls-files` shows pre-existing tracked `trades.csv` / `equity_curve.csv` /
  `.zipbak` artifacts under `07_BACKUPS/`, `05_MARKET_DATA_VAULT/`, and legacy
  directories. None are in the audited commit; this is pre-existing repository
  output debt not introduced by this patch —
  WARN_PREEXISTING_REPO_OUTPUT_DEBT_NOT_FROM_PATCH.

Classification: PASS_OUTPUT_POLICY (no blocker caused by this patch),
with the two pre-existing warnings above.

## 15. Findings Table

| id | severity | category | finding | evidence | implication | required_action |
|----|----------|----------|---------|----------|-------------|-----------------|
| P-01 | PASS | diff scope | Commit touches exactly the 8 authorized files | `git show --stat fdce9603` | No scope creep | None |
| P-02 | PASS | BO01 range | Exact 79-stamp M5 expected-set with endpoint/dup/off-grid/NaN/tz fail-closed, rows `< i` only | `BO01Strategy.py:66-145` | F-001 corrected | None |
| P-03 | PASS | MR02 range | Identical completeness contract; causal `iloc[max(0,i-3):i]` swing window | `MR02Strategy.py:66-135,235` | F-002 corrected | None |
| P-04 | PASS | BO01 tests | Missing/dup/wrong-cadence + short-side tests fail against prior count-only logic | `test_strategy_contract_bo01.py`, `test_strategy_tz_bo01.py` | F-003/F-005 corrected | None |
| P-05 | PASS | MR02 tests | Missing/dup/wrong-cadence + long-side + third-prior-bar tests | `test_strategy_contract_mr02.py`, `test_strategy_tz_mr02.py` | F-004/F-005/F-006 corrected | None |
| P-06 | PASS | test execution | 55 passed, 0 failed; synthetic, sub-second | unittest output | Coverage executes green | None |
| P-07 | PASS | safety scan | 0 blockers; strategy/test files clean; md hits are negative declarations | Grep over 8 files | No prohibited usage | None |
| P-08 | PASS | report/prompt | Sober; no edge/performance/profitability; no execution authorized | report + next prompt | Policy compliant | None |
| W-01 | WARN | repo hygiene | Pre-existing dirty tree under `strategy_research_intake/external_research_20260516/` | `git status --short` | Not patch-caused; untouched | Owner awareness only |
| W-02 | WARN | repo hygiene | Pre-existing tracked trades/equity/zipbak debt in `07_BACKUPS`/`05_MARKET_DATA_VAULT`/legacy | `git ls-files` | Not patch-caused; pre-dates this commit | Owner awareness only |
| I-01 | INFO | report wording | Report's "29 lines" safety tally differs from observed count; conclusion unchanged | report sec. 10 | Cosmetic only | Optional wording fix |

Blockers: 0. Warnings: 2 (W-01, W-02). Informational: 1 (I-01). Passes: 8.

## 16. Decision

- The Sub-Batch 1A blocker patch is apt for owner review. The declared
  blockers F-001 through F-006 are corrected in code and covered by tests
  that discriminate the patched implementation from the prior one.
- Warnings: two non-blocking, pre-existing repository-state warnings (W-01,
  W-02) that this patch did not introduce and this audit did not modify.
- Blockers: none.
- This audit does NOT authorize micro-run.
- This audit does NOT authorize dry-run.
- This audit does NOT authorize backtest or formal train.
- This audit does NOT authorize validation, holdout, or 2025/2026 data.
- This audit does NOT authorize optimization or sweep.

## 17. Allowed Next Step

A) Owner decision whether to request a separate micro-run protocol design
prompt. No execution is authorized by this step; it only enables the owner to
decide whether to commission a separate, independently audited micro-run
protocol design.

## 18. Forbidden Next Steps

- no immediate micro-run.
- no immediate dry-run.
- no immediate backtest.
- no formal train.
- no validation.
- no holdout.
- no 2025/2026.
- no optimization/sweep.
- no Sub-Batch 1B.
- no parallel writers.

## 19. Final Institutional Verdict

The patch corrects the BO01/MR02 Asian-range completeness blockers and their
test coverage within the authorized eight-file scope. Logic is causal and
fail-closed; 55 lightweight tests pass; the safety scan has zero blockers. Two
pre-existing repository warnings are documented and are not patch-caused. No
edge, performance, or profitability is asserted. The patch is ready for owner
review. Micro-run, dry-run, backtest, validation, holdout, 2025/2026, and
optimization/sweep remain unauthorized.
