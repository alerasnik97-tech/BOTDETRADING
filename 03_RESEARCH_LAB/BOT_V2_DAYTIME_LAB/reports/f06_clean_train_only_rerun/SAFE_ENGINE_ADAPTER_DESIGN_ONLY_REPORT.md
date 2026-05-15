# SAFE ENGINE ADAPTER DESIGN-ONLY REPORT

Generated: 2026-05-15
Branch: research/f06-clean-train-only-rerun-20260515
PR #6 head at design time: d988c03ad7a60c0080a767baf740161346b1222c
Mode: DESIGN ONLY. No adapter implemented. No engine executed. No real F06/backtest/validation/holdout/2025/2026.

## 1. Status
SAFE_ENGINE_ADAPTER_DESIGN_COMPLETE

## 2. Executive Summary
PR #6 was re-confirmed read-only (clean tree, 119/119, safe checks green). A read-only engine
inventory established that the canonical engine is the repo-root tracked package `research_lab/`
(`engine.py::run_backtest`, `config.py`), not a module inside the PR tree. A complete fail-closed
adapter contract was designed (`SAFE_ENGINE_ADAPTER_DESIGN_SPEC.md`), a 20-row risk register was
authored (`SAFE_ENGINE_ADAPTER_RISK_REGISTER.md`), and a ready future implementation prompt was
drafted (`NEXT_PROMPT_SAFE_ENGINE_ADAPTER_IMPLEMENTATION.md`). No executable adapter code was
created; the proposed `adapters/` package does not exist yet by design.

## 3. Phase A Confirmation
- git head: local = origin = `d988c03ad7a60c0080a767baf740161346b1222c`; branch `research/f06-clean-train-only-rerun-20260515`; not main; no tracked changes; no python processes.
- tests: `python -m unittest discover -s .../f06_evidence_rebuild/tests -p "test_*.py"` → 119 total / 119 passed / 0 failed.
- safe checks: validate_config = PASS; dry_run good = DRY_RUN_SCHEMA_VALIDATED; dry_run `--emit ../MANIFEST.json` = BLOCKED_FORBIDDEN_OUTPUT_PATH; run_phase3 without confirmation = BLOCKED_MISSING_EXPLICIT_REAL_RUN_CONFIRMATION.
- cleanup status: LOCAL_CLEANUP_COMPLETE_119_PASS (prior untracked pollution removed; not recreated).

## 4. Engine Inventory Summary
- Engine: `research_lab/engine.py` (998 lines, tracked, last commit cefdd8db). Public entrypoint `run_backtest(strategy_module, frame, params, engine_config, news_block, news_filter_used, *, ...) -> BacktestResult`.
- `run_backtest` does NOT load data and does NOT enforce Phase 3 governance — all governance must live in the adapter.
- Cost (spread/slippage/round-turn) and session/timezone surfaces exist and are configurable; `max_trades_per_day` is NOT engine-enforced (adapter must enforce).
- Forbidden surfaces identified: `validation.py`, `wfa.py`, `*_BACKUP_*`, `reports/canonical_*` engine copies, `mt5_demo_executor_lab`, engine native results dir, quarantine/v50b tokens.
- Inventory verdict: READY_FOR_ADAPTER_DESIGN; NEEDS_MORE_ENGINE_DISCOVERY before implementation.

## 5. Adapter Design Summary
- Proposed (NOT created): `pipelines/f06_evidence_rebuild/adapters/phase3_f06_engine_adapter.py` — sole module allowed to import `research_lab`.
- Fail-closed call flow, atomic temp→final publish, mandatory post-run validator, complete manifest + hashes, full Phase 3 output-contract mapping, explicit failure-mode table, and 11 required future tests are specified in `SAFE_ENGINE_ADAPTER_DESIGN_SPEC.md`.

## 6. Risk Register Summary
20 risks (R01–R20). Top blocking-before-implementation: R01 2025/2026 leakage, R02 validation, R03 holdout/WFA, R09 F06-selection ambiguity, R12 validator bypass, R19 scope escalation, R17 raw-data mutation. R09/R10 (engine API + loader ambiguity) are the specific reasons implementation is not yet authorized.

## 7. What Was Not Done
- adapter implementation: NO
- F06 real: NO
- backtest: NO
- validation: NO
- holdout: NO
- 2025/2026: NO
- F06 certification: NO

## 8. Required Conditions Before Adapter Implementation
1. PR #6 still open/draft, no merge, 119/119 reproduced, safe checks green.
2. A dedicated READ-ONLY F06 engine-discovery pass pins: exact F06 strategy module(s), canonical train-data loader contract, exact `EngineConfig` fields for Phase 3 (no engine execution).
3. Claude design audit approves this spec + risk register.
4. Explicit written authorization to implement (implementation ≠ real F06 run; real run is a further separate gate).

## 9. Decision
READY_FOR_CLAUDE_DESIGN_AUDIT

## 10. Copy-Paste Summary for ChatGPT
SAFE ENGINE ADAPTER DESIGN-ONLY complete on PR #6 (head d988c03a).
- Phase A re-confirmed: 119/119 PASS; validate_config PASS; dry_run DRY_RUN_SCHEMA_VALIDATED;
  bad emit BLOCKED_FORBIDDEN_OUTPUT_PATH; run_phase3 BLOCKED_MISSING_EXPLICIT_REAL_RUN_CONFIRMATION.
- Engine inventoried read-only: canonical = repo-root `research_lab/engine.py::run_backtest`
  (engine does not load data nor enforce governance; adapter owns all governance).
- Delivered design docs: ENGINE_INVENTORY, DESIGN_SPEC, RISK_REGISTER (20 risks),
  NEXT_PROMPT (future impl), this DESIGN_ONLY_REPORT.
- NOT done: adapter implementation, real F06, backtest, validation, holdout, 2025/2026, certification.
- Decision: READY_FOR_CLAUDE_DESIGN_AUDIT. Implementation still gated (needs read-only F06
  engine-discovery + design audit + explicit authorization). Implementation ≠ real run.
