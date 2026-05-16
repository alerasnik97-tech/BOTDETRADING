# NEXT PROMPT — SAFE ENGINE ADAPTER IMPLEMENTATION (FUTURE, DO NOT RUN NOW)

This is a ready-to-use prompt for the NEXT phase. It is documentation only. It does NOT authorize
any action now. Run it only after a Claude design audit explicitly approves implementation.

> IMPORTANT: Implementing the adapter is NOT the same as running F06 real.
> This future task implements + unit-tests the adapter with the engine still NOT invoked on real data.
> A real F06 run is a SEPARATE, later, independently-gated step.

---

## PROMPT START

Actuá como Claude Code Opus 4.7 Max en modo ingeniero institucional extremo, especialista en adapters
fail-closed hacia motores de backtest, prevención de leakage y reproducibilidad.

OBJETIVO
Implementar (solo código + unit tests) el `phase3_f06_engine_adapter` según
`SAFE_ENGINE_ADAPTER_DESIGN_SPEC.md`, dejándolo conectado al runner pero SIN ejecutar F06 real,
SIN backtest real, SIN tocar validation/holdout/2025/2026.

PROHIBICIONES ABSOLUTAS
- NO ejecutar F06 real. NO backtest real. NO estrategia/optimization/sweep.
- NO validation. NO holdout. NO 2025/2026. NO non-EURUSD. NO non-F06.
- NO tocar `research_lab/` core (solo importarlo desde el adapter).
- NO usar `validation.py`/`wfa.py`/OOS, `*_BACKUP_*`, `reports/canonical_*`, quarantine/v50b tokens, ZIP.
- NO merge, NO force push, NO ready conversion, NO certificar F06.
- Si algo intenta correr estrategia real: ABORT `BLOCKED_SCOPE_ESCALATION_DETECTED`.
- Si algo intenta tocar 2025/2026: ABORT `BLOCKED_TEST_LEAKAGE_RISK`.

PRECONDICIONES (verificar READ-ONLY antes de tocar nada)
- branch `research/f06-clean-train-only-rerun-20260515`, no main, no force.
- suite 119/119 + nuevos tests del adapter.
- safe checks PR #6 siguen verdes.
- design audit del adapter APROBADO.

SCOPE / FILES ALLOWED (crear/editar SOLO)
- `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/pipelines/f06_evidence_rebuild/adapters/__init__.py`
- `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/pipelines/f06_evidence_rebuild/adapters/phase3_f06_engine_adapter.py`
- minimal wiring in `scripts/f06_rebuild_pipeline.py` `cmd_run_phase3` to call the adapter,
  with the real-run path STILL fail-closed behind the explicit confirmation token AND a
  new `--allow-engine-bind` style guard that defaults to OFF (engine not invoked on real data).
- `tests/test_phase3_f06_engine_adapter.py` (+ adversarial cases).
- design reports updated under `reports/f06_clean_train_only_rerun/`.
FORBIDDEN to add: raw/tick/parquet, real trades, zips, locks, caches, __pycache__, .venv,
temp outputs, old outputs, engine core edits.

REQUIRED TESTS (all must pass; engine NOT run on real data)
- blocks 2025; blocks 2026 (pre-load guard, mocked frame).
- blocks validation; blocks holdout; refuses importing `validation.py`/`wfa.py`.
- blocks non-F06 family; blocks non-EURUSD.
- blocks quarantined/old-output tokens; blocks output dir existing; single run_id.
- no partial artifacts on failure (temp-only, atomic publish) — simulated failure injection.
- requires cost report (spread+slippage+round-turn) from a synthetic `BacktestResult`.
- requires hashes matching disk + manifest; manifest field-completeness.
- calls mandatory validator; refuses to interpret on validator failure.
- determinism: seed/mode pinned & recorded.
- run_phase3 WITHOUT confirmation still BLOCKED_MISSING_EXPLICIT_REAL_RUN_CONFIRMATION.
- run_phase3 WITH confirmation but engine-bind OFF → explicit
  `BLOCKED_ENGINE_BIND_DISABLED` (still no real run).

SAFE CHECKS (must keep passing)
- `python -m unittest discover -s 03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/pipelines/f06_evidence_rebuild/tests -p "test_*.py"` → all PASS
- `validate_config` → PASS
- `dry_run` good → DRY_RUN_SCHEMA_VALIDATED
- `dry_run --emit ../MANIFEST.json` → BLOCKED_FORBIDDEN_OUTPUT_PATH
- `run_phase3` without confirmation → BLOCKED_MISSING_EXPLICIT_REAL_RUN_CONFIRMATION
- `run_phase3` with confirmation, engine-bind OFF → BLOCKED_ENGINE_BIND_DISABLED (NO real run)

NO REAL F06 RUN
The adapter is wired and unit-tested with the engine mocked/guarded. The real F06 run remains a
separate future task requiring its own explicit authorization.

OUTPUT EXPECTED
- `phase3_f06_engine_adapter.py` implemented per spec (fail-closed).
- adapter unit + adversarial tests green.
- report `SAFE_ENGINE_ADAPTER_IMPLEMENTATION_REPORT.md`.

GITHUB UPDATE
- commit: `feat: implement fail-closed phase3 F06 engine adapter (no real run)`
- push `origin research/f06-clean-train-only-rerun-20260515` (no force, stays draft, no merge).
- PR #6 comment: adapter implemented, engine NOT run on real data, next gate Claude adapter audit.

FINAL FORMAT
1. STATUS:
2. ADAPTER_IMPLEMENTED: YES/NO
3. ENGINE_RUN_ON_REAL_DATA: NO
4. TESTS: total/passed/failed
5. SAFE_CHECKS: ...
6. SAFETY: real_f06_run NO / backtest NO / validation NO / holdout NO / 2025 NO / 2026 NO / F06_CERTIFIED NO
7. GITHUB: branch/commit/pushed/pr_url
8. DECISION: READY_FOR_CLAUDE_ADAPTER_AUDIT / NEEDS_FIXES / BLOCKED
9. NEXT_STEP: Claude adapter audit (real F06 run still separately gated)

## PROMPT END
