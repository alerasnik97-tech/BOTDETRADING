# NEXT PROMPT: EXTERNAL PRE-EXECUTION AUDIT V2 OF THE FORMAL RUNNER EXECUTE-PATH FIX (GATED)

Actuá como Claude Opus 4.7 Max en modo **auditor institucional externo de
infraestructura cuantitativa**, compuesto por:

1. Formal Backtest Runner Auditor.
2. Python Signature Compatibility Auditor.
3. Artifact Persistence Auditor.
4. Data Leakage Prevention Officer.
5. Metric Reconciliation Gatekeeper.
6. Git Safety Auditor.
7. Output Policy Auditor.
8. Pre-Execution Quant Gatekeeper.

Calidad máxima, seguridad total, agilidad inteligente, cero autoengaño, cero
narrativa sin evidencia. Es una auditoría **externa e independiente** del fix.

============================================================
CONTEXTO
============================================================

Auditoría v1: `FORMAL_RUNNER_PRE_EXECUTION_EXTERNAL_AUDIT_REPORT.md`
→ `FORMAL_RUNNER_BLOCKED_EXECUTE_SIGNATURE_RISK`.

Fix aplicado: `FORMAL_RUNNER_EXECUTE_PATH_FIX_REPORT.md`
→ `FORMAL_RUNNER_EXECUTE_PATH_FIX_DONE_READY_FOR_AUDIT_V2`.

Repo: `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo`
Branch fix: `fix/formal-runner-execute-path-20260517` (NO main).
Archivos del fix:
- `03_RESEARCH_LAB/research_lab/runners/formal_train_runner.py`
- `03_RESEARCH_LAB/research_lab/tests/test_formal_train_runner_execute_contract.py`
- `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/FORMAL_RUNNER_EXECUTE_PATH_FIX_REPORT.md`

Fix declara cerrados: B1 `run_backtest`, B2 `summarize_result`, B3 `data_dirs`,
B4 artifact-write, W2 git identity real, W3 cost-profile reconciliation en seal.
Tests reportados: 110/110. No real backtest/data.

============================================================
OBJETIVO
============================================================

Auditar externamente el fix y **decidir** si se autoriza, por fin, la
regeneración real de TP-01 train-only 2015–2024 con el runner oficial — o si
queda bloqueado de nuevo.

NO ejecutar backtest real. NO strategy run real. NO TP-01 real. NO MR-01.
NO optimization/sweep/validation/holdout. NO 2025/2026. NO news. NO high precision.
NO modificar código/tests/data. Solo: leer, correr tests seguros, dry-run,
execute=True **solo con fakes**, static scan, reporte, decisión, prompt siguiente.

============================================================
REGLAS ABSOLUTAS
============================================================

NO main. NO force push. NO merge. NO rebase. NO git add .
NO backtest/strategy/TP-01/MR-01 real. NO optimization/sweep/validation/walk-forward.
NO holdout/sealed_holdout. NO 2025/2026. NO F06. NO news. NO high precision.
NO modificar runner/tests/data. NO ZIP. NO outputs pesados reales. NO root files.
NO commitear scratch/local_outputs_do_not_commit. NO tocar dirty preexistente ajeno.
NO declarar edge/rentable. NO autorizar FTMO/demo/real.
Permitido: leer, correr tests seguros, dry-run/preflight, execute=True con fakes
existentes, static scan, escribir doc de auditoría v2 + próximo prompt, commit/push
SOLO docs de auditoría.

============================================================
BLOQUES
============================================================

**B0 Precheck.** git status/branch/HEAD; `git fetch --prune`; procesos python
(si backtest/sweep activo → `BLOCKED_ACTIVE_RESEARCH_PROCESS_DETECTED`). Crear
`audit/formal-runner-execute-path-fix-v2-20260517` desde la branch del fix.

**B1 Surface.** `git show --stat` del/los commits del fix: solo runner +
test_formal_train_runner_execute_contract.py + docs. Si aparece engine/strategy/
data/CSV/parquet/ZIP/scratch/root → `AUDIT_V2_BLOCKED_COMMIT_SURFACE`.

**B2 Re-verificar firmas REALES (no asumir).** Leer en código:
`engine.run_backtest` (engine.py:597-609), `report.summarize_result`
(report.py:281-293), `data_loader.load_backtest_data_bundle` (data_loader.py:
359-368), `config.INITIAL_CAPITAL`, rama no-news `engine.py:687-696`,
`metric_reconciliation.reconcile_all`. Confirmar que el runner llama EXACTO:
- `run_backtest(..., news_block, news_filter_used=False)` con `news_block`
  booleano all-False de largo `len(frame)`;
- `summarize_result(..., news_filter_used=False, initial_capital, selected_score=None)`;
- `data_dirs=[Path(...)]`;
- seal con `profiles=` (W3) y manifest sin placeholders (W2).
Cualquier mismatch → `AUDIT_V2_BLOCKED_SIGNATURE_RISK`.

**B3 Import safety.** Subproceso: importar el runner NO debe traer
engine/data_loader/strategies/report/numpy a `sys.modules`. `__main__` guard.

**B4 Artifact policy.** Revisar `write_run_artifacts`: todo bajo `req.output_dir`
validado; heavy solo en `local_outputs_do_not_commit/<profile>`; ZIP rechazado;
nada en dry-run/fallo. Confirmar escritura SOLO post-seal.

**B5 Git identity.** `get_git_identity`: list argv, sin `shell=True`, timeout,
Git no modificado; rechaza placeholders; fail-closed si no hay git. Manifest
real en dry-run y en execute (fakes).

**B6 Tests seguros.** Correr y exigir verde:
`test_formal_train_runner_execute_contract.py` · `test_formal_train_runner_contract.py`
· `test_cost_profiles.py` · `test_metric_reconciliation.py` · `test_engine.py`
· `test_engine_stop_entry.py` · `test_lab_preflight*.py`. Falla →
`AUDIT_V2_BLOCKED_TEST_FAILURE`. Verificar que el suite de execute usa fakes
(sin data real, sin backtest real) y que cubre: 3 profiles, artifact writes,
reconciliation-failure-blocks-seal, manifest branch/commit real, mislabel bloquea.

**B7 Dry-run / CLI fail-closed.** Dry-run sin `--execute` (exit 0, sin dir
creado, manifest con branch/commit reales). Matriz 8/8:
`--holdout/--validation/--optimize/--sweep/--high-precision/--news`,
`--end 2025-01-01`, output `.zip` → exit 2. **NO `--execute` contra data real.**

**B8 Execute con fakes (opcional).** Si hace falta, ejercer `execute=True`
ÚNICAMENTE vía los fakes existentes (sys.modules injection del test), nunca
contra el dataset real. Confirmar dossier escrito en tempdir y gate efectivo.

**B9 Static scan.** Tokens incl. `shell=True|subprocess|forex_factory|news|
holdout|2025|2026|zip|scratch|git add .|local_outputs_do_not_commit`. Clasificar.
Uso peligroso real → `AUDIT_V2_BLOCKED_STATIC_SCAN_SAFETY_VIOLATION`.

**B10 Reporte.** Crear
`06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/FORMAL_RUNNER_EXECUTE_PATH_FIX_EXTERNAL_AUDIT_V2_REPORT.md`
con STATUS:
- `EXECUTE_PATH_FIX_APPROVED_FOR_TP01_REGENERATION`
- `EXECUTE_PATH_FIX_APPROVED_WITH_WARNINGS_FOR_TP01_REGENERATION`
- `EXECUTE_PATH_FIX_BLOCKED_SIGNATURE_RISK`
- `EXECUTE_PATH_FIX_BLOCKED_ARTIFACT_POLICY`
- `EXECUTE_PATH_FIX_BLOCKED_SAFETY_RISK`
- `EXECUTE_PATH_FIX_BLOCKED_TEST_FAILURE`
Secciones: surface, signatures, import safety, artifact policy, git identity,
tests, dry-run/CLI, static scan, warnings, decision, safety verification
(real_backtest NO / fake_execute_only YES / data_modified NO / code_modified NO /
force_push NO / git_add_dot NO), copy-paste summary.

**B11 Próximo prompt.**
- Si APPROVED(+/-warnings): crear
  `NEXT_PROMPT_REGENERATE_TP01_WITH_OFFICIAL_RUNNER_EXECUTE_AUDITED.md`
  (regenerar TP-01 train-only 2015–2024, 3 profiles reales, dataset
  `prepared_train_2015_2024`, `execute=True` una sola vez, gate obligatorio,
  heavy a `local_outputs_do_not_commit`, comparar pre/post, MR-01 sigue
  bloqueado hasta TP-01 gate-green + auditado).
- Si BLOCKED: crear
  `NEXT_PROMPT_FIX_FORMAL_RUNNER_EXECUTE_PATH_V2_BLOCKERS.md`.

**B12 Git.** Stage explícito SOLO docs de auditoría v2 (+ próximo prompt). NO
code/tests/data/heavy/scratch/ZIP/root. `git add` por archivo, `git diff --cached
--name-only`, commit `docs: external audit v2 of formal runner execute-path fix`,
push `audit/formal-runner-execute-path-fix-v2-20260517`.

============================================================
CRITERIO DE APROBACIÓN
============================================================

Aprobar TP-01 regeneration solo si: firmas reales correctas (B1/B2/B3),
import-safe, dry-run seguro, CLI fail-closed 8/8, artifacts solo bajo output
validado y solo post-seal, git identity real fail-closed, W3 activo, tests
110/110 verdes con execute cubierto por fakes, static scan limpio, sin código/
data modificados por la auditoría, y queda prompt seguro de regeneración.
MR-01 permanece bloqueado hasta TP-01 regenerado limpio, gate-green y auditado.
