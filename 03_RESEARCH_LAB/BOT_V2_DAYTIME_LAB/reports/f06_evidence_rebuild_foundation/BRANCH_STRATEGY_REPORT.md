# BRANCH STRATEGY REPORT

**Fecha:** 2026-05-15
**Fase:** 2 — F06 Evidence Rebuild Foundation (scaffold only, no strategy run)

---

## 1. Branches

| | Valor |
| :--- | :--- |
| Base branch | `research/pre-claude-blocker-remediation-20260515` (PR #3 freeze pack) |
| Base commit | `d09aada1` (= `origin/main` + freeze pack; ya pusheado, sync 0/0) |
| New branch | `research/f06-evidence-rebuild-foundation-20260515` |
| New branch HEAD (inicio) | `d09aada1` (idéntico a base hasta el commit de foundation) |
| Relación con PR #3 | Hija del freeze pack. PR #3 NO se mergea ni se cierra. |

## 2. Reason

La Fase 2 construye el esqueleto fail-closed que hace imposible repetir los
blockers detectados (multi-RunID, ledger mezclado, SoT cuarentenado, ranking
degenerado, columnas validation en train-only, script ausente, manifest
incompleto, N=10, cost model sin spread). Se ramifica **desde el freeze pack**
para que el trabajo herede el estado institucional BLOCKED y la trazabilidad
de PR #3, sin tocar la evidencia inválida.

## 3. Por qué NO se usa `main` local

La verificación forense (Fase 1) confirmó que `main` local está **42 commits
adelante de `origin/main`** (contenido no revisado/no pusheado) y que la
evidencia addendada vive en la rama publicada, no en `main`. Ramificar desde
`main` local: (a) publicaría 42 commits no auditados, (b) rompería la cadena
con PR #3. Por eso la base es la rama publicada del freeze pack.

## 4. Reglas de trabajo respetadas

- No trabajar directo sobre `main`.
- No mergear ni cerrar PR #3.
- No `--force`, no history rewrite.
- Push solo de scaffold liviano (sin raw/tick/parquet/zip/lock/heavy).
- Fixtures 100% sintéticos.

## 5. Safety Status

| Gate | Estado |
| :--- | :--- |
| active_research_process | NONE (precheck: 0 procesos python) |
| strategy_run | NO |
| backtest_run | NO |
| validation_touched | NO |
| holdout_touched | NO |
| raw_data_mutated | NO |
| 2025_2026_touched | NO |
| quarantined_input_used | NO |
| scope_escalation | NONE |

Estado: **FOUNDATION_BRANCH_READY** — proceder a construir el scaffold.
