# NEXT PROMPT: FIX TP-01 METRIC INCONSISTENCY (SHARED METRIC / EQUITY LAYER)

Actuá como Claude Opus 4.7 Max en modo **Metric Integrity Remediation Engineer** senior, backtest correctness auditor, PnL accounting specialist y quant gatekeeper.

============================================================
CONTEXTO
============================================================

La auditoría externa `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/TP01_FORMAL_DOSSIER_EXTERNAL_AUDIT_REPORT.md`
bloqueó el dossier `TP01_FORMAL_RERUN_20260516_212500` con estado
`TP01_FORMAL_DOSSIER_BLOCKED_METRIC_INCONSISTENCY`.

Se detectaron **dos defectos independientes en la capa COMPARTIDA de métricas/equity**
(afectan a TODA estrategia corrida por el harness, no sólo TP-01):

**DEFECTO 1 — Equity curve desacoplada del ledger de trades.**
- `local_outputs_do_not_commit/base/equity_curve.csv`: 728,996 filas, monótona no-decreciente,
  `first=100000.00`, `last=235710.51`, `min=100000.00`, `max=235710.51`,
  columna `drawdown_pct` enteramente `0.0`.
- El mismo run en `trades.csv` suma **−13.06R / −$9,346.31** (PF<1, expectancy −0.0684R).
- Reportado `+135.71% / $235,710.51 / 1.32% maxDD` es un artefacto del bug, no performance real.
- Los 3 cost profiles dan ≈+135.5% aunque PF/expectancy sí difieren → la curva de equity
  NO se deriva de los trades que dice resumir.

**DEFECTO 2 — Inversión de signo direccional en el ledger.**
- 94/191 trades (49.2%): signo de `pnl_r` opuesto a la dirección real de precio.
- 63 trades con `exit_reason=stop_loss` etiquetados `result=win`;
  17 trades con `exit_reason=take_profit` etiquetados `result=loss` (lógicamente imposible).
- Ejemplo: 2015-01-05 `short`, entry 1.19104 → exit 1.192578875 (precio sube contra un short = pérdida),
  `exit_reason=stop_loss`, registrado como `result=win, pnl_r=+0.987`.

**ANOMALÍA 3 — `stress` == `conservative` byte a byte** (los multiplicadores stress no diferencian).

============================================================
REGLAS ABSOLUTAS
============================================================

NO correr backtest nuevo.
NO strategy run / optimization / sweep / validation / WFA.
NO holdout. NO 2025/2026. NO news. NO high precision.
NO tocar la estrategia `tp01_london_ny_momentum_pullback.py` (la lógica de señal NO está en duda).
NO tocar data cruda ni el vault.
NO commitear `local_outputs_do_not_commit/`.
SOLO corregir la capa de cálculo de métricas/equity y su test de reconciliación.

============================================================
PUNTOS DE REMEDIACIÓN REQUERIDOS
============================================================

1. **Localizar la capa de métricas/equity** (candidatos: `03_RESEARCH_LAB/research_lab/engine.py`,
   `03_RESEARCH_LAB/research_lab/report.py`). Identificar exactamente dónde se construyen:
   - `equity_curve` (serie temporal de equity)
   - `drawdown_pct`
   - `total_return_pct` / `ending_equity`
   - `max_dd_pct`
   - el signo de `pnl_r` / `pnl_usd` y la etiqueta `result` por trade.

2. **DEFECTO 1 — Reconstruir equity desde el ledger.**
   - La equity DEBE derivarse trade-a-trade desde `pnl_usd` (o `pnl_r`×riesgo) en orden cronológico.
   - `drawdown_pct` DEBE poblarse (peak-to-trough sobre la equity reconstruida).
   - `max_dd_pct` DEBE ser ≥ el peor DD anual (los años 2015–2018 muestran 4.8–9.5%).
   - Verificación obligatoria: `ending_equity ≈ 100000 + Σ pnl_usd` (additive) y la variante
     compounded a 0.5%R deben coincidir con el ledger (esperado ≈ −6% a −9%, NO +135%).

3. **DEFECTO 2 — Corregir el signo direccional.**
   - Auditar el cálculo de PnL para `short` (y revalidar `long`).
   - Invariante a forzar y testear: `exit_reason==stop_loss ⇒ pnl_r<0 ⇒ result==loss`;
     `exit_reason==take_profit ⇒ pnl_r>0 ⇒ result==win`.
   - 0 trades pueden tener `(result=win, exit_reason=stop_loss)` ni `(result=loss, exit_reason=take_profit)`.
   - NO "negar" el ledger viejo: recalcular correctamente desde precios de entrada/salida y dirección.

4. **ANOMALÍA 3 — stress vs conservative.**
   - Confirmar por qué `stress` produce salida idéntica a `conservative`; corregir para que
     `stress_spread_multiplier` / `stress_slippage_multiplier` se apliquen.

5. **Gate de reconciliación (nuevo test obligatorio).**
   - Agregar un test que, dado un `trades.csv`, recompute de forma independiente
     `Σpnl_r`, `PF`, `expectancy`, `ending_equity`, `max_dd` y falle si el dossier
     no reconcilia dentro de tolerancia estricta.
   - Ningún dossier puede sellarse (`*_SUCCESS_AND_SEALED`) hasta pasar este gate.

============================================================
CRITERIO DE SALIDA
============================================================

- Crear `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/TP01_METRIC_FIX_REPORT.md` con:
  estado (`TP01_METRIC_FIX_DONE` | `TP01_METRIC_FIX_BLOCKED`), diff conceptual de la causa raíz
  de cada defecto, evidencia de reconciliación post-fix, y confirmación de que NINGÚN
  resultado de edge previo (TP-01 u otra estrategia) puede aceptarse hasta re-ejecutar bajo el fix.
- NO re-ejecutar la corrida formal en esta fase: sólo corregir y reconciliar offline contra
  los `trades.csv` existentes.
- Stage explícito sólo de docs/código de métricas + su test. NO `git add .`.
  NO data pesada, NO ZIP, NO outputs pesados.

============================================================
NOTA DE GOBERNANZA
============================================================

El defecto vive en la capa COMPARTIDA: todo dossier histórico producido por este harness
queda **suspect** hasta que el fix esté mergeado y reconciliado. MR-01 y cualquier otro
backtest formal quedan **bloqueados** hasta entonces (ver
`NEXT_PROMPT_FORMAL_TRAIN_ONLY_MR01_BACKTEST.md`).
