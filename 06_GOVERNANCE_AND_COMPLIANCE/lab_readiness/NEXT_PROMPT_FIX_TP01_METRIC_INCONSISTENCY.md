# NEXT PROMPT: FIX TP-01 METRIC INCONSISTENCY (SHARED ENGINE METRIC / COST LAYER)

Actuá como Claude Opus 4.7 Max en modo **Metric Integrity Remediation Engineer** senior,
backtest correctness auditor, PnL accounting specialist y quant gatekeeper.

============================================================
CONTEXTO
============================================================

La auditoría externa v2
`06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/TP01_FORMAL_DOSSIER_EXTERNAL_AUDIT_REPORT.md`
bloqueó el dossier `TP01_FORMAL_RERUN_20260516_212500`:
estado `TP01_FORMAL_DOSSIER_BLOCKED_METRIC_INCONSISTENCY`.

Se confirmaron **CUATRO defectos en la capa COMPARTIDA de métricas/costos del engine**
(afectan a TODA estrategia del harness, no sólo TP-01):

**DEFECTO 1 — Equity curve desacoplada del ledger.**
`equity_curve.csv` sube de $100,000.00 a $235,710.51, su `min` nunca baja del start,
sólo 272 micro-bajadas en 728,996 filas, columna `drawdown_pct` enteramente `0.0`,
mientras `trades.csv` suma **−13.06R / −$9,346.31**. Reportado `+135.71% / 1.32% maxDD`
es artefacto del bug. Equity real ≈ $90,654 (additive) / $93,326 (compounded@0.5%R), maxDD real ≈8.5%.

**DEFECTO 2 — Inversión de signo direccional.**
94/191 (49.2%) trades con signo `pnl_r` opuesto a la dirección de precio.
63 `exit_reason=stop_loss` etiquetados `result=win`; 17 `take_profit` etiquetados `loss`.
Ej: 2015-01-05 `short` entry 1.19104 → exit 1.192578875 (precio sube = pérdida en short),
`stop_loss`, registrado `win, pnl_r=+0.987`.

**DEFECTO 3 — `summary.json` nativo del engine auto-contradictorio.**
`profile_reports/*/summary.json`: PF<1, expectancy<0, `negative_years=3`,
`positive_years=1`, `negative_months=21–22` — y a la vez `total_return_pct≈+135%`,
`max_drawdown_pct≈1.3%`. El bug está en la capa core de métricas, no sólo en el MD.

**DEFECTO 4 — Mislabel/duplicación de cost profiles.**
`base/summary.json` = `{cost_profile:base, execution_mode:normal_mode}`.
`conservative/summary.json` Y `stress/summary.json` = `{cost_profile:stress, execution_mode:conservative_mode}`.
Sólo existen 2 corridas (base + una etiquetada stress/conservative_mode duplicada en
ambas carpetas). No hay corrida conservative real; stress nunca corrió en modo stress.
`RUN_MANIFEST.cost_profiles_run=[base,conservative,stress]` es engañoso.

============================================================
REGLAS ABSOLUTAS
============================================================

NO main. NO force push. NO merge. NO rebase.
NO backtest nuevo / strategy run / optimization / sweep / validation / WFA / holdout.
NO 2025/2026. NO news. NO high precision. NO F06.
NO tocar la lógica de señal de `tp01_london_ny_momentum_pullback.py` (no está en duda).
NO tocar data cruda / vault. NO borrar outputs. NO ZIP. NO git add . NO commitear `local_outputs_do_not_commit/`.
SOLO corregir la capa de métricas/equity/costos del engine + tests de reconciliación.

============================================================
REMEDIACIÓN REQUERIDA
============================================================

1. Localizar la capa de métricas/equity/costos (candidatos: `03_RESEARCH_LAB/research_lab/engine.py`,
   `03_RESEARCH_LAB/research_lab/report.py`). Identificar dónde se construyen
   `equity_curve`, `drawdown_pct`, `total_return_pct`, `ending_equity`, `max_dd_pct`,
   el signo de `pnl_r`/`pnl_usd`/`result`, y el ruteo de `cost_profile`/`execution_mode`.

2. DEFECTO 1+3: reconstruir equity trade-a-trade desde el ledger (orden cronológico),
   poblar `drawdown_pct` real (peak-to-trough), y propagar lo mismo a `summary.json`.
   Invariante: `ending_equity ≈ 100000 + Σ pnl_usd`; `max_dd ≥` peor DD anual (años 4.8–9.5%).
   Esperado post-fix ≈ −6% a −9% (NO +135%).

3. DEFECTO 2: corregir el signo direccional (revisar PnL `short` y revalidar `long`).
   Forzar y testear: `stop_loss ⇒ pnl_r<0 ⇒ result=loss`; `take_profit ⇒ pnl_r>0 ⇒ result=win`.
   0 trades con `(win,stop_loss)` o `(loss,take_profit)`. NO "negar" el ledger viejo:
   recalcular desde entry/exit/dirección.

4. DEFECTO 4: corregir el ruteo de cost profiles. `conservative` debe correr conservative real;
   `stress` debe aplicar `stress_spread_multiplier`/`stress_slippage_multiplier` y modo stress;
   cada carpeta debe contener su propio profile y `summary.json` debe auto-reportar el profile correcto.
   `RUN_MANIFEST` debe reflejar corridas realmente distintas.

5. Gate de reconciliación (test nuevo obligatorio): dado `trades.csv`, recomputar
   `ΣR, PF, expectancy, ending_equity, max_dd` y FALLAR si el dossier/summary no reconcilia
   dentro de tolerancia estricta; validar invariantes de signo y unicidad de cost profiles.
   Ningún dossier puede sellarse (`*_SUCCESS_AND_SEALED`) sin pasar este gate.

============================================================
SALIDA
============================================================

- Crear `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/TP01_METRIC_FIX_REPORT.md`
  (estado `TP01_METRIC_FIX_DONE` | `TP01_METRIC_FIX_BLOCKED`, causa raíz por defecto,
  evidencia de reconciliación post-fix, confirmación de que ningún edge previo es aceptable
  hasta re-ejecutar bajo el fix).
- NO re-ejecutar la corrida formal aquí: sólo corregir y reconciliar offline contra los `trades.csv` existentes.
- Stage explícito sólo de código de métricas/costos + sus tests + el report. NO `git add .`.
  NO data pesada, NO ZIP, NO outputs pesados, NO root files.

============================================================
GOBERNANZA
============================================================

Defecto en capa COMPARTIDA → todo dossier histórico del harness queda **suspect**.
MR-01 y cualquier backtest formal quedan **bloqueados** hasta merge+reconciliación
(ver `NEXT_PROMPT_FORMAL_TRAIN_ONLY_MR01_BACKTEST.md`).
