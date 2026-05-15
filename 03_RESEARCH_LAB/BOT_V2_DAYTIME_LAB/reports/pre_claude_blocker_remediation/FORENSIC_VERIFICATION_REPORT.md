# FORENSIC VERIFICATION REPORT

**Fecha:** 2026-05-15
**Modo:** READ-ONLY (no se modificó evidencia; solo lectura y conteo)
**Propósito:** Verificar independientemente los hallazgos de CLAUDE EXTREME AUDIT antes de la auditoría nocturna.

---

## 1. Status

**CLAUDE_FINDINGS_CONFIRMED**

Todos los hallazgos materiales del audit extremo se confirmaron con comandos
read-only sobre los archivos reales. Varios resultaron **peores** de lo
estimado.

## 2. Executive Summary

- El ledger "certificado" `V50B_RERUN_TRADES.csv` está **91.7% compuesto por
  los dos RunIDs explícitamente cuarentenados** (`bfe49625` + `24bb295d`).
  La contaminación Multi-RunID **no** fue remediada — fue acumulada.
- El RunID canónico `68fa2280` aporta **1.700 filas totales** (todas las
  familias/configs/meses combinados); la muestra efectiva por config de F06
  en Cost Hardening es **N=10**.
- `V50B_RERUN_MASTER_RANKING.csv`: **150 filas de config, solo 6 tuplas de
  resultado únicas**. Sweep degenerado confirmado.
- Columnas de validación (`N_val, PF_val, Total_R_val, WR_val, val_pass,
  combined_pass`) **presentes y pobladas** pese a atestaciones
  `validation_touched: NO`.
- Script generador `v50b_limited_rerun_ultra.py`: **NO encontrado** en todo
  el árbol. Reproducibilidad rota.
- `QUARANTINED_DO_NOT_USE.md` **sigue presente** en el directorio que
  contiene la fuente de verdad. Cuarentena nunca levantada.
- Cost model: solo `slippage` + `comm`; **sin componente de spread**.
  STRESS_COMBO no es worst-case institucional.
- Git: `main` local **42 commits adelante** de `origin/main`; rama de
  entrega es **1 commit squash**; 5 árboles `BOT_V2_DAYTIME_LAB`; data
  pesada/holdout-period en historia.

## 3. Git Findings

| Item | Valor |
| :--- | :--- |
| Current branch | `research/v50b-cost-hardening-clean-20260515` |
| HEAD | `28f225c16740ac4ed70697649027076a4053ecd1` |
| origin/main | `bd32a412` |
| main vs origin/main | **0 / 42** (local main 42 commits adelante de origin) |
| Rama de entrega | 1 commit (`28f225c1 "... clean history"`) sobre `bd32a412` — lineage de remediación colapsado |
| Remote | `https://github.com/alerasnik97-tech/bottrading.git` |
| Untracked clave | `COST_HARDENING_BEST_CONFIGS.csv`, `…/v50b_limited_real_gauntlet_rerun_sw/trades/` (incluye el TRADES.csv — **SoT no auditable desde GitHub**) |
| Árboles `BOT_V2_DAYTIME_LAB` | **5**: `\BOT_V2_DAYTIME_LAB`, `\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB`, `\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB` (recursivo), `\BOT_V2_DAYTIME_LAB\BOT_V2_DAYTIME_LAB`, `\BOT_V2_DAYTIME_LAB\src\BOT_V2_DAYTIME_LAB` |
| Tracked sensibles | `.zip` = 1, `.lock` = 2, paths con `tick` = 936 |
| Pesados en historia (audit object scan) | `GIT_BACKUP_*.bundle.lock` 396 MB, `bot2_v2_s01_orb_v1.json` 331 MB, `tick_export_2015_11_UTC.csv` 298 MB (raw tick), `000_PARA_CHATGPT.zip` 226 MB, `EURUSD_M3_{BID,ASK,SPREAD}_2020_2026.csv`, `processed_2015_2019` M1 |
| Pesados en árbol HEAD | `000_PARA_CHATGPT.zip` (226 MB), `EURUSD_M3_DATA_QUALITY_MASK_2020_2026.csv` (período holdout) |

## 4. RunID / Ledger Findings

`V50B_RERUN_TRADES.csv` — **42.727 filas**, distribución por `run_id`:

| run_id | filas | clasificación |
| :--- | ---: | :--- |
| `bfe49625` | **37.463** | **CUARENTENADO (contaminado)** |
| `68fa2280` | 1.700 | "canónico" (minoritario) |
| `1fa40f18` | 1.700 | ABORTED/PREV |
| `24bb295d` | **1.699** | **CUARENTENADO (contaminado)** |
| `aeb2f02d` | 55 | preflight-class |
| `e0897fd3` | 55 | preflight-class |
| `129e106b` | 55 | PREFLIGHT |

- RunIDs cuarentenados (`bfe49625` + `24bb295d`) = **39.162 / 42.727 = 91.7%**
  del archivo usado como input del Cost Hardening.
- El archivo es exactamente uno de los 4 nombrados como contaminados en
  `QUARANTINED_DO_NOT_USE.md`. La remediación "Single-Writer / output
  isolation: SUCCESS" es **factualmente falsa**.
- 68fa2280 = 1.700 filas (todas familias/configs/meses). El Cost Hardening
  de F06 corre sobre **N=10** por config → certificación sobre muestra
  trivial confirmada.
- Meses presentes: 2020-03, 2021-08, 2022-05, 2023-01, 2024-04 (5 meses
  cherry-picked). **No aparecen 2025 ni 2026 a nivel de trade** (consistente
  con holdout no tocado en el ledger; ver §10).

## 5. Ranking Degeneracy Findings

`V50B_RERUN_MASTER_RANKING.csv`:

- 150 filas de config (F06=50, F08=50, F12=50).
- **Solo 6 tuplas únicas** de `(family_id,N_train,PF_train,Total_R_train,
  WR_train,N_val,PF_val,Total_R_val,WR_val)`.
- ⇒ ~98% de las filas son duplicados. El "sweep de 50 configs por familia"
  no testea parámetros: produce 1–3 resultados por familia repetidos ~50×.
- No puede sostener ninguna afirmación de "robustez ante configuraciones".

## 6. Validation Column Findings

- Columnas presentes: `N_val, PF_val, Total_R_val, WR_val, val_pass,
  combined_pass` — **confirmadas y pobladas** (`val_pass=True`,
  `combined_pass=True`).
- Contradicción directa con `validation_touched: NO` en
  `FULL_RERUN_TRAIN_ONLY_REPORT.md`, `EVIDENCE_RECONCILIATION_REPORT.md` y
  `COST_HARDENING_REPORT.md`.
- Clasificación: **atestación de seguridad inválida**. Requiere determinar
  qué período es "val" antes de cualquier afirmación de no-leakage. Si "val"
  toca 2025 → leakage de holdout. Bloqueante en cualquier caso.

## 7. Script Reproducibility Findings

- `v50b_limited_rerun_ultra.py` (script generador declarado en
  `FULL_RERUN_TRAIN_ONLY_REPORT.md §2`): **NO encontrado** en
  `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo` (búsqueda recursiva).
- La evidencia canónica fue producida por un script ausente/no trackeado.
  Reproducibilidad = 0. Combinado con la degeneración del ranking, sugiere
  que el ranking no proviene de un loop de backtest por config real.

## 8. Quarantine Findings

- `…/v50b_limited_real_gauntlet_rerun_sw/QUARANTINED_DO_NOT_USE.md` →
  **presente (True)**.
- Contenido: contaminación Multi-RunID, RunIDs `24bb295d` y `bfe49625`,
  archivos contaminados incl. `V50B_RERUN_TRADES.csv`, estado **BLOQUEADO**,
  sin marcador de levantamiento.
- La fuente de verdad declarada (`results/V50B_RERUN_MASTER_RANKING.csv`,
  `trades/V50B_RERUN_TRADES.csv`) vive **dentro** de ese directorio
  cuarentenado. Incompatible con certificación institucional.

## 9. Cost Hardening Findings

`v50b_cost_hardening_stress.py`:

- **Input exacto:** `reports/v50b_limited_real_gauntlet_rerun_sw/trades/
  V50B_RERUN_TRADES.csv` (archivo cuarentenado, untracked, 91.7%
  contaminado). Filtra `run_id == "68fa2280"`.
- **Trades por familia (F06):** N=10 por config (confirmado en
  `COST_HARDENING_BY_SCENARIO.csv`).
- **Spread:** **NO modelado** (ninguna clave `spread` en el script ni en el
  src del lab; solo `slippage` y `comm`).
- **Modelo:** `slip_r = slippage/sl_pips`; `comm_r = comm/(sl_pips*10)`
  (comisión one-way, pip value/lote hardcodeados). Resta determinista sobre
  `gross_r` precalculado. `if sl_pips<=0: return gross_r` (costo cero
  silencioso).
- **STRESS_COMBO (1.0 pip + $10/lot, spread 0):** NO es worst-case
  institucional.
- F06 "COST_ROBUST" se basa en **N=10** de un archivo 91.7% contaminado.

## 10. Safety Verification

| Gate | Estado verificado |
| :--- | :--- |
| test_touched | **NO** (sin 2025/2026 a nivel de trade en el ledger) |
| validation_touched | **SÍ / AMBIGUO** — columnas `_val` pobladas y `val_pass` evaluado; contradice la atestación. Bloqueante hasta aclarar el período "val". |
| holdout_touched | **NO confirmado limpio** — sin 2025/2026 en trades, pero data período-holdout (M3 2020-2026, quality mask) presente en repo/historia (no air-gapped) |
| raw_data_mutated | **NO** (esta verificación fue read-only; no se mutó nada) |
| sweep_run | **NO** (no se ejecutó sweep en esta verificación) |
| optimization_run | **NO** (no se ejecutó optimización en esta verificación) |

Nota: esta sección certifica que **la verificación forense** no tocó nada
prohibido. NO certifica que la evidencia histórica V50B respetó esos gates;
de hecho la columna `val` y la contaminación demuestran lo contrario.

## 11. Decision

**BLOCKED_MAJOR_RISK_CONFIRMED**

Existe evidencia real y reproducible de: (a) contaminación Multi-RunID no
remediada (91.7%), (b) muestra de certificación trivial (N=10), (c) ranking
degenerado, (d) columnas de validación contradiciendo atestaciones, (e)
script generador ausente, (f) cuarentena no levantada, (g) cost model sin
spread. Cualquiera de (a)–(g) bastaría para bloquear; en conjunto invalidan
la evidencia V50B/F06 por completo.

No se reinstala ninguna certificación previa. No se ejecuta re-corrida
correctiva. Se procede a supersede formal y plan de reconstrucción.
