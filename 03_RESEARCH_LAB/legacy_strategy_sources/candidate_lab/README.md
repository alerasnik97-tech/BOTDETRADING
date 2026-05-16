# Institutional Research Candidate Lab

Lab aislado de research institucional para `SCBI_M5_GLOBAL`.

Estado:
- `RESEARCH_ONLY`
- `NO_PRODUCTION`

Objetivo:
- replicar exactamente la baseline real actual del runner productivo;
- generar variantes objetivas y comparables;
- medir robustez multi-anual;
- producir un dossier claro para futura shadow line;
- no tocar el core productivo.

## Estructura

```text
institutional_research_candidate_lab/
  __init__.py
  config.py
  data_io.py
  baseline_truth_model.py
  candidate_matrix.py
  ranking.py
  reporting.py
  orchestrator.py
  README.md
  outputs/
  tests/
```

Scripts CLI en la raiz del proyecto:
- `run_candidate_baseline.py`
- `run_candidate_matrix.py`
- `summarize_candidate_research.py`

## Baseline exacta replicada

- Instrumento: `EURUSD`
- H1 para sweep y niveles
- M5 para confirmacion, entrada y salida
- Niveles: `PDH/PDL`, `Asia H/L`, `London H/L`
- Long: `low < nivel` y `close > nivel`
- Short: `high > nivel` y `close < nivel`
- Confirmacion M5 baseline: `+1h a +2h`, primera vela cuyo `close` queda del lado correcto del nivel
- Entrada long: `next_open + 0.3 pips`
- Entrada short: `next_open`
- Riesgo minimo: `2.0 pips`
- SL: extremo del sweep `+-1 pip`
- TP: `1.5R`
- Timeout: `4 horas`
- Maximo: `1 trade por dia`
- Noticias: filtro simplificado alrededor del `sweep_time`

No agrega:
- `CHOCH`
- `FVG`
- `BE`
- trailing
- filtros ATR o de regimen

## Variantes de research

Gestion:
- `TP`: `1.0R`, `1.25R`, `1.5R`, `1.75R`, `2.0R`
- `timeout`: `2h`, `4h`, `6h`
- `SL buffer`: `0.5`, `1.0`, `1.5 pips`
- `long entry buffer`: `0.0`, `0.3`, `0.5 pips`

Confirmacion:
- ventanas `+0h a +1h`, `+1h a +2h`, `+0h a +2h`
- primera confirmacion vs mejor confirmacion
- reclaim simple vs reclaim con cuerpo minimo M5

Niveles:
- todos
- solo `PD`
- solo `Asia`
- solo `London`

Noticias:
- sin filtro
- `+-15m`
- `+-30m`
- `+-60m`

## Outputs principales

Dentro de `institutional_research_candidate_lab/outputs/`:
- `baseline_summary.json`
- `baseline_trades.csv`
- `baseline_sweep_audit.csv`
- `research_matrix_results.csv`
- `research_top_variants.csv`
- `research_baseline_vs_variants.md`
- `research_summary.json`
- `candidate_dossier.md`
- `shadow_candidate_spec.md`

## Como correrlo

Baseline:

```powershell
python C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\run_candidate_baseline.py
```

Matriz:

```powershell
python C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\run_candidate_matrix.py --profile axis_scan
```

Resumen regenerado:

```powershell
python C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\summarize_candidate_research.py
```

## Que NO toca

No modifica:
- `scratch/run_scbi_forward_phase1.py`
- `scratch/run_scbi_phase1_autopilot.py`
- validators
- promotion
- chain
- bundle builder
- monitoring layer
- `operational_analytics.py`
- datasets canonicos productivos

Solo lee datasets existentes y escribe artefactos nuevos dentro del lab aislado.
