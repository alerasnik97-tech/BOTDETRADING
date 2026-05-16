# PROMPT: OPEN EURUSD TRAIN-ONLY RESEARCH LAB (SMOKE RUN)

Actuá como Codex GPT-5.5 Max en modo Senior Quant Researcher.

OBJETIVO: Ejecutar la primera corrida de investigación (SMOKE RUN) en el laboratorio EURUSD Train-Only recién autorizado.

REGLAS:
- Solo modo TRAIN-ONLY (2015-2024).
- No tocar HOLDOUT.
- No tocar NEWS.
- Generar evidencias según LAB_OUTPUT_EVIDENCE_CONTRACT.md.

ACCIONES:

1. CONFIGURACIÓN:
- Usar estrategia: `asia_london_sweep_reversion_pm` (Canónica).
- Configuración de motor: `EngineConfig(pair='EURUSD', risk_pct=0.5, cost_profile='normal_mode')`.

2. EJECUCIÓN (SMOKE):
$env:PYTHONPATH="03_RESEARCH_LAB"
python -m research_lab.research_runner --strategy asia_london_sweep_reversion_pm --pair EURUSD --mode TRAIN-ONLY --output-dir 03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/smoke_run_20260516

3. VERIFICACIÓN:
- Verificar que se genere `trades.csv` y `manifest.json`.
- Validar que no haya leaks temporales (min_date >= 2015, max_date <= 2024).

4. HANDOFF:
Reportar resultados (N trades, PF, WR) y ubicación de evidencias.
