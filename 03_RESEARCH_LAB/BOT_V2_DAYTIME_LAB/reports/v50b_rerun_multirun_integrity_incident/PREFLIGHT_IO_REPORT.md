# PREFLIGHT IO REPORT - 2026-05-15

## 1. Status
**PREFLIGHT_PASS**

## 2. Command Run
`$env:PYTHONPATH="03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB"; python 03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/v50b_limited_real_gauntlet_rerun_sw/scripts/v50b_limited_rerun_single_writer_runner.py preflight_io`

## 3. ZIP Verification
- **Path**: `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\000_PARA_CHATGPT.zip`
- **Size**: 11,856 bytes (Initial state)
- **SHA256**: `504DA9A34B1D3F281E0515956DEFAD08D40D0FAE5596BAC63780F99C00C8966D`
- **Testzip**: SUCCESS (Verified prior to preflight)

## 4. Single-Writer Verification
- **Lock Atómico**: VALIDADO. El runner adquiere y libera `V50B_RERUN.lock` correctamente usando `os.O_EXCL`.
- **RunID Único**: VALIDADO. Se asignó el ID `e0897fd3`.
- **Aislamiento de Salida**: VALIDADO. Los archivos se escribieron primero en `runs/e0897fd3/`.
- **Paso de Publicación**: VALIDADO. El runner añadió correctamente los resultados a los archivos maestros (`V50B_RERUN_REJECTION_AUDIT.csv`, etc.) al finalizar.
- **Manifest**: VALIDADO. Se generó `MANIFEST_e0897fd3.json` con el estado `PUBLISHED_SUCCESSFULLY`.

## 5. Safety Verification
- **test_touched**: NO (Confirmado 0% leakage).
- **raw_data_mutated**: NO (Parquets protegidos).
- **sweep_run**: NO (Solo preflight de 1 configuración).
- **full_backtest_run**: NO.
- **optimization_run**: NO.

## 6. Warnings
- NINGUNO. El ZIP oficial fue localizado en la ruta correcta desde el inicio de la sesión.

## 7. Decision
**READY_FOR_CLEAN_MICRO_SMOKE_RERUN**

## 8. Next Step
Se autoriza la ejecución de un **Micro Smoke Rerun** de las familias F06, F08 y F12 para un único mes (e.g., 2022-05) para confirmar la estabilidad estadística con el nuevo motor antes de la corrida completa.

---
**Auditado por**: Antigravity Senior Technical Auditor
