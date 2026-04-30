# PHASE42 PREFLIGHT

## Validaciones
- [x] CWD correcto: `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo`
- [x] Carpeta MANIPULANTE existe.
- [x] Carpeta BOT_V2_DAYTIME_LAB existe.
- [x] START/STATUS/STOP existen en raiz de MANIPULANTE.
- [x] Reporte Phase38/38B existe.
- [x] Reporte Phase41 existe.
- [x] Regla: No tocar MT5 real.
- [x] Regla: No tocar ejecucion live.
- [x] Regla: No modificar parametros de MANIPULANTE.

## Inventario de Archivos Criticos
- `MANIPULANTE/10_LOGS_PAPER/ftmo_trial_bot/decisions.csv`
- `MANIPULANTE/10_LOGS_PAPER/ftmo_trial_bot/quick_status.txt`
- `BOT_V2_DAYTIME_LAB/src/phase37_ftmo_trial_bot_runner.py`

## Objetivos
Crear un entorno de control forward que permita medir objetivamente la readiness para cuenta paga.
Validar protecciones mediante stress tests simulados.
