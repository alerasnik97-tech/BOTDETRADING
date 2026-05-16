# PROMPT: FINAL PRE-LAB GATE RETRY (POST-REMEDIACIÓN)

Actuá como Codex GPT-5.5 Max en modo Institutional Lab Readiness Officer.

OBJETIVO: Ejecutar el `FINAL_PRE_LAB_GATE` sobre la rama canónica `governance/root-strict-final-pass-20260516`.

REGLAS:
- Usar únicamente la rama designada.
- No modificar el código.
- Verificar integridad de datos.

ACCIONES:

1. CARGA DE ENTORNO:
$env:PYTHONPATH="03_RESEARCH_LAB"

2. EJECUCIÓN DE GATE:
python -m research_lab.lab_preflight --pair EURUSD --mode TRAIN-ONLY --check-foundation --check-engine --check-news

3. VERIFICACIÓN DE REPORTE:
Verificar que `06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/EURUSD_TRAIN_ONLY_AUTHORIZATION.md` se genere con estado APPROVED.

4. CERTIFICACIÓN:
Si el preflight pasa, el laboratorio queda oficialmente abierto para investigación TRAIN-ONLY.
