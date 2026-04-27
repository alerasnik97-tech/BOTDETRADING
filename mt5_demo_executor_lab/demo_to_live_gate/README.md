# Demo to Live Gate: Puente Institucional

Este directorio gestiona la transición del entorno demo al entorno **Live Sandbox (100 USD)**. 

## Flujo de Trabajo
1. **Ejecución Demo:** El bot corre en `mt5_demo_executor_lab`.
2. **Trade TP:** Se espera a que ocurra un trade automático cerrado en ganancia (TP).
3. **Auditoría:** Se ejecuta `demo_tp_perfect_trade_gate.py` para auditar la perfección técnica de ese trade.
4. **Habilitación:** Si el veredicto es `DEMO_TP_GATE_PASS`, se habilita la fase de preparación de real.

## Archivos Clave
- `gate_policy.md`: Las reglas de la transición.
- `demo_tp_perfect_trade_gate.py`: El script que emite el veredicto.
- `live_sandbox_100usd_policy.md`: Las reglas para el dinero real.
- `live_sandbox_activation_checklist.md`: Lo que debe estar listo antes de tocar el botón de real.

---
**No hay activación automática de real. El gate es una habilitación técnica, no una promoción operativa.**
