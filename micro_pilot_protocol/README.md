# Protocolo de Micro Piloto Real Controlado

Este directorio contiene la documentación operativa y técnica necesaria para la ejecución de la fase de **Micro Piloto Real**.

## ESTADO ACTUAL DEL PROTOCOLO
> [!IMPORTANT]
> **ESTADO: NOT_ACTIVE_UNTIL_MICRO_PILOT_ALLOWED**
> 
> Este protocolo está en estado de **bloqueo preventivo**. No se autoriza ninguna operación real hasta que el `micro_pilot_gate` emita un veredicto de `MICRO_PILOT_ALLOWED`.

## Contenido del Directorio

- [MICRO_PILOT_PROTOCOL.md](MICRO_PILOT_PROTOCOL.md): Documento maestro del protocolo.
- [activation_checklist.md](activation_checklist.md): Requisitos previos para la activación.
- [daily_operator_checklist.md](daily_operator_checklist.md): Rutina diaria del operador.
- [kill_switch_rules.md](kill_switch_rules.md): Reglas de parada inmediata y retorno a Shadow.
- [risk_limits.md](risk_limits.md): Límites de riesgo ultra-conservadores.
- [execution_rules.md](execution_rules.md): Reglas de ejecución no negociables.
- [escalation_rules.md](escalation_rules.md): Reglas contra el escalado impulsivo.
- [status_template.json](status_template.json): Plantilla de estado operativo.
- [protocol_summary.md](protocol_summary.md): Resumen ejecutivo del protocolo.

## Regla de Oro
Cualquier desviación del protocolo o cambio en el gate institucional obliga al retorno inmediato a **SHADOW_ONLY**.
