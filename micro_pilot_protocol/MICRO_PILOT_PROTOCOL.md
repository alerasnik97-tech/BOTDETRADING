# MICRO PILOT PROTOCOL (MPP) - ESTRATEGIA EURUSD

## 1. Propósito del Micro Piloto
El propósito de esta fase es validar la ejecución de la estrategia en un entorno real bajo condiciones de riesgo mínimo (nanolotes/microlotes) para verificar:
- Conectividad y ejecución técnica.
- Desviación (slippage/spread) real vs teórica.
- Disciplina operativa y cumplimiento de protocolos.
- Consistencia con la evidencia Shadow acumulada.

## 2. Qué es el Micro Piloto
Es un experimento controlado de baja exposición diseñado para "comprar evidencia" estadística en real.

## 3. Qué NO es el Micro Piloto
- NO es un despliegue completo (Full Deployment).
- NO es una fase de generación de beneficios materiales.
- NO es una fase de optimización de parámetros.
- NO es escalable sin revisión institucional previa.

## 4. Estado Actual
> [!CAUTION]
> **NOT_ACTIVE_UNTIL_MICRO_PILOT_ALLOWED**
> 
> Actualmente el sistema se encuentra en **SHADOW_ONLY**. Este documento es operativo pero está **BLOQUEADO**.

## 5. Prerrequisitos Obligatorios de Activación
- `micro_pilot_gate` status = `MICRO_PILOT_ALLOWED`.
- Evidencia Shadow (N) >= 10 trades con métricas positivas.
- `activation_checklist.md` completada al 100% en verde.
- Kill Switch operativo y probado.

## 6. Límites de Riesgo (PILOT_DEFAULTS_CONSERVATIVE)
> [!CAUTION]
> **ESTADO: NOT_ACTIVE_UNTIL_MICRO_PILOT_ALLOWED**

- **Riesgo por Trade:** 0.10% a 0.25% del capital asignado.
- **Máximo Trades/Día:** 1.
- **Stop Diario:** 1.0%.
- **Stop Semanal:** 2.5%.
- **Kill Switch de Piloto:** Drawdown acumulado > 5.0%.

## 7. Rutina Operativa Diaria
> [!CAUTION]
> **ESTADO: NOT_ACTIVE_UNTIL_MICRO_PILOT_ALLOWED**

Referencia: `daily_operator_checklist.md`.
Consiste en Pre-Sesión (Revisión de Gate/Shadow), Sesión (Ejecución estricta) y Post-Sesión (Audit/Registro).

## 8. Kill Switch
Referencia: `kill_switch_rules.md`.
Cualquier violación de límites o inconsistencia técnica dispara la pausa inmediata.

## 9. Condiciones de Pausa
- Fallas de conectividad.
- Noticias de alto impacto no previstas.
- Desviación material entre Shadow y Real.

## 10. Condiciones de Retorno a SHADOW_ONLY
- Breach de riesgo (Stop semanal o Kill Switch de Piloto).
- Cambio de veredicto en el Gate Institucional a `NOT_READY`.
- Error en la lógica del core productivo detectado.

## 11. Regla de No-Escalado
> [!CAUTION]
> **ESTADO: NOT_ACTIVE_UNTIL_MICRO_PILOT_ALLOWED**

El tamaño de la posición es **FIJO** durante toda la fase de Micro Piloto. No se autoriza compounding ni aumento de riesgo por confianza.

## 12. Rol del Operador
El operador es un ejecutor pasivo del protocolo. Su única libertad es NO operar si detecta riesgos externos, pero nunca operar fuera de las reglas.

## 13. Archivos que Mandan
1. `micro_pilot_gate/outputs/micro_pilot_scorecard.json`
2. `micro_pilot_protocol/risk_limits.md`
3. `micro_pilot_protocol/kill_switch_rules.md`

## 14. Siguiente Paso cuando el Gate Cambie
Una vez el gate cambie a `MICRO_PILOT_ALLOWED`, el operador debe ejecutar el `activation_checklist.md` y documentar la primera sesión en el log del piloto.
