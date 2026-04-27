# Kill Switch Manual: Reglas de Detención Inmediata

Cualquier trigger listado aquí obliga a la **detención inmediata del micro-piloto manual**.

| Código | Trigger (Disparador) | Severidad | Acción Inmediata | Re-evaluación |
| :--- | :--- | :---: | :--- | :--- |
| KS-M-01 | 3 pérdidas consecutivas (3 SL) | Alta | Pausar piloto 48h | Revisar fidelidad de setups |
| KS-M-02 | Drawdown del Piloto > 5% | Crítica | Cierre definitivo del piloto | Volver a SHADOW_ONLY indefinido |
| KS-M-03 | Breach del Stop Diario (1%) | Media | Detener por el día | Revisar slippage o error de tamaño |
| KS-M-04 | Breach del Stop Semanal (2.5%) | Alta | Detener por la semana | Auditoría de riesgos |
| KS-M-05 | Cambio discrecional en estrategia | Alta | Pausar piloto 7 días | Re-entrenamiento en reglas |
| KS-M-06 | Operar > 1 trade por día | Media | Detener 48h | Revisar disciplina |
| KS-M-07 | Cambio de tamaño sin autorización | Alta | Detener 72h | Auditoría de capital |
| KS-M-08 | Inconsistencia grave con Shadow | Media | Pausar piloto | Sincronizar lectura de niveles |
| KS-M-09 | Intento de recuperar pérdida (Revenge) | Crítica | Detener 7 días | Revisión psicológica |

## Procedimiento de Activación
1. Cerrar cualquier posición abierta a mercado.
2. Anotar el código del trigger en el diario de trading.
3. Notificar el estado de "PAUSADO" o "TERMINADO" en el daily review.
4. No abrir órdenes reales hasta cumplir el tiempo de re-evaluación.

---
**EL KILL SWITCH PROTEGE EL CAPITAL CUANDO LA DISCIPLINA FALLA.**
