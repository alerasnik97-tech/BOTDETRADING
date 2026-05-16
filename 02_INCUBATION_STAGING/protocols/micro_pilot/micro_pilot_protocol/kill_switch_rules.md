# Reglas del Kill Switch Duro - Micro Piloto

> [!CAUTION]
> **ESTADO: NOT_ACTIVE_UNTIL_MICRO_PILOT_ALLOWED**

El Kill Switch es el mecanismo de defensa final. Su activación es mandatoria y no discutible ante los siguientes triggers.

## Triggers de Activación

| Código | Trigger | Severidad | Acción Inmediata | Consecuencia |
|--------|---------|-----------|------------------|--------------|
| `KS_DD_5` | Drawdown acumulado del piloto > 5.0% | **CRÍTICA** | Cerrar todo | Retorno a **SHADOW_ONLY** (Bloqueo 30 días) |
| `KS_STOP_W` | Breach de Stop Semanal (2.5%) | **ALTA** | Cerrar todo | Pausa hasta lunes + Review obligatorio |
| `KS_STOP_D` | Breach de Stop Diario (1.0%) | **MEDIA** | Cerrar todo | Pausa hasta mañana |
| `KS_LOSS_3` | 3 Pérdidas consecutivas en Real | **ALTA** | Pausa operativa | Review de alineación con Shadow |
| `KS_SHADOW_MISMATCH` | Inconsistencia material Real vs Shadow | **ALTA** | Pausa inmediata | Auditoría de conectividad/lógica |
| `KS_GATE_LOCK` | Cambio de Gate a `NOT_READY` | **BLOQUEANTE** | Cerrar todo | Retorno inmediato a **SHADOW_ONLY** |
| `KS_TECH_FAIL` | Falla de trazabilidad / Ausencia de logs | **MEDIA** | No operar | Reparación de infraestructura |
| `KS_DISC_OVERRIDE` | Cambio manual no autorizado en parámetros | **CRÍTICA** | Bloqueo manual | Tribunal Institucional |

## Protocolo de Ejecución
1. **Detectar:** El operador o el sistema detecta el trigger.
2. **Cerrar:** Se cierran todas las posiciones abiertas a mercado (Panic Button).
3. **Bloquear:** Se cambia el estado en `status_template.json` a `allowed_to_trade: false`.
4. **Documentar:** Se registra la causa en `micro_pilot_protocol/outputs/incident_log.md`.
5. **Revisar:** No se puede reactivar sin una revisión manual y firma de veredicto.

## Reactivación
La reactivación tras un `KS_DD_5` requiere reiniciar el ciclo de incubación Shadow (N=0) y obtener un nuevo `MICRO_PILOT_ALLOWED`.
