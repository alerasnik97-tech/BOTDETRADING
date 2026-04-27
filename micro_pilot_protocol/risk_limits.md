# Límites de Riesgo - Micro Piloto

> [!CAUTION]
> **ESTADO: NOT_ACTIVE_UNTIL_MICRO_PILOT_ALLOWED**

Este documento define la política de gestión de capital para la fase de micro piloto real. Estas reglas son **DUROS** y no admiten excepciones.

## ESTADO: PENDING_GATE_APPROVAL
Los siguientes parámetros son los establecidos como **PILOT_DEFAULTS_CONSERVATIVE** y solo serán operativos tras el cambio de estado del gate.

## Parámetros de Exposición
| Parámetro | Valor | Código |
|-----------|-------|--------|
| Riesgo por Trade | 0.10% - 0.25% | `MAX_RISK_PER_TRADE` |
| Máximo Trades por Día | 1 | `MAX_DAILY_TRADES` |
| Máximo Exposición Simultánea | 1 Trade | `MAX_OPEN_POSITIONS` |
| Tamaño de Lote | Fijo (Mínimo posible) | `FIX_LOT_SIZE` |

## Parámetros de Protección (Drawdown)
| Parámetro | Valor | Acción |
|-----------|-------|--------|
| Stop Diario | 1.0% | Pausa hasta mañana |
| Stop Semanal | 2.5% | Pausa hasta próxima semana + Review |
| Kill Switch de Piloto | 5.0% DD | Retorno obligatorio a **SHADOW_ONLY** |

## Reglas de Comportamiento No Negociables
1. **Sin Scaling:** No se aumenta el tamaño de la posición bajo ninguna circunstancia.
2. **Sin Parámetros Dinámicos:** No se cambian SL/TP/Filtros durante la ejecución del piloto.
3. **Línea Única:** Solo se opera la línea autorizada por el tribunal (EURUSD CORE).
4. **No Mezclar Señales:** Prohibido tomar señales de otras estrategias o discrecionales.
5. **Prohibición de "Venganza":** Prohibido aumentar riesgo para recuperar pérdidas.
6. **Prohibición de Exceso de Confianza:** Una racha ganadora NO autoriza a subir el riesgo.

## Validación de Límites
Antes de abrir cualquier operación, el operador DEBE validar que el riesgo resultante no excede el 0.25% del balance actual de la cuenta de piloto.
