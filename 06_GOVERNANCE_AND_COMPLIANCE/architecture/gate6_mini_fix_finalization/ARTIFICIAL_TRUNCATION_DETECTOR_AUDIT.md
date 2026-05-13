# AUDITORÍA DEL DETECTOR DE TRUNCAMIENTO ARTIFICIAL
**Módulo Auditado:** `src/v7_engine/engine.py` (`close_position_with_costs`)  
**Fase:** Gate 6 Mini Fix Finalization  
**Fecha:** 2026-05-13  
**Estado:** `TRUNCATION_DETECTOR_WINDOW_COMPLETENESS_ACTIVE`

---

## 1. Cómo se detectaba antes
El motor unificado clasificaba una salida intradiaria bajo la tipología espuria `ARTIFICIAL_TRUNCATION` únicamente si se daban de forma concurrente dos condiciones de código imperativo:
```python
if exit_result.reason == "EOM":
    if scanned_cnt == 3000:
        eom_type = "ARTIFICIAL_TRUNCATION"
```

## 2. Por qué `scanned_cnt == 3000` quedó obsoleto
En la fase previa de remediación forense del runner (Gate 6 Mini Fix), se eliminó la extracción forzada `.head(3000)` sobre el streaming intradiario de la posición (`ticks_during`), permitiendo iterar incondicionalmente todos los ticks hasta la caducidad teórica de la posición. En consecuencia, un recuento estático exacto de 3000 ticks dejó de ser representativo de un corte de software forzado, pudiendo ocurrir naturalmente por regímenes de baja frecuencia en horarios no líquidos.

## 3. Cómo se detecta ahora
La nueva implementación supedita el estatus de truncamiento artificial a la **incompletitud temporal física** de la rebanada subyacente observada en relación a la frontera de finalización de la orden:
```python
tick_window_complete = (w_end >= intended_end or exit_result.reason in ["TP", "SL", "BE", "TIME"])
if exit_result.reason == "EOM":
    if not tick_window_complete:
        eom_type = "ARTIFICIAL_TRUNCATION"
```

## 4. Campos Físicos de Soporte
La trazabilidad criptográfica en la estructura `TradeRecord` queda respaldada por las siguientes variables nativas capturadas in situ:
*   `intended_position_end`: Horizonte máximo en que el trade debía continuar abierto en ausencia de cruces de precio (T+8 horas).
*   `actual_tick_window_end`: Marca temporal real del último tick consumido por el simulador de salidas.
*   `tick_window_complete`: Bandera booleana evaluada dinámicamente.

## 5. Escenario ante Ventana Incompleta
Si un backtest o un mes discreto carece de datos suficientes y el streaming de cotizaciones cesa de forma imprevista antes de alcanzar `intended_position_end` sin que el precio haya tocado un nivel de salida, el atributo `tick_window_complete` evalúa como `False`. El sistema asigna automáticamente la tipología `ARTIFICIAL_TRUNCATION` para evitar la falsificación de la esperanza matemática.

## 6. Inclusión en Métricas de Rentabilidad
**NO, queda tajante y programáticamente excluido de las métricas**. Cualquier registro etiquetado con truncamiento artificial fuerza la asignación booleana `valid_closed_trade = False`. El agregador global omite incondicionalmente estos vectores del cómputo de rentabilidad neta, Profit Factor y R acumulada.
