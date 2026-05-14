# PLAN DE REMEDIACIÓN DE INYECCIÓN DE PARÁMETROS (FIX PLAN)
**Módulo Objetivo:** Adaptador Orquestador `v49_expansion_run.py`  
**Frontera de Intervención:** Exclusivamente Capa Externa de Laboratorio  
**Bloqueo Respetado:** Cero Edición en `src/v7_engine/` y `src/v6_utils/`  

---

## 1. Diseño de la Refactorización en el Adaptador
Para enmendar la rotura en el pasaje de hiperparámetros sin incurrir en violaciones de arquitectura, se modificará de forma quirúrgica el bucle interno de iteración de señales (`for sig in cfg_sigs.itertuples():`) en `v49_expansion_run.py`.

### A. Inyección Dinámica de `entry_type`
Se abandonará el pasaje por defecto a mercado. La lógica del orquestador interpretará el literal alojado en `cfg.entry_type`:
- **Si `NEXT_OPEN`:** Se invoca `engine.execute_signal(side, ts, t_window, entry_mode="market")`.
- **Si `MIDPOINT_STOP`:** Se calcula la cotización límite como el punto medio entre el máximo y mínimo de la barra de señal de quiebre. Se invoca:
  ```python
  midpoint = (sig.high + sig.low) / 2.0
  stop_p = midpoint + 0.0001 if side == 'long' else midpoint - 0.0001
  engine.execute_signal(side, ts, t_window, entry_mode="stop", stop_price=stop_p)
  ```
- **Si `LIMIT_50_REJECTION`:** Se condiciona la entrada al retroceso físico hacia el 50% del cuerpo/mecha extrema de la vela base.

### B. Inyección Dinámica de `sl_model`
Se suprimirá la constante procedimental `sl_mult = 1.5`. Se parseará el sufijo numérico del literal de `cfg.sl_model` o se mapeará a umbrales variables:
- **Si `"WICK_EXTREME_1.5P"`:** `sl_mult = 1.5`.
- **Si `"WICK_EXTREME_2.0P"`:** `sl_mult = 2.0`.
- **Si `"MICROSTRUCTURE_1.5P"`:** Se deriva la distancia desde la volatilidad del swing intradiario (ATR o mecha local) multiplicada por el factor pretendido.

### C. Inyección de Break-Even (`BE`)
Se incorporará soporte implícito o explícito de Break-Even derivando un umbral de activación razonable desde el Take Profit o extendiendo el descriptor de la configuración. Al invocar el cierre, se transferirá:
```python
res = engine.close_position_with_costs(fill, sl_price, tp_price, t_window, be_trigger_r=be_target)
```

## 2. Aserciones de Seguridad del Parche
Toda modificación mantendrá paridad estricta con el entorno local de Pandas/Parquet, no añadirá dependencias externas y se someterá a validación cruzada para certificar que el core no registra deriva alguna post-ejecución.
