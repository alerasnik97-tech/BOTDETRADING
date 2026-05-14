# AUDITORÍA DE TRAZA: DIMENSIÓN `target`
**Estrategia:** R1 (Mean Reversion / NY Open Absorption)  
**Estatus de Traza:** PARTIALLY_HONORED / BE_IGNORED  

---

## 1. Definición en el Search Space
- **Ubicación:** Matriz de combinaciones en `generate_batch3_configs()`.
- **Valores Permutados:** `["FIXED_1.5R", "FIXED_2.0R", "FIXED_2.5R"]`.

## 2. Carga y Validación en Memoria
- **Carga:** Registrado como string literal en el campo `cfg.target_model`.
- **Validación:** Empleado en la huella hash del identificador determinista.

## 3. Interacción con Capas de Lógica
- **¿Pasado al Motor?** **PARCIALMENTE.** En la línea 150 de `v49_expansion_run.py`, el orquestador extrae el escalar numérico de la cadena:
  ```python
  tp_r = float(cfg.target_model.split("_")[1].replace("R", ""))
  tp_price = fill.fill_price + (sl_dist * tp_r) if sig.direction == 'LONG' else fill.fill_price - (sl_dist * tp_r)
  ```
  La cotización objetivo es inyectada correctamente en `engine.close_position_with_costs()`.
- **Omisión Forense del Break-Even (`BE`):** Aunque el Take Profit es honrado a nivel de precio de orden de salida, el pasaje omite capturar o suministrar el argumento opcional `be_trigger_r` admitido por la capa de cierre en el motor central, provocando que la dimensión de Break-Even quede absolutamente suprimida del backtest.

$$\text{VEREDICTO} = \mathbf{TARGET\_HONORED\ /\ BE\_NOT\_HONORED}$$
