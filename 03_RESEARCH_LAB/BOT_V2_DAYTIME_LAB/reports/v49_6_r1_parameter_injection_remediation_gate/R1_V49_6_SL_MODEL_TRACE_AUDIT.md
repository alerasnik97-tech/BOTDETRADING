# AUDITORÍA DE TRAZA: DIMENSIÓN `sl_model`
**Estrategia:** R1 (Mean Reversion / NY Open Absorption)  
**Estatus de Traza:** BUG_CONFIRMED / IGNORED  

---

## 1. Definición en el Search Space
- **Ubicación:** Configuración de dimensiones permutadas en `v49_expansion_run.py`.
- **Valores Permutados:** `["WICK_EXTREME_1.5P", "WICK_EXTREME_2.0P"]` (y derivados conceptuales).

## 2. Carga y Validación en Memoria
- **Carga:** Almacenado como cadena literal en el descriptor `R1Config.sl_model`.
- **Validación:** Empleado de forma puramente representativa en el cálculo del hash identificador del grid.

## 3. Interacción con Capas de Lógica
- **¿Pasado al Detector?** **NO.** El detector extrae mechas y umbrales base de absorción sin computar distancias monetarias de riesgo.
- **¿Pasado al Motor?** **NO.** El orquestador intercepta el llenado exitoso y en las líneas 146-147 impone incondicionalmente:
  ```python
  sl_mult = 1.5
  sl_dist = abs(fill.fill_price - (sig.low if sig.direction == 'LONG' else sig.high)) * sl_mult
  ```
  La variable `cfg.sl_model` jamás es leída, evaluada o desglosada para inyectar su multiplicador o ancla específica.

## 4. Impacto Físico Observado
- **¿Altera cotizaciones de SL o resultados?** **NO.** Toda configuración genera idéntica cotización de salida por Stop Loss y Take Profit al derivar el ratio sobre la misma constante estática de `sl_mult = 1.5`.

$$\text{VEREDICTO} = \mathbf{SL\_MODEL\_NOT\_HONORED}$$
