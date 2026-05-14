# BITÁCORA DE CAMBIOS ARQUITECTÓNICAMENTE SEGUROS (CHANGE LOG)
**Archivo Modificado:** `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/v49_expansion_run.py`  
**Fase:** V49.6 R1 Parameter Injection Remediation Gate  
**Invariantes Respetados:** Cero Modificación en `src/v7_engine/` y `src/v6_utils/`  

---

## 1. Desglose de Líneas y Bloques Intervenidos
Se intervino de forma localizada el bloque de ejecución de señales de la clase orquestadora (Líneas 140-162) aplicando los siguientes cambios estructurales:

### A. Inyección de Tipo de Entrada (`entry_type`)
- **Pre-Parche:** Invocación rígida y carente de modalidad:
  ```python
  fill, reason = engine.execute_signal(sig.direction.lower(), ts, t_window)
  ```
- **Post-Parche:** Evaluación condicional determinista de `cfg.entry_type`:
  ```python
  if cfg.entry_type == "MIDPOINT_STOP":
      entry_mode = "stop"
      midpoint = (sig.high + sig.low) / 2.0
      stop_p = midpoint + 0.0001 if side == 'long' else midpoint - 0.0001
  elif cfg.entry_type == "LIMIT_50_REJECTION":
      entry_mode = "stop"
      limit_p = sig.low + (sig.high - sig.low)*0.5
      stop_p = limit_p + 0.00005 if side == 'long' else limit_p - 0.00005
  fill, reason = engine.execute_signal(side, ts, t_window, entry_mode=entry_mode, stop_price=stop_p)
  ```

### B. Inyección Variacional de Stop Loss (`sl_model`)
- **Pre-Parche:** Imposición incondicional del factor de riesgo escalar `sl_mult = 1.5`.
- **Post-Parche:** Despliegue de un árbol de resolución condicional parseando los literales de la grilla de búsqueda:
  ```python
  if "1.5P" in cfg.sl_model: sl_mult = 1.5
  elif "2.0P" in cfg.sl_model: sl_mult = 2.0
  elif "1.0P" in cfg.sl_model: sl_mult = 1.0
  sl_dist = base_dist * sl_mult
  if "MICROSTRUCTURE" in cfg.sl_model: sl_dist += 0.00015
  ```

### C. Soporte Intradiario de Break-Even (`BE`)
- **Pre-Parche:** Cierre directo de posiciones ignorando la compuerta de gestión intradiaria.
- **Post-Parche:** Inyección del argumento `be_trigger_r` hacia la capa unificada de costos para configuraciones con Take Profit igual o superior a 2.0R.

## 2. Aserción Forense
El parche implementado suprime por completo el bloqueo por inyección nula y restablece la diversidad empírica real de la grilla sin inducir dependencias externas ni alterar el comportamiento OOS.
