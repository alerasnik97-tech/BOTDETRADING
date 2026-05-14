# AUDITORÍA DE TRAZA: DIMENSIÓN `entry_type`
**Estrategia:** R1 (Mean Reversion / NY Open Absorption)  
**Estatus de Traza:** BUG_CONFIRMED / IGNORED  

---

## 1. Definición en el Search Space
- **Ubicación:** `v49_expansion_run.py` dentro de la función `generate_batch3_configs()`.
- **Valores Permutados:** `["NEXT_OPEN", "MIDPOINT_STOP"]` (y teóricamente extensivo a `LIMIT_50_REJECTION`).

## 2. Carga y Validación en Memoria
- **Carga:** Mapeado como un string literal en la instanciación de la clase de datos `R1Config`.
- **Validación:** Incorporado en la cadena de hashing criptográfico `get_config_hash()` para la presunta deduplicación del grid.

## 3. Interacción con Capas de Lógica
- **¿Pasado al Detector (`R1AbsorptionDetector`)?** **NO.** El extractor de señales desconoce la modalidad de entrada, emitiendo candidatos abstractos basados en la absorción del nivel.
- **¿Pasado al Motor (`UnifiedV7Engine`)?** **NO.** El bucle de ejecución transaccional en la línea 143 invoca unívocamente:
  ```python
  fill, reason = engine.execute_signal(sig.direction.lower(), ts, t_window)
  ```
  Al no pasarse los parámetros opcionales `entry_mode` ni `stop_price`, la capa de ejecución recurre implícitamente al modo a mercado por defecto.

## 4. Impacto Físico Observado
- **¿Debería impactar el trade?** **SÍ.** Debería condicionar el llenado a la superación del punto medio de la barra de señal (`MIDPOINT_STOP`) o re-evaluar la cotización límite (`LIMIT_50_REJECTION`).
- **¿Aparece en outputs o altera resultados?** **NO.** Todas las operaciones de configuraciones hermanas registran tiempos exactos de fill y cotizaciones idénticas de entrada, probando la completa irrelevancia del parámetro.

$$\text{VEREDICTO} = \mathbf{ENTRY\_TYPE\_NOT\_HONORED}$$
