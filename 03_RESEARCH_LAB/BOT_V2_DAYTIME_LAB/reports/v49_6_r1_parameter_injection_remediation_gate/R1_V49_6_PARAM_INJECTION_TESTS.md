# ESPECIFICACIÓN DE PRUEBAS DE INYECCIÓN PARAMÉTRICA
**Arnés de Aislamiento:** `run_r1_v49_6_param_injection_microtest.py`  
**Metodología:** Evaluación Ceteris Paribus Unidimensional  
**Muestra Restringida:** Exclusivamente TRAIN (`2023-01`) y VAL (`2024-06`)  

---

## 1. Arquitectura del Arnés de Aislamiento
El script de micro-pruebas se desacopla por diseño de la generación aleatoria de grillas gigantescas. Su función es instanciar una **Configuración Base de Control** y someterla a variaciones atómicas de un solo parámetro a la vez sobre el flujo físico de datos en formato Parquet.

## 2. Permutaciones Contractuales Obligatorias

### A. Dimensión Entrada (`entry_type`)
- **Base:** `NEXT_OPEN` (Cierre de barra de señal lanza fill inmediato a mercado).
- **Variante 1:** `LIMIT_50_REJECTION` (Condicionado al retroceso del precio).
- **Variante 2:** `MIDPOINT_STOP` (Orden condicional exigiendo cruce de cotización ask/bid superior/inferior).

### B. Dimensión Fricción y Salida (`sl_model`)
- **Base:** `WICK_PLUS_1_0` (Multiplicador base o ancla ajustada).
- **Variantes:** `WICK_PLUS_1_5`, `WICK_PLUS_2_0`, `MICROSTRUCTURE_PLUS_1_5`.

### C. Dimensión Objetivo (`target`)
- **Base:** `1.25R`
- **Variantes:** `1.5R`, `2.0R`, `2.5R`

### D. Dimensión Protección (`BE`)
- **Base:** `none` (Sin Break-Even activado).
- **Variantes:** `1.0R`, `1.25R`, `1.5R`

## 3. Salidas de Evidencia Física
El arnés automatizará el registro inmutable de:
1. `R1_V49_6_PARAM_TEST_CONFIGS.csv` (Compendio de las configuraciones atómicas ensayadas).
2. `R1_V49_6_PARAM_TEST_TRADES.csv` (Bitácora transaccional física a nivel de tick/fill).
3. `R1_V49_6_PARAM_TEST_RESULTS.csv` (Agregación escalar de $PnL$ y Profit Factor).
4. `R1_V49_6_PARAM_TEST_DIFFS.csv` (Matriz de aserción comprobando matemáticamente que variaciones en la dimensión de entrada/salida inducen cambios de valor en los campos `entry_price`, `sl_price`, `tp_price` y en el identificador criptográfico `trade_set_hash`).
