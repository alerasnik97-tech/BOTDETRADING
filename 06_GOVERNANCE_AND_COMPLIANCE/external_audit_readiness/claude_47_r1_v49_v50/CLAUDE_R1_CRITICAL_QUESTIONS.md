# PREGUNTAS CRÍTICAS OBLIGATORIAS PARA LA AUDITORÍA DE CLAUDE 4.7 HIGH
**Estrategia:** R1 (Mean Reversion / NY Open Absorption)  
**Iteración Objetivo:** Evaluación Transicional V49 $\rightarrow$ V50  

---

Se instruye formalmente al agente auditor Claude 4.7 High a responder de manera explícita, incondicional y exhaustivamente fundamentada en pruebas físicas a cada una de las siguientes 10 interrogantes críticas antes de emitir su dictamen de viabilidad:

### 1. ¿Hay evidencia física suficiente?
> **Criterio de Escrutinio:** ¿Existen en el repositorio archivos de transacciones individuales granulares (`*TRADES.csv`) que respalden las métricas agregadas presentadas en las matrices de resumen, o se depende de aserciones de rendimiento huérfanas de respaldo transaccional?

### 2. ¿Los rowcounts coinciden?
> **Criterio de Escrutinio:** ¿El número exacto de filas (operaciones físicas) contabilizadas en los archivos de bitácora transaccional es estrictamente idéntico al valor escalar $N$ reportado en las cabeceras de los resúmenes estadísticos oficiales?

### 3. ¿N coincide con trades reales?
> **Criterio de Escrutinio:** ¿Cada registro de la bitácora corresponde unívocamente a una operación con precios de entrada, salida, timestamps exactos y markdowns causales aplicados, descartando la inserción de rellenos sintéticos o duplicaciones artificiales?

### 4. ¿Hay TEST leakage?
> **Criterio de Escrutinio:** ¿Se puede probar matemáticamente y mediante inspección del código fuente que los hiperparámetros óptimos extraídos durante los barridos de TRAIN/VAL en V48/V49 mantuvieron un aislamiento hermético respecto a los datos del período reservado TEST OOS?

### 5. ¿Hay config_id mismatch?
> **Criterio de Escrutinio:** ¿Existe una correspondencia biunívoca perfecta entre los identificadores de configuración (`config_id`) declarados en los archivos `*_RUN_CONFIG.json` y las firmas de metadatos estampadas en las bitácoras transaccionales resultantes?

### 6. ¿Hay duplicados?
> **Criterio de Escrutinio:** ¿Se ha verificado la unicidad absoluta de cada transacción mediante un escaneo de claves compuestas (`timestamp_entrada` + `precio_entrada` + `direccion`), asegurando la ausencia de dobles contabilizaciones que inflen artificialmente el Profit Factor o el conteo de operaciones?

### 7. ¿Las métricas recalculan desde trades?
> **Criterio de Escrutinio:** ¿Al ejecutar una suma independiente de las columnas `profit_loss_net` (o equivalentes brutas menos comisiones y slippage) directamente sobre el archivo CSV físico, se obtienen exactamente los mismos valores de ganancia neta, Profit Factor y Drawdown declarados institucionalmente?

### 8. ¿Top5 fue elegido solo con TRAIN/VAL?
> **Criterio de Escrutinio:** ¿El subconjunto de las 5 mejores configuraciones candidatas (Top 5) fue promovido aplicando estrictamente criterios de dominancia sobre las particiones In-Sample y de Validación, sin que ninguna ejecución preliminar sobre la muestra TEST haya influido en la poda del espacio dimensional?

### 9. ¿V49 merece V50 o necesita más batches?
> **Criterio de Escrutinio:** En base al comportamiento de la agregación lógica actual y la asimilación de fricción asimétrica, ¿la iteración V49 posee un edge estadístico neto suficiente ($PF_{net} \ge 1.5$) para justificar la apertura de la Acceptance Gate hacia el entorno de incubación V50, o la varianza observada exige la inyección obligatoria de lotes de muestreo adicionales (más batches)?

### 10. ¿Cuál es la única próxima acción correcta?
> **Criterio de Escrutinio:** Conforme al marco normativo institucional del proyecto y basándose de forma exclusiva en el resultado puro y auditable de las 9 preguntas anteriores, ¿cuál es el único paso secuencial, seguro y metodológicamente inquebrantable que debe ejecutar el equipo de ingeniería (ej. Promover a V50, Ejecutar Batch 4, o Emitir Dictamen RED de Muerte Rápida)?
