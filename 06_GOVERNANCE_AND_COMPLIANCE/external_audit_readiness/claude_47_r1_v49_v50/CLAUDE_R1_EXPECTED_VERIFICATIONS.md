# PROTOCOLO DE VERIFICACIONES ESPERADAS DEL AUDITOR EXTERNO
**Estrategia:** R1 (Mean Reversion / NY Open Absorption)  
**Nivel de Exigencia:** Máximo Escrutinio Institucional (Pureza Física)  

---

Se instruye al agente auditor Claude 4.7 High a ejecutar incondicionalmente las siguientes 6 pruebas de verificación física sobre los entregables de la estrategia R1 antes de emitir su dictamen de paso de compuerta (Acceptance Gate):

## 1. Verificación de Integridad Transaccional (Pureza Granular)
- **Acción Esperada:** Extraer de forma directa el archivo de transacciones de la iteración en revisión (ej. `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/v40_r1_absorption_mean_reversion/R1_MICRO_PROBE_TRADES.csv`).
- **Aserción:** Contar físicamente el número de registros (filas excluyendo cabecera) y verificar que este recuento exacto sea la única base de cálculo utilizada para reportar el tamaño muestral $N$. Prohibido aceptar resúmenes escalares que difieran del conteo en bitácora.

## 2. Verificación de Inmutabilidad de la Bóveda de Datos
- **Acción Esperada:** Auditar los metadatos y sumas de verificación de los archivos consumidos en `05_MARKET_DATA_VAULT`.
- **Aserción:** Certificar que ninguna ejecución de lotes o agregaciones en R1 ha alterado o reescrito un solo byte de los históricos de mercado o calendarios de noticias premium, respetando la estricta naturaleza de solo lectura (READ-ONLY) de la bóveda.

## 3. Verificación de Ausencia de Sesgo de Futuro (Pureza Causal)
- **Acción Esperada:** Revisar la implementación de la lógica de construcción de barras y gatillos de absorción en `r1_detector.py` y `r1_levels.py`.
- **Aserción:** Comprobar que el cálculo de umbrales intradiarios y la decisión de entrada operan exclusivamente con información disponible en o antes de la barra causal actual ($t$), garantizando cero filtración de precios o volatilidad futura.

## 4. Verificación de Modelado de Costos Realista (Fricción FTMO)
- **Acción Esperada:** Auditar las fórmulas de deducción aplicadas a la columna de ganancias y pérdidas brutas.
- **Aserción:** Certificar que toda métrica neta oficial incorpora de forma inquebrantable una comisión base redonda de **USD 5.00 por lote negociado** y un estrés de slippage asimétrico de **0.2 a 0.5 pips**, penalizando de forma realista las ejecuciones tempranas en la apertura de Nueva York.

## 5. Verificación de Aislamiento de Muestra TEST (Pureza OOS)
- **Acción Esperada:** Rastrear las fechas de corte en las configuraciones de los lotes de barrido (Batches 1 a 3).
- **Aserción:** Probar documental y matemáticamente que la selección de las Top 5 configuraciones candidatas de R1 se realizó utilizando exclusivamente los períodos In-Sample y de Validación, preservando la pureza absoluta y virginidad estadística del período TEST.

## 6. Verificación de Recálculo Determinista
- **Acción Esperada:** Escribir y ejecutar un script de lectura pura sobre el CSV transaccional físico para recomputar desde cero las métricas clave.
- **Aserción:** Confirmar que las sumas acumuladas de la columna neta reproducen con precisión de coma flotante el Profit Factor neto, el Win Rate y la curva de Drawdown institucionalmente proclamada. Si existe una discrepancia $> 0.0001$, la verificación se considerará **FALLIDA**.
