## Estado
BENCHMARK_READY

## Resumen
Se ha completado de manera aséptica y forense la reconstrucción documental del **Benchmark Histórico de Manipulante Original**, estableciendo el mapa de fuentes canónicas y el marco comparativo de aserción para futuras implementaciones. Toda la suite de gobierno reside intacta y sellada en la ruta:  
`C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\06_GOVERNANCE_AND_COMPLIANCE\benchmarks\manipulante_original\`

## Benchmark Manipulante Original
- **PF:** Programado Full (2015-2026) = **2.79**; Programado Autoridad (2020-2026) = **2.94**; Manual Baseline = **1.88** bruto / **1.53** normalizado en $R$.
- **WR:** Programado Full = **32.5%**; Programado Autoridad = **38.5%**; Manual = **35.0%**.
- **DD:** Programado Full = **-5.58 R**; Programado Autoridad = **-5.0 R**; Manual = **UNKNOWN** (Pendiente de auditoría fina).
- **total R:** Programado Full = **+737.47 R** acumulados brutos.
- **expectancy:** Programado Full = **+0.281 R**; Programado Autoridad = **+0.309 R**; Manual = **+0.36 R** bruto / **+0.25 R** neto normalizado.
- **trades:** Programado Full = **2,625**; Programado Autoridad = **1,602**; Manual = **841** operaciones vivas.
- **período:** Programado Full = **Enero 2015 a Abril 2026** (11 años contiguos); Manual = **2020 a 2026**.
- **horario:** Programado = **07:00 a 20:30 NY**; Manual = **08:00 a 11:00 NY** (Concentración nativa en Killzone).
- **gross/net:** Métricas iniciales reportadas en bruto (**GROSS**), sin deducción asimétrica nativa de slippage en la curva base.
- **confianza:** **HIGH** para métricas algorítmicas de reconstrucción; **MEDIUM** para registros de bitácora manual.

## Gaps importantes
- **Drawdown Manual Faltante (`GAP_001`):** Carencia de registros auditados sobre la máxima excursión adversa de capital del usuario, impidiendo el cálculo de ratios de eficiencia ajustados por riesgo.
- **Deducción Nativa de Costos (`GAP_002`):** Las series base originales no sustraen de forma continua la comisión FTMO ni el estrés de slippage en sus sumarios primarios, exigiendo un ajuste asimétrico en comparaciones futuras.
- **Resolución Intratrade (`GAP_003`):** Ausencia de vectores de excursión adversa continua tick a tick (MAE) dentro de las barras lentas en los resúmenes serializados heredados.

## Target futuro
- **EURUSD:** Exclusivo y obligatorio.
- **horario:** Búsqueda restringida de **07:00 a 17:00 NY time** como límite máximo, recomendando encarecidamente acotar al **NY Open Killzone (08:00 a 11:00 NY)**.
- **max trades/day:** Límite superior incondicional de **3 operaciones diarias** ($\le 3$).
- **PF mínimo:** $\text{PF}_{\text{val\_net}} \ge 1.15$ como piso de supervivencia; objetivo supremo de diseño $\text{PF}_{\text{net}} \ge 2.94$ (igualar a la autoridad tras deducir costos).
- **DD máximo:** Retroceso de capital contenido con una cota estricta $\text{Max DD}_{\text{net}} \le -5.0\text{ R}$.
- **costos:** Sustracción nativa obligatoria de **$5.00 por lote** modelados en unidades internas de riesgo.
- **slippage:** Supervivencia incondicional demostrada bajo un estrés asimétrico continuo de **0.2 pips**.
- **FTMO:** Inmunidad total garantizada sin quiebres de pérdida diaria o global en el simulador de fondeo (`ftmo_blown = False`).

## Recomendación
Al concluir las iteraciones del laboratorio activo (ej. Manipulante 4), el equipo de investigación debe someter las curvas de equidad candidatas a la matriz del **Target Comparison Framework** (`TARGET_COMPARISON_FRAMEWORK.md`). Se prohíbe autorizar el avance hacia Forward Demo si la nueva formulación no supera el pliego de condiciones de rendimiento neto o si su aparente ventaja estadística incurre en las cláusulas de exclusión (sobreoperar, operar de noche, ametrallar el libro o seleccionar hiperparámetros espiando la partición testigo TEST).

## Prohibiciones respetadas
Se certifica de forma absoluta y categórica el cumplimiento de las fronteras de confinamiento del Agente Paralelo:
- **no código tocado:** Confirmado (Cero alteraciones en motores, lógicas o scripts `.py`).
- **no runner tocado:** Confirmado (Orquestadores de simulación preservados intactos).
- **no tests tocados:** Confirmado (Árbol de pruebas de aserción inmutable).
- **no datos tocados:** Confirmado (Bóvedas de mercado y series de noticias leídas con candado estricto de solo lectura).
- **no ZIP tocado:** Confirmado (Archivo canónico de entrega `000_PARA_CHATGPT.zip` inalterado).
- **no backtest:** Confirmado (Cero invocaciones a simulaciones de I/O sobre Parquet).
- **no sweep:** Confirmado (Cero iteraciones de minería o barrido de parámetros).
- **no commit:** Confirmado (Cero confirmaciones de cambios en el repositorio local).
- **no push:** Confirmado (Cero sincronizaciones con repositorios remotos).
- **no Explorer:** Confirmado (Cero aperturas de interfaces gráficas del sistema operativo).
