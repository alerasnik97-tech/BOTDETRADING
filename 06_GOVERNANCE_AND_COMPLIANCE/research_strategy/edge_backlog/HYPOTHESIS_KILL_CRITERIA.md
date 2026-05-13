# CRITERIOS INSTITUCIONALES DE MUERTE RÁPIDA Y ESCALADO DE HIPÓTESIS
**Contexto de Aplicación:** Búsqueda Cuantitativa de Edge Intradiario en EURUSD  
**Gobernanza:** Supresión determinística de sobreoptimización y minería de datos por fuerza bruta.

---

## 1. Criterios de Muerte Rápida (Kill Criteria)
Una hipótesis de investigación queda incondicional e irrevocablemente **clausurada en estado RED** si durante las fases de pre-selección o validación cruzada incurre en cualquiera de las siguientes aserciones de fallo:

> [!CAUTION]  
> *   **Rendimiento Neto Deficiente:** El Profit Factor neto en la partición de validación cae por debajo del umbral crítico de viabilidad: $\text{PF}_{\text{val\_net}} < 1.05$ deducidos los costos de comisión y sometido a una fricción continua obligatoria de **$0.2$ pips de slippage asimétrico**.
> *   **Deterioro en Test Witness:** El Profit Factor neto en el conjunto de prueba (Test set testigo) colapsa a niveles inoperables: $\text{PF}_{\text{test\_net}} < 0.90$ bajo latencia estresada de $0.2$ pips.
> *   **Inanición Estadística de la Muestra:** La cantidad total de transacciones generadas resulta insuficiente para sustentar inferencia probabilística: $N_{\text{val}} < 40$ o $N_{\text{test}} < 40$ en sus respectivos intervalos históricos.
> *   **Riesgo de Ruina FTMO:** La curva teórica incurre en una violación de pérdida diaria máxima o pérdida global máxima de una cuenta de fondeo FTMO estándar en etapas tempranas de simulación (`ftmo_blown = True`).
> *   **Contaminación por Liquidaciones Artificiales:** Se demuestra que la rentabilidad global positiva depende de la inyección de cierres forzados de fin de mes (EOM) a precios irreales que inflan engañosamente la equidad.
> *   **Fragilidad Extrema ante Fricción:** La estrategia aparenta ser altamente rentable en un entorno ideal (`slippage = 0.0`), pero la inyección de la penalización nativa de $0.2$ pips destruye la ventaja, indicando que el sistema se basa en capturar variaciones milimétricas del spread interbancario.
> *   **Concentración de Cola Larga:** El factor de beneficio neto depende en más de un $40\%$ de los retornos generados por tan solo 1 o 2 operaciones de recorrido masivo atípico (outliers).
> *   **Sobreajuste Dimensional:** El modelo requiere la conjunción simultánea de demasiadas condiciones y sub-filtros booleanos ad-hoc para arrojar un resultado positivo.
> *   **Subjetividad Algorítmica Latente:** La traducción a código exige incorporar parámetros o umbrales visuales que no pueden formularse de manera estrictamente causal sin espiar el futuro.
> *   **Fallo de Verificación Independiente:** Las métricas arrojadas por el motor de simulación principal discrepan frente a la reconstrucción independiente en un segundo lenguaje o motor testigo (`independent_verify = Failed`).

---

## 2. Criterios para Escalar a Barridos Mayores (Escalation Criteria)
Una hipótesis piloto solo califica para ser promovida a barridos de optimización secundaria, pruebas de estrés computacional o despliegues en Forward Demo si satisface simultáneamente el siguiente pliego de condiciones de robustez:

> [!IMPORTANT]  
> *   **Excelencia en Validación:** Demostración ineludible de un Profit Factor neto sustancial: $\text{PF}_{\text{val\_net}} \ge 1.15$ sostenido bajo la inyección incondicional de **$0.2$ pips de slippage asimétrico**.
> *   **Supervivencia en Testigo OOS:** Preservación de la esperanza matemática en el conjunto de prueba testigo: $\text{PF}_{\text{test\_net}} \ge 1.0$ con deducción de fricción de $0.2$ pips.
> *   **Densidad Muestral Suficiente:** Generación consistente de señales con un tamaño de muestra robusto superando las cotas mínimas de significancia estadística a lo largo de los meses evaluados.
> *   **Estabilidad de Drawdown:** Incurre en un retroceso máximo (Max DD) contenido y coherente con las reglas de gestión de riesgo institucional, permitiendo una rápida recuperación en unidades de $R$.
> *   **Supervivencia FTMO Garantizada:** Cero violaciones de las reglas de capital de la firma de fondeo a lo largo de la totalidad de las trayectorias simuladas.
> *   **Higiene de Datos y Calendario:** Aprobación aséptica sin bloqueos de lectura en la bóveda de Parquet ni carencia de registros en el vector de noticias premium.
> *   **Distribución de Retornos Homogénea:** Retornos distribuidos de forma balanceada a través de distintos regímenes de volatilidad sin concentración extrema en un solo año o trimestre.
> *   **Alineación Forense:** Coincidencia milimétrica y bit a bit en las pruebas de verificación cruzada e independencia de motores.
