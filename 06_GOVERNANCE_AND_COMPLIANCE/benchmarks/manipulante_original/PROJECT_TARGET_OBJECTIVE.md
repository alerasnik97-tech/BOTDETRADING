# OBJETIVO CUANTITATIVO SUPREMO DEL PROYECTO (PROJECT TARGET OBJECTIVE)
**Alineación Estratégica:** Definición del pliego de condiciones de rendimiento y restricciones operativas que toda futura formulación de la familia Manipulante debe satisfacer para autorizar su pase a producción.

---

## 1. Directivas Fundacionales y Filosofía de Diseño
El desarrollo de futuros candidatos de investigación abandona de forma definitiva la traducción literal por fuerza bruta de los parámetros visuales históricos. El mandato de ingeniería cuantitativa impone la formulación de una estrategia **100% objetiva, programable y determinística**, sustentada en la lógica madre de absorción ("Barrido + Cambio de Estructura"), orientada a igualar o superar el benchmark histórico pero asimilando de forma nativa la fricción del mundo real.

### Restricciones de Dominio Inmutables
*   **Universo de Activos:** **EURUSD** exclusivamente. Prohibido extrapolar la lógica a cruces secundarios sin auditoría previa de su microestructura.
*   **Ventana Operativa Máxima:** Búsqueda de señales acotada estrictamente a la franja de **07:00 a 17:00 NY time** (Supresión total de la operativa de tarde-noche y exclusión nativa de la franja diaria de rollover 16:55 a 17:15 NY).
*   **Concentración Recomendada:** Dado que el benchmark manual demuestra que la rentabilidad excepcional ($1.88\text{ PF}$) reside en el **NY Open Killzone (08:00 a 11:00 NY)**, se recomienda encarecidamente acotar los futuros barridos a esta sub-ventana para maximizar el retorno por unidad de tiempo.
*   **Frecuencia Diaria de Ejecución:** Máximo incondicional de **3 operaciones por día calendario** ($\text{max\_trades/day} \le 3$). Un sistema que requiere ametrallar el libro de órdenes para capturar una ventaja carece de viabilidad institucional.

---

## 2. Pliego de Condiciones Mínimas Institucionales (Hard Thresholds)
Cualquier configuración generada por los laboratorios de investigación solo será calificada como *Candidato Aprobado* si su evaluación fuera de la muestra (OOS) supera el siguiente pliego de aserciones de rendimiento neto:

> [!IMPORTANT]  
> *   **Profit Factor Neto en Validación:** $\text{PF}_{\text{val\_net}} \ge \max(\text{Benchmark Original}, 1.15)$. Tomando como referencia la autoridad programada confiable ($\text{PF}_{\text{gross}} = 2.94$), el objetivo supremo es alcanzar un $\text{PF}_{\text{net}} \ge 2.94$ deducidos los costos de fricción. El piso absoluto de viabilidad se fija en $\text{PF}_{\text{net}} \ge 1.15$.
> *   **Preservación en Conjunto Testigo (TEST):** Supervivencia obligatoria en la partición de custodia final: $\text{PF}_{\text{test\_net}} \ge 1.00$ como mínimo absoluto, siendo el estándar ideal $\text{PF}_{\text{test\_net}} \ge 1.15$.
> *   **Esperanza Matemática Neta:** $\text{Expectancy}_{\text{net}} > 0.0\text{ R}$ en todas las particiones evaluadas tras sustraer el modelado completo de comisiones.
> *   **Contención de Drawdown:** Retroceso máximo contenido y estrictamente comparable o inferior a la marca original: $\text{Max DD}_{\text{net}} \le -5.0\text{ R}$.
> *   **Límite de Exposición Diaria:** Cumplimiento riguroso de la restricción de sobreoperar ($\text{trades/day} \le 3$).
> *   **Inmunidad de Capital Prop-Firm:** Cero ocurrencias de quiebre de reglas de pérdida diaria o global en el orquestador FTMO simulado (`ftmo_blown = False`).
> *   **Supervivencia Friccional Obligatoria:** Todas las métricas anteriores deben sostenerse de manera inmutable bajo la adición continua de un estrés de **0.2 pips de slippage asimétrico**.
> *   **Deducción Nativa de Comisiones:** Integración física incondicional del costo institucional de $5.00 por lote round-turn modelado en unidades de $R$.
> *   **Higiene de Equidad:** Cero inyecciones de cierres de fin de mes (EOM) a precios arbitrarios que distorsionen el PnL acumulado.
> *   **Conformidad de Bóvedas:** Aprobación sin advertencias de lectura en el Parquet de mercado ni baches de registros en el calendario de noticias curado.
> *   **Verificación Independiente:** Coincidencia exacta de las curvas de equidad al someter el sistema a un segundo motor de inferencia testigo.
> *   **Aislamiento de Muestra TEST:** Prohibido utilizar el desempeño en la partición TEST para seleccionar, rankear o refinar las variables del modelo.
