# MARCO DE EVALUACIÓN COMPARATIVA CONTRA BENCHMARK (TARGET COMPARISON FRAMEWORK)
**Alineación Documental:** Establecimiento de la rúbrica de auditoría estricta para contrastar objetivamente el desempeño de futuros motores de investigación frente a la línea base histórica canónica de la estrategia Manipulante.

---

## 1. Vectores Obligatorios de Comparación
Para que el dictamen de una nueva estrategia sea auditable, el equipo de investigación debe reportar de forma tabular y lado a lado las siguientes 15 dimensiones operativas frente a los valores correspondientes del Benchmark Original (Fase 25/27 o Baseline Manual):

1.  **Profit Factor Neto ($\text{PF}_{\text{net}}$):** Contraste directo del factor de beneficio tras deducir la totalidad de las fricciones simuladas frente al $\text{PF}$ bruto histórico.
2.  **Tasa de Acierto (Win Rate - $\text{WR}$):** Eficiencia en la captación de retornos positivos relativos al número total de entradas.
3.  **Drawdown Máximo Neto ($\text{Max DD}_{\text{net}}$):** Profundidad del peor retroceso de capital medido en unidades estandarizadas de riesgo ($R$).
4.  **Esperanza Matemática Neta ($\text{Expectancy}_{\text{net}}$):** Retorno medio esperado por transacción individual deducidos los costos de corretaje.
5.  **Retorno Total Acumulado ($\text{Total } R$):** Sumatoria del PnL en la ventana de evaluación superpuesta.
6.  **Densidad Operativa Mensual ($\text{Trades/Month}$):** Consistencia en la frecuencia de disparo a través de los meses del calendario.
7.  **Densidad Operativa Diaria ($\text{Trades/Day}$):** Medición de la exposición intradiaria al mercado.
8.  **Estabilidad Mensual (Monthly Consistency):** Proporción de meses cerrados con equidad positiva sobre el total de la serie histórica.
9.  **Sensibilidad al Deslizamiento (Slippage Decay):** Tasa de degradación del Profit Factor al someter la curva a la matriz de estrés de latencia ($0.0$ a $2.0$ pips).
10. **Cumplimiento Contractual FTMO (Prop-Firm Compliance):** Certificación de supervivencia sin quiebres de límites de pérdida diaria o máxima global.
11. **Cumplimiento de Escudo de Noticias (News Compliance):** Verificación de asimilación de buffers Tier-1 y consumo del calendario premium curado.
12. **Robustez Fuera de la Muestra (OOS Robustness):** Desempeño contrastado en segmentos históricos puramente de validación o no observados en el diseño.
13. **Riesgo de Minería de Datos (Data Mining Risk):** Ponderación cualitativa y cuantitativa de la cantidad de variables libres y grados de libertad del hiper-cubo explorado.
14. **Similitud Causal Lógica (Logical Proximity):** Grado de fidelidad conceptual frente al núcleo discrecional de absorción ("Barrido + Cambio de Estructura").
15. **Divergencia Lógica (Logical Divergence):** Explicación arquitectónica de las innovaciones ortogonales inyectadas (ej. filtros de régimen o temporalidades dinámicas).

---

## 2. Cláusulas de Exclusión y Falsos Positivos (Regla de Dominancia Invalida)

> [!CAUTION]  
> El Sub-sistema de Gobierno declara formalmente que **UNA ESTRATEGIA CANDIDATA NO SUPERA AL BENCHMARK ORIGINAL** (y su pase a Forward Demo queda automáticamente bloqueado) si su aparente superioridad en métricas brutas deriva exclusivamente de incurrir en cualquiera de los siguientes **8 vicios metodológicos prohibidos**:
> 
> *   **Sobreoperar (Overtrading):** Generar una cantidad artificialmente elevada de transacciones para inflar el retorno nominal acumulado a expensas de degradar la esperanza matemática por trade.
> *   **Omisión de Costos de Corretaje:** Reportar métricas brutas sin sustraer el modelado físico de comisiones FTMO ($5.00/lote).
> *   **Evaluación Idealizada sin Latencia:** Extraer curvas de capital asumiendo un entorno sin fricción (`slippage = 0.0`), ignorando la penalización obligatoria de $0.2$ pips.
> *   **Violación de la Ventana Horaria:** Disparar órdenes fuera del pliego horario permitido (**07:00 a 17:00 NY**), capturando movimientos erráticos nocturnos o asumiendo fills ideales durante la franja ilíquida de rollover.
> *   **Exceso de Densidad Diaria:** Infringir la cota máxima institucional de ejecuciones intradiarias ($\text{max\_trades/day} > 3$).
> *   **Sesgo de Selección por Conjunto Testigo (TEST Selection):** Sintonizar o elegir la configuración ganadora basándose en haber observado previamente su rendimiento en la partición reservada TEST.
> *   **Dependencia de Retornos Atípicos (Outlier Dominance):** Sustentar el Profit Factor en un puñado muy reducido de transacciones con un recorrido explosivo de cola larga que enmascaran un comportamiento mediano perdedor.
> *   **Carencia de Auditoría Independiente:** Incapacidad de reproducir las curvas de equidad de manera determinística al ejecutar la lógica en un orquestador secundario testigo.
