# BENCHMARK INSTITUCIONAL CANÓNICO: MANIPULANTE ORIGINAL
**Autoridad Documental:** Sub-sistema de Gobierno y Auditoría Forense  
**Objetivo:** Establecer la línea base inmutable de rendimiento histórico de la estrategia original en sus dos encarnaciones principales (Lógica Manual vs. Traducción Algorítmica Fase 25/27).

---

## 1. Matriz de Rendimiento Comparativo

La siguiente tabla consolida los parámetros operativos y los resultados de backtest reconstruidos a partir de la evidencia de auditoría certificada en el repositorio:

| Métrica de Rendimiento | Manipulante Programado (Full 2015–2026) | Manipulante Programado (Control 2020–2026) | Baseline Manual del Usuario (2020–2026) |
| :--- | :--- | :--- | :--- |
| **Origen / Referencia** | `PHASE27_..._REPORT.md` | `PHASE25_FINAL_CLOSEOUT_REPORT.md` | `MANUAL_VS_BOT_GAP_ANALYSIS.md` |
| **Activo Evaluado** | **EURUSD** | **EURUSD** | **EURUSD** |
| **Período Histórico** | Enero 2015 a Abril 2026 | Enero 2020 a Abril 2026 | Enero 2020 a Abril 2026 |
| **Tamaño de Muestra ($N$)** | **2,625** operaciones | **1,602** operaciones | **841** operaciones |
| **Profit Factor (PF)** | **2.79** (Bruto Base) | **2.94** (Autoridad) / **2.74** (Repro) | **1.88** (Bruto PnL) / **1.53** (Norm. $R$) |
| **Win Rate (WR)** | **32.5%** | **38.5%** / **32.1%** | **35.0%** (R/R promedio $2:1$ a $3:1$) |
| **Expectancy (Esperanza)** | **+0.281 R** | **+0.309 R** | **+0.36 R** (Bruto) / **+0.25 R** (Norm.) |
| **Max Drawdown (DD)** | **-5.58 R** | **-5.0 R** | **UNKNOWN** (Estimado moderado) |
| **Retorno Total Acumulado**| **+737.47 R** | **UNKNOWN** (OOS 2015-19 aportó $+328.6R$) | **UNKNOWN** |
| **Frecuencia Mensual** | **19.4** trades / mes | **19.0** trades / mes (OOS) | $\approx \textbf{11.5}$ trades / mes |
| **Frecuencia Diaria** | $\approx \textbf{0.9}$ trades / día | $\approx \textbf{0.9}$ trades / día | $\approx \textbf{0.55}$ trades / día |
| **Racha de Pérdidas Max**| **14** consecutivas | **UNKNOWN** | **UNKNOWN** |

---

## 2. Parámetros Estructurales y de Gestión de Riesgo
*   **Horario Operativo:** 
    *   *Programado:* **07:00 a 20:30 NY** (Diluyó la ventaja al capturar ruido de tarde).
    *   *Manual:* **08:00 a 11:00 NY** (Concentración absoluta en el *NY Open Killzone*).
*   **Modelo de Salida de Beneficios (TP):** Fijo a **$1.4R$** contractual en la versión consolidada.
*   **Colocación de Stop Loss (SL):** Dinámico en el extremo absoluto de la estructura del barrido fractal.
*   **Gatillo de Break-Even (BE):** Ubicado a una distancia de **$0.4R$** condicionado por un *Body Filter* del $70\%$.
*   **Riesgo Contractual Base:** **$1.0R$** por operación individual.
*   **Política de Cierre Global (Hard Close):** Liquidación forzosa incondicional todos los viernes a las **16:55 NY**.

---

## 3. Inventario de Deducción de Costos y Fricción
*   **Comisión FTMO Incluida:** **NO** de forma nativa en la serie base de la Fase 25/27. Las deducciones realistas ($5.00/lote) se implementaron en fases analíticas posteriores (Fase 38B).
*   **Slippage Incluido:** **NO** en la curva principal. Sin embargo, la Fase 27 aporta una **prueba de estrés asimétrica** excepcional demostrando la supervivencia de la lógica bajo latencia extrema:
    *   $0.0\text{ pips slippage} \rightarrow \text{PF } 2.79$
    *   $0.25\text{ pips slippage} \rightarrow \text{PF } 2.56$
    *   $0.50\text{ pips slippage} \rightarrow \text{PF } 2.37$
    *   $1.00\text{ pips slippage} \rightarrow \text{PF } 2.08$
    *   $2.00\text{ pips slippage} \rightarrow \text{PF } 1.68$
*   **Spread / Bid-Ask Integrado:** **SÍ**, implícito en los datos de reconstrucción de ticks de MT5, aunque sujeto a variaciones de granularidad según el año.
*   **Filtro de Noticias (News Guard):** **SÍ**, implementado nativamente como un escudo *Fail-Closed* suprimiendo ejecuciones en ventanas de volatilidad macroeconómica.
*   **Reglas de Supervivencia FTMO:** **NO** integradas de forma continua en el orquestador base original; validadas externamente mediante simuladores prop-firm en la Fase 31.

---

## 4. Metodología de Validación y Nivel de Confianza
*   **Estructura OOS (Out-of-Sample):** **SÍ**. El tramo histórico **2015–2019** ($1,141$ trades, $\text{PF } 2.86$) actuó como un conjunto de validación puramente OOS sin optimización de parámetros, confirmando que la rentabilidad no fue un artefacto sobreajustado del régimen post-pandemia 2020-2026.
*   **Nivel de Confianza Documental:**
    *   *Manipulante Programado:* **HIGH**. Sustentado en reportes de validación bit a bit y hashes de serialización inmutables (`0d7a18d...`).
    *   *Manipulante Manual:* **MEDIUM**. Extraído de bitácoras visuales del usuario sujetas a sesgo de supervivencia y omisiones de registro en rachas adversas.
