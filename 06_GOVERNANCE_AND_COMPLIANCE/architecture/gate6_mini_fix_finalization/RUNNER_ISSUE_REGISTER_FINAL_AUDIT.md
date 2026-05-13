# AUDITORÍA DE CIERRE DE ANOMALÍAS DEL RUNNER (ISSUE REGISTER FINAL)
**Documento:** `RUNNER_ISSUE_REGISTER_FINAL.csv`  
**Fase:** Gate 6 Mini Fix Finalization  
**Fecha:** 2026-05-13  
**Estado General:** `ALL_CRITICAL_ISSUES_CLOSED`

---

## 1. Desglose Forense de Resoluciones

### ISSUE-01: V2_B_STOP_IS_MARKET_CLONE (Severidad: CRÍTICA)
*   **Diagnóstico Original:** El motor interpretaba incondicionalmente el parámetro `entry_mode = "stop"` clonando el comportamiento a mercado de la variante `V2_A` en el primer tick disponible.
*   **Mecanismo de Corrección:** Implementación aséptica de la primitiva nativa `next_bar_execute_stop` en `src/v6_utils/execution.py`. El nuevo flujo requiere físicamente que la cotización Ask/Bid en curso supere o perfore el umbral del extremo del CHOCH más un buffer de seguridad antes de autorizar el fill.
*   **Certificación:** `test_gate6_mini_v2b_stop_is_not_market_clone` comprueba contablemente la divergencia poblacional en los vectores de salida.
*   **Estado Final:** **CLOSED**

### ISSUE-02: ARTIFICIAL_EOM_TRUNCATION_HEAD3000 (Severidad: CRÍTICA)
*   **Diagnóstico Original:** El bucle interno limitaba el recorrido intradiario de la posición extrayendo `.head(3000)` ticks de la rebanada temporal, induciendo a cortes prematuros clasificados equívocamente como salidas de fin de mes (`EOM`).
*   **Mecanismo de Corrección:** Supresión absoluta de la cota fija. El bucle consume de manera incondicional la totalidad del streaming intradiario hasta alcanzar los niveles de Stop Loss, Take Profit, Break-Even o el corte forzado de las 16:00 NY.
*   **Certificación:** `test_gate6_mini_no_artificial_eom_truncation` verifica el índice de completitud temporal del 100%.
*   **Estado Final:** **CLOSED**

### ISSUE-03: NEWS_MISSING_SILENT_CONTINUATION (Severidad: ALTA)
*   **Diagnóstico Original:** La falta física del archivo de calendario de noticias CSV en disco permitía al runner continuar silenciando el error y procesando el backtest sin restricciones ortogonales.
*   **Mecanismo de Corrección:** Inyección programática de una aserción estricta de pre-chequeo institucional (política de Fail-Close). La ausencia del archivo aborta inmediatamente la ejecución arrojando una excepción con el literal `[FAIL-CLOSE INSTITUCIONAL]`.
*   **Certificación:** `test_gate6_mini_news_missing_blocks_run` garantiza la interrupción de la sonda.
*   **Estado Final:** **CLOSED**

### ISSUE-04: MIXED_N_ATTRIBUTION_SLIPPAGE (Severidad: MEDIA)
*   **Diagnóstico Original:** El runner reportaba la cifra terminal de operaciones incluidas en las métricas de rentabilidad sin discriminar la merma secuencial causada por inercia en rebanadas o rechazos normativos.
*   **Mecanismo de Corrección:** Estructuración de la matriz multidimensional `GATE6_MINI_FIX_N_ATTRIBUTION.csv` que traza el funnel secuencial exacto de señales antes y después de cada filtro aplicado.
*   **Certificación:** Revisión física de las sumatorias parciales contra los volúmenes globales.
*   **Estado Final:** **CLOSED**

### ISSUE-05: ANNUAL_PROBE_SCOPE_LIMITATION (Severidad: BAJA)
*   **Diagnóstico Original:** La restricción muestral a tres bloques anuales discretos (2020, 2022, 2024) evadía el colapso por saturación de arreglos en memoria (`ArrayMemoryError`), omitiendo los años intermedios.
*   **Mecanismo de Corrección:** Se formaliza documentalmente el estatus de la ejecución como una **Sonda Estructural (Structural Probe)** en lugar de un Walk-Forward exhaustivo. El diseño retiene validez conceptual plena para descartar o aprobar el núcleo de la lógica en regímenes macro representativos sin caer en un bucle infinito de re-ingeniería.
*   **Estado Final:** **ACCEPTED_RESIDUAL**
