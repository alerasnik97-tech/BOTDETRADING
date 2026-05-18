# CORRELATION RISK VS. MANIPULANTE — SECURITY & PORTFOLIO INTEGRITY
**Date:** 2026-05-18
**Project:** Systematic Infrastructure Professionalization — Portfolio Correlation and Governance
**Security Status:** READ-ONLY AUDIT & COMPILATION — NO CODE OR REPOSITORY MUTATION

---

## 1. Introducción al Control de Correlación

El objetivo prioritario del owner en el laboratorio cuantitativo es construir un portafolio de estrategias verdaderamente diversificado y robusto, reduciendo el riesgo de drawdowns coincidentes y de canibalización de capital en cuenta real. 

La estrategia principal activa en producción, conocida como `Manipulante` (frecuentemente denominada "Barrido de Liquidez + Cambio de Estructura"), explota la manipulación fractal de precios y la captura de stop-loss en zonas de alta liquidez intradía (máximos y mínimos clave) para entrar en la reversión inmediata de la tendencia. 

Para proteger la integridad de esta estrategia premium de producción, se ha realizado un riguroso análisis causal y analítico de las **30 ideas cuantitativas**, identificando solapamientos y decretando **exclusiones operacionales absolutas** donde exista duplicidad teórica o canibalización.

---

## 2. Matriz de Correlación Teórica y Solapamiento de Señales

```
+-------------------------------------------------------------------------------------------------------------------+
|                                      MATRIZ DE CORRELACIÓN Y SOLAPAMIENTO CON MANIPULANTE                         |
+------+-----------------------+-----------------------+--------------------+---------------------+-----------------+
| ID   | Nombre del Sistema    | Solapamiento Horario  | Duplicidad Teórica | Riesgo de Drawdown  | Clasificación   |
+------+-----------------------+-----------------------+--------------------+---------------------+-----------------+
| MR17 | London Close Rev      | BAJO (11:30 - 16:30)  | NULO               | BAJO                | APROBADA (A)    |
| VE01 | ORB Volatility        | MEDIO (09:00 - 12:00) | BAJO (Breakout)    | BAJO                | APROBADA (A)    |
| MR05 | VWAP Reversion        | ALTO (09:30 - 16:30)  | BAJO (Mean)        | BAJO                | APROBADA (A)    |
| TP12 | Trend Pullback EMA    | ALTO (08:00 - 17:00)  | NULO (Trend)       | BAJO                | APROBADA (A)    |
| VE18 | NY Mid-Day Breakout   | BAJO (12:00 - 14:00)  | BAJO (Breakout)    | BAJO                | APROBADA (A)    |
| SD10 | Asian Fakeout         | MEDIO (07:00 - 11:30) | CRÍTICO (Barrido)  | CRÍTICO (Coincidente)| EXCLUIDA (D)   |
| SD11 | NY Initial Balance    | CRÍTICO (08:30-12:00) | CRÍTICO (Falsos)   | CRÍTICO (Coincidente)| EXCLUIDA (D)   |
+------+-----------------------+-----------------------+--------------------+---------------------+-----------------+
```

---

## 3. Análisis Causal de las Exclusiones Operacionales

### A. Exclusión de Asian Range Liquidity Fakeout (SD10)
*   **Fundamento Técnico de Manipulante:** `Manipulante` busca barridos de stops en máximos y mínimos locales intradiarios, lo que incluye de forma natural el máximo y mínimo de la sesión de Tokio como imanes de liquidez de primer orden al inicio de la sesión de Londres y Nueva York.
*   **Mecanismo de Canibalización:** Habilitar `Asian Range Liquidity Fakeout` (SD10) provocaría que dos algoritmos diferentes intenten operar la misma reversión del rango asiático al mismo tiempo. En caso de una ruptura real fuerte (Trend Day alcista o bajista), ambos algoritmos registrarían pérdidas simultáneas, violando drásticamente el límite de pérdida diaria diaria exigido por FTMO.
*   **Decisión de Gobernanza:** **EXCLUSIÓN ABSOLUTA**. Se prohíbe su codificación para evitar la sobreexposición y la correlación teórica directa.

### B. Exclusión de NY Opening Reversal / Initial Balance Failure (SD11)
*   **Fundamento Técnico de Manipulante:** El Initial Balance (07:00-08:30 NY) define los límites máximos y mínimos de la primera hora y media de Wall Street. Los falsos rompimientos de estos niveles representan barridos estructurales clásicos capturados de forma óptima por el motor de producción `Manipulante`.
*   **Mecanismo de Canibalización:** `SD11` busca disparar la reversión exactamente en el momento en que el precio vuelve a entrar en el rango. Esto causaría un solapamiento masivo de órdenes long/short en el mismo bróker, generando una competencia interna por la liquidez y aumentando los deslizamientos (slippages) innecesarios.
*   **Decisión de Gobernanza:** **EXCLUSIÓN ABSOLUTA**. Protege el capital concentrado en la hipótesis premium de producción.

---

## 4. Por Qué se Prioriza Mean Reversion y Volatility Expansion

Para maximizar la robustez del portafolio consolidado, se ha decretado priorizar las familias con menor correlación lógica con `Manipulante`:

1.  **Mean Reversion VWAP (LCMR-VWAP / Z-Score):** Opera la fatiga dinámica del volumen y el retorno al valor promedio diario del precio. No le interesa si hay o no rompimiento de extremos locales; busca el reequilibrio matemático. Esto proporciona una curva de equity complementaria que asciende en periodos donde `Manipulante` permanece plano.
2.  **Volatility Expansion (ORB / Keltner / Mid-Day):** Son sistemas puramente direccionales y de continuación. Capturan el desbalance de volumen que se desplaza de forma violenta en una sola dirección. Tienen un comportamiento de retornos correlacionado negativamente con las reversiones, amortiguando de forma impecable los drawdowns del portafolio en días de tendencias fuertes unidireccionales (Trend Days).

> [!IMPORTANT]
> **SEGURIDAD DE LA CARTERA:**
> Mantener las exclusiones y retrasar el testeo de sistemas con alta correlación estructural es una directriz de cumplimiento obligatorio e irrenunciable para garantizar la sostenibilidad a largo plazo en cuentas de fondeo de alto valor.
