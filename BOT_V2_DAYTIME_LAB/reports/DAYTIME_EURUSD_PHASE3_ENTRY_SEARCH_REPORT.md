# DAYTIME_EURUSD_PHASE3_ENTRY_SEARCH_REPORT

## 1. RESUMEN EJECUTIVO
Se ha completado la investigación sistemática para traducir el **Edge Manual (PF 1.88)** a un modelo programable institucional. Tras auditar 11 años de datos (2015-2026) y realizar pruebas de paridad en múltiples temporalidades y modelos de entrada, el veredicto es concluyente.

## 2. VEREDICTO FINAL
**Veredicto:** `NO_CANDIDATE_FOUND_PHASE3`

La estrategia programable no logra alcanzar el umbral de robustez de **PF 1.30** requerido para una promoción a fase de fondeo/real. Aunque se identificaron modelos levemente rentables (PF 1.10), la ventaja manual parece residir en filtros discrecionales de alto orden no capturables por las reglas testeadas.

## 3. MÉTRICAS DEL MEJOR CANDIDATO (FVG - M3)
*   **Timeframe:** M3
*   **Modelo de Entrada:** FVG (Fair Value Gap) Post-Sweep
*   **Ventana Operativa:** 08:30 – 11:00 NY
*   **Nivel Reactivo:** PDH (Previous Day High)
*   **Sample Size:** 301 trades (2020-2026)
*   **Profit Factor (PF):** **1.1031**
*   **Expectancy:** +0.0557 R
*   **Win Rate:** 35.55%
*   **Max Drawdown Est.:** 12R

## 4. COMPARATIVA DE TIMEFRAMES (2015-2026)
| Timeframe | Sample | PF | Expectancy | Veredicto |
|-----------|--------|----|------------|-----------|
| M15 | 341 | 1.03 | +0.01 | Breakeven |
| M5 | 491 | 0.91 | -0.04 | Negative |
| M3 | 563 | 1.01 | +0.00 | Breakeven |
| M1 | 684 | 0.99 | -0.00 | Breakeven |

## 5. COMPARATIVA DE MODELOS DE ENTRADA (M3 - 2020-2026)
| Modelo | PF | Win Rate | Veredicto |
|--------|----|----------|-----------|
| Reclaim Simple | 0.98 | 32.9% | Failure |
| CHoCH | 0.86 | 30.2% | Failure |
| **FVG** | **1.10** | **35.5%** | **Best (Weak)** |

## 6. ROBUSTEZ POR PERÍODO (FVG M3)
*   **2015-2019:** PF 0.94 (Inestable)
*   **2020-2022:** PF 1.15 (Positivo)
*   **2023-2026:** PF 1.02 (Breakeven)

## 7. ANÁLISIS DE BRECHA (VS MANUAL)
*   **Edge Manual:** PF 1.88
*   **Edge Programado:** PF 1.10
*   **Conclusión:** La "limpieza" algorítmica de barrer liquidez y entrar en el primer desplazamiento (FVG) captura solo una fracción del edge. El usuario manual probablemente utiliza la **narrativa de H4/D1** o el **News Flow** para filtrar setups, lo cual no fue incluido en esta fase programable pura.

## 8. ACCIÓN FINAL
1.  Se cierra la investigación de Bot V2.
2.  Se preserva el motor `research_v2_engine.py` para futuros filtros de mayor nivel.
3.  **No se recomienda el despliegue de esta familia en real.**

---
**Firmado:** Antigravity AI
**Fecha:** 2026-04-26
**Estatus:** FAIL-CLOSED (Blocked for Safety)
