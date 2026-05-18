# RESEARCH SYNTHESIS INDEX — INTEL INTAKE MASTER HUB
**Date:** 2026-05-18
**Project:** Quantitative Research Laboratory (EURUSD Intraday)
**Security Status:** READ-ONLY AUDIT & COMPILATION — NO CODE OR REPOSITORY MUTATION

---

## 1. Propósito de la Síntesis

Este documento sirve como **Índice Maestro y Punto de Control Central** para toda la investigación cuantitativa externa incorporada al proyecto. Integra y sintetiza los hallazgos de las auditorías de Claude Opus 4.7, los reportes de profesionalización de ChatGPT/Manus, y el backlog de hipótesis de estrategias sobre EURUSD en el tramo horario de Nueva York (**07:00 – 19:00 NY**).

El objetivo es proveer al Owner un mapa unificado, libre de duplicados, alineado con las restricciones del sistema `Manipulante` y estructurado para prevenir el autoengaño y el sobreajuste.

---

## 2. Mapa de Navegación de la Síntesis

Los entregables creados en esta campaña de síntesis se organizan en los siguientes módulos, localizados en el directorio exterior `C:\Users\alera\Desktop\AGENTE_READONLY_RESEARCH_SYNTHESIS_AUDIT\`:

```mermaid
graph TD
    A[RESEARCH_SYNTHESIS_INDEX.md] --> B[STRATEGY_FAMILY_CANDIDATE_MAP.md]
    A --> C[TOP_STRATEGY_IDEAS_PRELIMINARY_RANKING.md]
    A --> D[QUANT_PROJECT_GROWTH_PRINCIPLES.md]
    A --> E[RISKS_OVERFITTING_AND_SELF_DECEPTION.md]
    
    style A fill:#1a365d,stroke:#3b82f6,stroke-width:2px,color:#fff
    style B fill:#0f172a,stroke:#3b82f6,color:#fff
    style C fill:#0f172a,stroke:#3b82f6,color:#fff
    style D fill:#0f172a,stroke:#3b82f6,color:#fff
    style E fill:#7f1d1d,stroke:#ef4444,stroke-width:2px,color:#fff
```

### Documentos del Suite de Síntesis:
1.  **[EXISTING_WORK_AUDIT.md](file:///C:/Users/alera/Desktop/AGENTE_READONLY_RESEARCH_SYNTHESIS_AUDIT/EXISTING_WORK_AUDIT.md):** Reporte formal de existencia y brechas de la auditoría inicial.
2.  **[RESEARCH_SYNTHESIS_INDEX.md](file:///C:/Users/alera/Desktop/AGENTE_READONLY_RESEARCH_SYNTHESIS_AUDIT/RESEARCH_SYNTHESIS_INDEX.md):** (Este documento) Punto de control, glosario y mapeo de referencias cruzadas.
3.  **[STRATEGY_FAMILY_CANDIDATE_MAP.md](file:///C:/Users/alera/Desktop/AGENTE_READONLY_RESEARCH_SYNTHESIS_AUDIT/STRATEGY_FAMILY_CANDIDATE_MAP.md):** Estructura taxonómica de familias de estrategias, lógica operativa, restricciones de datos y presupuesto de correlación frente al núcleo `Manipulante`.
4.  **[TOP_STRATEGY_IDEAS_PRELIMINARY_RANKING.md](file:///C:/Users/alera/Desktop/AGENTE_READONLY_RESEARCH_SYNTHESIS_AUDIT/TOP_STRATEGY_IDEAS_PRELIMINARY_RANKING.md):** Clasificación detallada de las 10 familias top de investigación y un backlog consolidado de 20 ideas con criterios cuantitativos listos para su posterior testeo.
5.  **[QUANT_PROJECT_GROWTH_PRINCIPLES.md](file:///C:/Users/alera/Desktop/AGENTE_READONLY_RESEARCH_SYNTHESIS_AUDIT/QUANT_PROJECT_GROWTH_PRINCIPLES.md):** Hoja de ruta arquitectónica basada en plataformas institucionales (LEAN, NautilusTrader, Freqtrade, Hummingbot) y requerimientos de firmas de fondeo (FTMO).
6.  **[RISKS_OVERFITTING_AND_SELF_DECEPTION.md](file:///C:/Users/alera/Desktop/AGENTE_READONLY_RESEARCH_SYNTHESIS_AUDIT/RISKS_OVERFITTING_AND_SELF_DECEPTION.md):** Manual metodológico para blindar el laboratorio contra el leakage temporal, el lookahead bias y los falsos descubrimientos.

---

## 3. Matriz de Referencias Cruzadas (Fuentes Ingeridas)

La investigación externa consolidada se basa en 6 fuentes primarias legibles y stubs de documentación técnica estructurada en el repositorio local `BOT DE TRADING ultimo`:

| Código de Fuente | Nombre de Archivo / Directorio | Aporte Clave a la Síntesis |
| :--- | :--- | :--- |
| **[F1]** | `EURUSD 07_00-19_00 NY Strategy Research Report GPT.pdf` | Especificaciones de parámetros verbatim de la Priority A (MR-01, VE-01, TP-01). |
| **[F2]** | `EURUSD 07_00-19_00 NY Strategy Research Report.pdf` | Validación de umbrales horarios y comportamientos estacionales. |
| **[F3]** | `EURUSD_Strategy_Research_Report.md` | Lógica de expansión por canales de Donchian y cruces dinámicos de VWAP. |
| **[F6]** | `Investigación Estrategias Algorítmicas EURUSD.pdf` | Definición de MR-02 (VWAP Stretch) y estimación de correlación con sweeps de liquidez. |
| **[K1]** | `knowledge_intake/external_quant_project_growth_20260518/` | Diagnósticos comparativos de software cuantitativo institucional (ChatGPT y Manus). |
| **[S1]** | `strategy_research_intake/external_research_20260516/` | Backlog auditado por Claude Opus 4.7 y reportes de deltas Gemini-to-Claude. |

---

## 4. Clasificación de Decisiones Técnicas (Gates de Gobernanza)

Para facilitar la planificación de las próximas campañas de investigación, hemos unificado los veredictos de gobernanza en una taxonomía de 4 estados, aplicados a todas las hipótesis en el mapa de candidatos:

*   **[IMPLEMENTABLE_NOW_OHLCV]:** Estrategias que solo requieren datos OHLCV en barras estándar de M1/M5/M15. No dependen de noticias macro ni de datos de alta precisión (Bid/Tick/Spread). Prioridad de codificación inmediata para tests de señal.
*   **[IMPLEMENTABLE_AFTER_SPEC_REFINEMENT]:** Estrategias robustas pero que requieren blindar su especificación matemática contra lookahead bias (por ejemplo, cálculo de percentiles y medias con ventanas estrictamente rodantes hacia atrás).
*   **[DEFER_NEWS]:** Hipótesis de alta calidad que dependen del calendario y releases macro. Quedan diferidas de forma fail-closed hasta que el `News Fortress` esté plenamente certificado en el `05_MARKET_DATA_VAULT`.
*   **[DEFER_HIGH_PRECISION]:** Estrategias que requieren cotización bid/ask, control dinámico de spread o validación tick-by-tick. Diferidas hasta contar con el motor de cotización en vivo y simulación de latencia/ reality modeling.
*   **[REJECT]:** Estrategias descartadas por poseer reglas discrecionales, inviabilidad matemática o un perfil de sobreajuste estructural imposible de mitigar.

---

## 5. Próximo Paso Recomendado para el Owner

1.  **Revisión del Mapa de Familias:** Leer [STRATEGY_FAMILY_CANDIDATE_MAP.md](file:///C:/Users/alera/Desktop/AGENTE_READONLY_RESEARCH_SYNTHESIS_AUDIT/STRATEGY_FAMILY_CANDIDATE_MAP.md) para comprender la correlación esperada de las nuevas ideas con `Manipulante` y cómo evitar canibalizar el presupuesto de riesgo de la cuenta.
2.  **Aprobación del Plan de Implementación de Prioridad A:** Iniciar la codificación de las señales base de las cuatro estrategias de Priority A aprobadas por Claude (MR-01, MR-02, TP-01, VE-01) en entornos aislados de `03_RESEARCH_LAB`, respetando los protocolos anti-lookahead.
