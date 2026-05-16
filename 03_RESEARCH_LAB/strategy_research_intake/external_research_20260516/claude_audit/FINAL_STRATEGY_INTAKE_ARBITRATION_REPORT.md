# FINAL STRATEGY INTAKE ARBITRATION REPORT

## 1. Status
**FINAL_PRIORITY_A_REDUCED_CONSERVATIVE**

## 2. Executive Summary
Este reporte arbitra la contradicción entre el Backlog Certificado por el Claude Principal y las advertencias de riesgo del Auditor Paralelo. Tras una revisión crítica de las lógicas de mercado y los riesgos de implementación, se concluye que el lote "Priority A" debe ser reducido para minimizar el riesgo de "p-hacking" y correlación espuria con la estrategia `Manipulante`. Se aprueba un lote inicial de **3 estrategias** para la fase de skeletons, bloqueando el resto hasta que se provean especificaciones matemáticas completas.

## 3. Evidence Reviewed
- `CLAUDE_STRATEGY_INTAKE_AUDIT_REPORT.md` (Commit 8dd92a67).
- `EURUSD_HYPOTHESIS_BACKLOG_CLAUDE_AUDITED.md`.
- `PARALLEL_RISK_AUDIT_SUMMARY.md` (Transcribed findings).
- Fuentes originales de investigación (6 documentos).

## 4. Conflict Matrix

| candidate | main_claude_decision | parallel_auditor_decision | final_decision | reason |
|-----------|----------------------|---------------------------|----------------|--------|
| **MR-01** | Priority A | Approved (Strong) | **PRIORITY A** | Lógica limpia, OHLCV-only, baja correlación. |
| **VE-01** | Priority B (hallucinated) | Questioned (over-param) | **REVIEW** | Umbrales no anclados a fuentes; requiere simplificación. |
| **TP-01** | Priority A | Questioned (undefined) | **REVIEW** | "Trend Day" y "Giro M1" son términos subjetivos sin reglas. |
| **SD-01** | Rejected | Questioned (correlation) | **PRIORITY B** | Alto solapamiento con `Manipulante`; umbral 0.08 ATR arbitrario. |
| **MR-02** | Priority A | Promote (Clean) | **PRIORITY A** | Reemplazo robusto para VE-01; implementación estándar. |
| **VE-ORB**| Priority A | No explicit objection | **PRIORITY A** | Lógica de sesión clásica; filtro de volatilidad objetivo. |
| **ED-01** | DEFERRED (News) | (N/A) | **DEFERRED** | News Data no certificado. |

## 5. Final Priority A (First Implementation Wave)
1. **MR-01 Anchor Elastic**: Reversión desde extremos del APM (Ancla de Precio Medio).
2. **MR-02 VWAP Stretch Reversion**: Reversión desde ±2.25 SD del VWAP de apertura NY.
3. **VE-ORB Volatility Expansion**: Ruptura de rango 07:00-08:00 NY con filtro ATR(14) > p70.

## 6. Priority B / Review
- **SD-01 Europe Extreme Failure**: Movida a B por correlación con `Manipulante`.
- **VE-01 RV Shock Break**: Movida a REVIEW por parámetros sospechosos (rv5/rv15).
- **TP-01 Trend Day EMA Pullback**: Movida a REVIEW por falta de definición objetiva.

## 7. Deferred News / High Precision
- **ED-01 Post-News Stabilization**: Bloqueada hasta certificación de News Data.

## 8. Rejected / Not First Wave
- **Grid/Martingale variants**: Rechazo absoluto por reglas de fondeo.
- **Micro-scalping (<5 pips)**: Rechazo por sensibilidad extrema a costos.

## 9. Skeleton Implementation Rules
- **No Lookahead**: Los indicadores deben calcularse exclusivamente con datos de `t-1`.
- **Frozen Parameters**: Usar valores indicados en el reporte (no optimizar en esta fase).
- **Signal-Only**: El esqueleto debe devolver {-1, 0, 1} y niveles técnicos de SL/TP.

## 10. Safety Verification
- backtest_run: NO
- strategy_run: NO
- optimization_run: NO
- holdout_used: NO
- 2025_2026_used: NO
- data_modified: NO
- engine_modified: NO
- force_push: NO
- git_add_dot_used: NO

## 11. Copy-Paste Summary for ChatGPT
El arbitraje final ha reducido el lote Priority A a tres estrategias (MR-01, MR-02, VE-ORB) para asegurar la máxima calidad y descorrelación. Se han bloqueado TP-01 y VE-01 por riesgos de sobre-parametrización. Proceder con el prompt: `NEXT_PROMPT_IMPLEMENT_FINAL_APPROVED_PRIORITY_A_SKELETONS.md`.
