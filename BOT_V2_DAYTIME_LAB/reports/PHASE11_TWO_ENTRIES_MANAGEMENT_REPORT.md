# PHASE 11 TWO NEW ENTRIES + MANAGEMENT OPTIMIZATION REPORT

## 1. Objetivo
Buscar 2 nuevos métodos de entrada eficientes y optimizar la gestión (TP/BE/SL/Timeout) de candidatos previos para elevar el Profit Factor (PF) hacia 1.70.

## 2. Punto de Partida (Fase 0)
- Phase 8: PF 2.09 (Baja frecuencia).
- Phase 7: PF 1.50 (Balanceada).
- Phase 10: PF 1.31 (Selective Fakeout).

## 3. Screening de Métodos Nuevos (Fase 1)
| Método | Lógica | Frecuencia | PF Inicial | Veredicto |
|--------|--------|------------|------------|-----------|
| **M1: Trend Pullback** | H1 Bias + M5 Pullback EMA 20 | ~70/mes | **1.332** | PROCEED TO MGMT |
| **M2: Axis Reversion** | Extensión H1 EMA 50 | ~40/mes | 0.095 | REJECTED_FATAL |
| **M2 V2: Range Fade** | Falso quiebre Rango Asia | ~25/mes | 0.819 | REJECTED |

## 4. Gestión Probada (Fase 2) - Método 1 (Trend Pullback)
Se evaluaron 48 combinaciones de TP (1.0-2.0), BE (0.5-1.0) y Timeouts (1h-4h).
- **Mejor Configuración:** TP 2.0R, BE 0.5R, Timeout 2h.
- **Profit Factor Resultante:** **1.125**.
- **Conclusión:** La gestión avanzada no logra salvar una entrada con edge débil. El BE corta trades que luego van a TP.

## 5. Gestión Probada (Fase 3) - Candidatos Previos
Se aplicó gestión dinámica a Phase 8, Phase 7 y Selective Fakeout.
- **Hallazgo Crítico:** El uso de Break Even (BE) degrada el PF en todos los casos entre un 20% y 40%.
- **Razón Técnica:** El EURUSD diurno tiende a retestear los niveles de entrada múltiples veces antes de expandir. El BE protege el capital pero destruye el edge estadístico de los sweeps.

## 6. Top Candidates
| Candidato | Status | PF Final | Veredicto |
|-----------|--------|----------|-----------|
| **Phase 8 High Precision** | PRESERVED | 2.09 | **STRONG_CANDIDATE** |
| **Phase 7 Repaired** | PRESERVED | 1.50 | **BALANCED_CANDIDATE** |
| **Selective Fakeout V2** | WATCHLIST | 1.31 | **WATCHLIST_ONLY** |
| **Trend Pullback M1** | REJECTED | 1.12 | **REJECTED_PHASE11** |

## 7. Robustez 2023–2025
Los métodos de tendencia pura (M1) sufrieron pérdidas masivas en 2024 debido a los "v-reversals" frecuentes en la sesión de NY. Solo los modelos de reversión tras barrido (Phase 8) mantuvieron un PF > 1.20 en este periodo.

## 8. Sensibilidad a Costos
Con un PF de 1.12 para el mejor método nuevo, el impacto del spread y slippage reduce la expectativa a cero o negativa. No es operable en condiciones reales.

## 9. Veredicto Final
**`NO_CANDIDATE_FOUND_PHASE11`**

La investigación confirma que:
1. No hay atajos mediante gestión para salvar entradas débiles.
2. El Break Even es contraproducente en la lógica institucional de sweeps diurnos.
3. Se mantienen **Phase 8** y **Phase 7** como el límite de lo programable con edge real.

---
*Reporte generado por el laboratorio de optimización Phase 11.*
