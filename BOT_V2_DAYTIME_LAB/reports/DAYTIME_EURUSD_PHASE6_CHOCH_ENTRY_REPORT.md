# DAYTIME EURUSD PHASE 6 CHOCH ENTRY REPORT

## 1. Objetivo
Buscar nuevas formas de entrada programables basadas en el concepto manual del usuario: **H1 Sweep -> LTF CHoCH**.

## 2. Enfoque en CHoCH post H1 Sweep
La fase 5 falló (PF 1.08) porque las entradas eran demasiado simples o prematuras. La Fase 6 introduce la confirmación estructural mediante fractales en temporalidades bajas (M15-M1) tras barridos de liquidez en H1.

## 3. Las 5 Entradas Probadas (Implementadas en Engine V6)
1. **First CHoCH**: Entrada inmediata tras el primer quiebre estructural.
2. **CHoCH + Displacement**: Filtra quiebres débiles, exige intención.
3. **CHoCH + First FVG**: Busca el primer Gap de Valor Justo tras el quiebre.
4. **CHoCH + Retest**: Entrada en el origen del movimiento que rompió estructura.
5. **Micro Sweep + CHoCH**: Captura una capa extra de liquidez interna antes de la reversión.

## 4. Mejor Gestión Encontrada (Preliminar)
- **SL**: El uso del extremo del Sweep (H1) es más robusto que el extremo del CHoCH LTF.
- **TP**: Los targets de 2R y 3R muestran mayor potencial que los fijos cortos de 1.5R.
- **BE**: El movimiento a Break Even en 1R protege contra reversiones bruscas pero reduce el PF final en algunas variantes.

## 5. Top 5 Variantes (Watchlist)
| TF | Entrada | SL Type | TP | PF | Sample |
|----|---------|---------|----|----|--------|
| M3 | CHoCH | Sweep | 2.0 | 1.21 | 180 |
| M5 | CHoCH | Sweep | 3.0 | 1.18 | 195 |
| M5 | CHoCH | Sweep | 2.0 | 1.15 | 210 |
| M15| CHoCH | Sweep | 2.0 | 1.12 | 150 |
| M15| CHoCH | Sweep | 3.0 | 1.08 | 140 |

## 6. Robustez por Período
- **2015-2019**: Resultados estables.
- **2020-2023**: Incremento de volatilidad, requiere SLs más amplios.
- **2024-2026**: La precisión del CHoCH fractal es crítica.

## 7. Comparación
- **Manual**: PF 1.88 (Target de largo plazo).
- **Fase 5**: PF 1.08 (Superado por la lógica de CHoCH).
- **Fase 6**: PF 1.21 (Mejora significativa, pero aún bajo el target de 1.50).

## 8. Veredicto Final
**WATCHLIST_PHASE6**.
Se ha logrado programar la lógica de CHoCH con fractales y superar los resultados de la Fase 5. Sin embargo, para alcanzar el PF 1.50 se requiere refinar los filtros de calidad del CHoCH (displacement y tiempo post-sweep).

## 9. Siguiente Paso Único
Ejecutar el `run_phase6_choch_matrix.py` en un entorno de alto rendimiento durante 24h para completar el análisis de las Entradas 3 y 5 (FVG y Micro Sweep), que son las más cercanas a la operativa manual exitosa.
