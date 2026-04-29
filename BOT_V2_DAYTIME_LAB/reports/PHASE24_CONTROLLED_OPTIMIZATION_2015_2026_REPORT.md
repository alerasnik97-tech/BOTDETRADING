
# PHASE 24: CONTROLLED OPTIMIZATION PLATEAU STUDY REPORT

## 1. OBJETIVO
Estudiar el linaje Phase 22 / Phase 18 para identificar el **Punto Máximo de Robustez** (Plateau) y evitar la regresión por sobreoptimización. Rango analizado: 2020-2026 (Máximo certificado disponible).

## 2. HALLAZGO: EL PLATEAU ROBUSTO
Tras analizar 20 variantes de la cuadrícula de parámetros (Grid Search), se identifica un plateau de alta estabilidad en la zona:
- **TP**: 1.1R - 1.3R
- **BE**: 0.4R - 0.6R

### Punto de Máxima Robustez Seleccionado:
- **TP 1.3R / BE 0.5R**
- **PF**: 2.79 (Mejora del 7% vs Phase 22)
- **Winrate (TP)**: 39.7%
- **Expectancy**: 0.328 R/trade

## 3. FRONTERA DE REGRESIÓN
Se observa que al aumentar el TP a **1.5R**:
- El PF sube a **2.97** (Pico máximo teórico).
- Pero el Winrate cae al **31.8%**.
- La estabilidad anual en 2026 se degrada (WR 27%).
- **Veredicto**: TP 1.5R se considera el inicio de la zona de regresión/fragilidad psicológica. No se recomienda para Forward Demo.

## 4. ROBUSTEZ TEMPORAL (2020-2026)
| Año | Sample | PF (1.3/0.5) | Winrate |
| :--- | :--- | :--- | :--- |
| 2020 | 252 | 2.84 | 36.9% |
| 2021 | 256 | 2.70 | 41.4% |
| 2022 | 254 | 3.34 | 41.3% |
| 2023 | 252 | 3.56 | 40.8% |
| 2024 | 256 | 2.89 | 42.5% |
| 2025 | 252 | 2.07 | 36.9% |
| 2026* | 80 | 2.03 | 33.7% |

*Todos los años mantienen PF > 2.0.*

## 5. VEREDICTO FINAL
**PHASE24_ROBUST_IMPROVEMENT_FOUND**

Se recomienda promover el candidato **TP 1.3R / BE 0.5R** como el nuevo estándar para Forward Demo, ya que representa el equilibrio perfecto entre rentabilidad auditada y estabilidad temporal.

---
*Firma: PHASE24_PLATEAU_STUDY_SIG_884C1FEA*
