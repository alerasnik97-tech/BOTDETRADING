
# PHASE 25: MAX ROBUST PLATEAU + REGRESSION BOUNDARY REPORT

## 1. OBJETIVO
Identificar el techo de robustez del sistema y determinar el punto de regresión donde la optimización se vuelve contraproducente.

## 2. HALLAZGO: TECHO DE ROBUSTEZ (MAX ROBUST PEAK)
Tras auditar 36 variantes, se identifica el punto máximo defendible:
- **Configuración**: **TP 1.4R / BE 0.4R**
- **Profit Factor**: **2.94**
- **Expectancy**: 0.309 R/trade
- **Winrate (TP)**: 38.5%
- **Max Drawdown**: **-5.0 R**

Este punto ofrece un PF superior a la Phase 24 (2.79) manteniendo el mismo nivel de drawdown.

## 3. FRONTERA DE REGRESIÓN
Se detecta una ruptura de robustez a partir de **TP 1.5R**:
- El Drawdown salta de -5.0R a **-8.5R** en configuraciones con BE amplio.
- El Winrate cae por debajo del 32%.
- El PF se estanca en 2.97, lo que no justifica el aumento del 70% en el riesgo de DD.

## 4. VEREDICTO INSTITUCIONAL
**PHASE25_ROBUST_PLATEAU_CONFIRMED**

Se establece **TP 1.4R / BE 0.4R** como la configuración de autoridad para Forward Demo. No se recomienda superar el TP 1.4R bajo el modelo actual.

---
*Firma: PHASE25_AUTHORITY_SIG_20260428*
