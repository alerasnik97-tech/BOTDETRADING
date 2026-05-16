# Sunday Fix Revalidation Notes

## 1. Implementación Técnica
Se ha aplicado el `REMEDIATION_PLAN_MINIMAL` en `institutional_research_candidate_lab/baseline_truth_model.py`.
- **Lógica:** Si el día actual es lunes y existe una sesión previa de domingo, el sistema busca el viernes previo y colapsa los rangos (Viernes + Domingo) para determinar el PDH/PDL del lunes.
- **Resultado:** Los niveles del lunes ahora reflejan la volatilidad institucional del cierre semanal previo, en lugar de limitarse al rango estrecho del domingo.

## 2. Comparativa Before vs After (Tramo 2026)

| Métrica | Antes del Fix | Después del Fix | Impacto |
| :--- | :---: | :---: | :--- |
| **Total Trades** | 89 | 89 | Neutral (0.0%) |
| **Profit Factor** | 3.067 | 3.057 | Desviación mínima (-0.3%) |
| **Expectancy (R)** | 0.615 | 0.612 | Desviación mínima (-0.4%) |
| **Max Drawdown (R)** | -3.12 | -3.12 | Invariante |
| **Win Rate** | 67.4% | 67.4% | Invariante |
| **Mondays Sum PnL (R)** | +7.14 | +6.88 | Ajuste de realismo (-3.6%) |

## 3. Hallazgos en Niveles (Post-Fix)
- **London (H/L):** Se mantiene como la sesión dominante con un **PF de 5.95** y 35 trades. La corrección de domingos no ha diluido el edge principal.
- **Asia (H/L):** Muestra un **PF de 3.12** con 29 trades.
- **PD (H/L):** El segmento más sensible al fix muestra un **PF de 1.25**. Se observa que PDL es el nivel más débil (PF 0.86).

## 4. Veredicto Institucional
**`SUNDAY_FIX_CONFIRMS_STRONG_2026_VALIDATION`**
La robustez del sistema en 2026 es estructural y no dependía del bug de los domingos. El sistema sigue siendo apto para la fase de micro-piloto manual ultra chico.

## 5. Recomendación Operativa
**Seguir hacia la revisión final del micro-piloto manual ultra chico.** La "limpieza" de los niveles del lunes aporta una mayor confianza en la ejecución manual, ya que los niveles PDH/PDL que el operador verá en su terminal coincidirán con la lógica del laboratorio.
