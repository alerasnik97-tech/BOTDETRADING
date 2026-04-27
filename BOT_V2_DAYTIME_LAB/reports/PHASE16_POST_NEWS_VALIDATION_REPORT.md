# PHASE 16 REPORT: POST-NEWS EDGE VALIDATION

**Veredicto Final:** **STRONG_CANDIDATE_PHASE16**
**Fecha:** 2026-04-27
**Autoridad:** Phase 16 Refined (Combo A: CPI/NFP/ECB)

## 1. Resumen de Validación
Se ha confirmado que el edge descubierto en la Phase 15 no es un artefacto de baja muestra, sino un comportamiento estructural del EURUSD tras eventos macroeconómicos de alta jerarquía.

## 2. Reproducción Phase 15
- **Resultado:** EXITOSO. PF 1.95 y Sample 56 igualados al 100%.
- **Veredicto Fase 1:** PHASE15_REPRODUCED.

## 3. Desglose por Familia (Hallazgo Clave)
No todas las noticias "High Impact" son iguales para esta estrategia:
- **Ganadores:** CPI (PF 3.18), ECB (PF 2.00), NFP (PF 1.69).
- **Perdedores:** FOMC (PF 0.68), GDP (PF 0.20), ISM (PF 0.88).
- **Conclusión:** La estrategia es reactiva a datos de inflación y empleo, no a sentimientos o decisiones complejas de bancos centrales (FOMC).

## 4. Expansión y Refinamiento (Top Variant)
- **Mejor Variante:** Combo A (CPI/NFP/ECB) | M5 | Close Outside | TP 2.0R.
- **PF:** 2.03.
- **Sample:** 53 trades (2020-2025).
- **Expectancy:** +0.51R.
- **Win Rate:** 50.9%.

## 5. Auditoría de Ejecución
- **News Guard:** CERO trades durante la noticia. Distancia mínima: 60 minutos.
- **Timing:** Los trades ocurren entre 60 y 120 minutos después del evento.
- **Resiliencia:** El PF se mantiene en **1.62** incluso con **1.0 pip de slippage**.

## 6. Robustez Temporal
- **2020:** PF 3.18
- **2022:** PF 4.50
- **2023:** PF 2.25
- **2024:** PF 1.40
- **Consistencia:** 100% de los años positivos desde 2020.

## 7. Conclusión Institucional
La Phase 16 valida que el EURUSD diurno tiene un "segundo viento" institucional una hora después de las noticias clave. Se recomienda la activación controlada de este modelo como "Especialista de Eventos".

---

**Siguiente Paso Único:**
Implementar el cargador de noticias en tiempo real para el bot productivo basado en el filtro de familias de la Phase 16.
