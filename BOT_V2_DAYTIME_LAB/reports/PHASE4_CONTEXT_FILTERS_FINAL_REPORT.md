# Reporte Final: Bot V2 — Fase 4: Filtros de Contexto

## Veredicto Institucional
**`NO_CANDIDATE_FOUND_PHASE4`**

Tras auditar 11 años de datos certificados (2015-2026) con el motor vectorizado V2, la conclusión es que los filtros de contexto automáticos probados mejoran significativamente la entrada técnica, pero no logran alcanzar el estándar de rentabilidad requerido para real/fondeo.

## Hallazgos Clave

### 1. Evolución del Edge (PF)
* **Bot M3 FVG Simple:** PF 0.83 (Perdedor sistemático).
* **Bot M3 SFP + Displacement:** PF 1.13 (Marginalmente positivo, pero con DD alto).
* **Usuario Manual:** PF 1.88 (Punto de referencia).

### 2. Eficacia de los Filtros
* **SFP (Swing Failure Pattern):** Es el filtro individual más potente. Cambiar de "entrar en cualquier barrido" a "entrar solo si hay rechazo (SFP)" elevó el PF de 0.61 a 1.02.
* **Displacement (Fuerza de Reversión):** Al añadir una exigencia de fuerza en la vela de rechazo, el PF subió a 1.13.
* **Niveles Semanales (PWH):** Muestran el mayor potencial individual (PF 1.28), pero con una frecuencia de operativa muy baja (aprox. 10 trades al año).

### 3. El Problema de las Combinaciones
Al intentar cruzar filtros (ej: PWH + DXY Aligned + Asia Small), la muestra estadística cae por debajo de 50-100 trades en 11 años, y el PF se degrada. Esto indica que el bot está "sobre-filtrando" y perdiendo las oportunidades buenas que el usuario manual sí toma basándose en una lectura discrecional de la estructura.

## Recomendaciones Técnicas
1. **No activar trading real.** El bot en su estado actual (automático puro) no tiene ventaja estadística suficiente frente al spread y las comisiones en el largo plazo.
2. **Explorar IA / Machine Learning:** La brecha entre el PF 1.13 y el 1.88 parece ser una cuestión de "lectura de narrativa" que los filtros lineales no capturan.
3. **Preservar el Laboratorio:** Se han documentado todos los fallos y éxitos parciales en `Bot V2`.

---
*Mandato de Fase 4 ejecutado al 100%.*
