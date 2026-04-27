# PHASE 14 REPORT: BEST CANDIDATE SEARCH & TIME RESTRICTION AUDIT

**Veredicto Final:** La restricción horaria de iniciar operaciones a las 07:00 NY **destruye el edge** de todos los modelos de reversión/reclamo analizados. No se ha encontrado ningún candidato diurno que supere PF 1.0 en la ventana 07:00–20:00 NY.

## 1. El Hallazgo Crítico
- **Ventana 03:00–07:00 NY (London Open):** El modelo London Reclaim alcanza un **PF de 7.57**.
- **Ventana 07:00–11:00 NY (NY Open):** El mismo modelo colapsa a un **PF de 0.90**.
- **Impacto:** Operar después de las 07:00 NY transforma una estrategia ganadora en una perdedora debido a la expansión de rangos y la volatilidad direccional de la apertura de Nueva York.

## 2. Bloque A: Estrategias Previas (Window 07:00-20:00)
- **Phase 13 London Reclaim:** PF 0.89 (RECHAZADA para esta ventana).
- **Phase 8 Simulation:** PF 0.79 (RECHAZADA para esta ventana).
- **Phase 7 Repaired:** PF < 1.0 (RECHAZADA para esta ventana).

## 3. Bloque B: Estrategias Nuevas (Window 07:00-20:00)
- **S1 (HTF Sweep):** PF 0.64 (Alta selectividad no rescata el edge).
- **S2 (London Reclaim Continuation):** PF 0.89.
- **S3 (Opening Range Fakeout):** PF 0.88.

## 4. Comparación Global
- **Mejor Candidato Global:** Phase 13 (03:00-07:00 NY) - **PF 7.57**.
- **Mejor Candidato Práctico NY:** Ninguno.
- **Veredicto NY (07:00-20:00):** **NEGATIVO**.

## 5. Conclusión y Riesgos
Intentar forzar una estrategia de reversión diurna que empiece a las 07:00 NY es un riesgo institucional elevado. El mercado diurno en ese horario es predominantemente tendencial o de ruido errático para estos modelos.

---
**Siguiente Paso Único:** Reevaluar la viabilidad de operar la sesión de Londres (03:00-07:00 NY) o pivotar hacia estrategias puras de seguimiento de tendencia (Trend Following) para la sesión de Nueva York.
