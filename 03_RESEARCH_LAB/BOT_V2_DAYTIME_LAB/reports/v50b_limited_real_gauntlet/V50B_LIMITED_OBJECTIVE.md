# V50B LIMITED OBJECTIVE

**Meta**: Obtener la primera métrica real de desempeño (PF, N, R) para las familias F06, F08 y F12 bajo condiciones operativas de Nueva York y noticias de alto impacto reales.

## Objetivos Especficos
1. **Validacin de Robustez**: Evaluar 50 configuraciones por familia en un periodo de TRAIN (2020-2022) y VAL (2023-2024).
2. **Auditora de Aislamiento**: Garantizar que el motor no arrastre estado (trades abiertos, throttler) entre diferentes configuraciones.
3. **Cribado Causal**: Identificar qué familias sobreviven a los filtros de costos reales (slippage/spread) y noticias de alto impacto.
4. **Ranking de Candidatos**: Generar un ranking Top 20 basado en estabilidad TRAIN+VAL para su posterior expansión.

**Veredicto Esperado**: Identificación de al menos una familia con Edge estadístico real listo para expansión (V50C).
