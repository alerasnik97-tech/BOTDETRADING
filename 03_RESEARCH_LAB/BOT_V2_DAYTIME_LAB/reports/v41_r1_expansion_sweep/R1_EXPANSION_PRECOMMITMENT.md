# MANIFIESTO DE PRE-COMPROMISO METODOLÓGICO DE LA FASE DE EXPANSIÓN (EXPANSION PRECOMMITMENT)

## 1. Reglas de Selección y Evaluación
En apego irrestricto al protocolo de prevención de sesgos (Data Mining / Look-ahead bias), se establecen de antemano las siguientes pautas de calificación para el espacio de vecindad de la configuración líder (`cfg_r1_absorption_v4_p3`):

- **Espacio Paramétrico Desplegado**: El barrido consta exactamente de **108 configuraciones concurrentes** (comprendiendo 3 subventanas horarias $\times$ 3 variaciones finas de ratio mecha-cuerpo $\times$ 3 holguras de parada $\times$ 2 objetivos cercanos $\times$ 2 umbrales de Break Even).
- **Prohibición Ciega OOS**: La porción de datos del tramo de prueba (**TEST: 2025-01 a 2026-04**) queda estrictamente vedada para labores de filtrado, optimización o descarte.
- **Jerarquía de Criterios sobre TRAIN/VAL**:
  1. **Consistencia de Rentabilidad**: Se exige un $PF_{val\_net\_0.2} \ge 1.20$ neto de slippage físico y comisiones FTMO.
  2. **Densidad de Señales**: Un volumen mínimo de operaciones representativo ($N_{val} \ge 50$ en los 24 meses de validación).
  3. **Expectativa Unitaria**: Retención de una expectativa neta positiva y robusta en R puros.
- **Unicidad de la Ejecución de Prueba**: La parametrización o ensamble final que domine en la matriz de selección será sometido a un **único pase definitivo** sobre la muestra `TEST` (*single-run final*). Los estadísticos emergentes de este cómputo ciego serán definitivos e inalterables.
