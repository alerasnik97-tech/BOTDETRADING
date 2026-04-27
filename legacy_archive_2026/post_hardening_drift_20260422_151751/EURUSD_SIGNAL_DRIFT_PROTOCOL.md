# EURUSD Signal Drift - Protocol

## Objetivo
Establecer un framework institucional automatizado para detectar desviaciones materiales entre el comportamiento de investigación (Research) y la ejecución en tiempo real (Forward).

## Taxonomía de Drift (Ex-Ante)

### 1. NO_DRIFT
La línea permanece dentro de los intervalos de confianza del 95% de la baseline histórica en todas las dimensiones críticas.

### 2. TOLERABLE_VARIATION
Desviación observable en una dimensión, pero todavía explicable por la variabilidad histórica normal (p.ej. una racha de SL que ha ocurrido antes).

### 3. STRUCTURAL_DRIFT
Desviación persistente fuera de los límites históricos en frecuencia, composición de niveles o distribución de resultados. Sugiere que el edge está cambiando o el mercado ha mutado.

### 4. DATA_OR_PIPELINE_DRIFT
Diferencia causada por cambios en el motor de ejecución, calidad de data o filtros técnicos, no por la lógica de alpha.

### 5. NOT_COMPARABLE_YET
Muestra forward insuficiente (N < 20) para emitir juicio estadístico serio.

## Dimensiones de Monitoreo

### A. Signal Frequency Drift
- **Métrica**: Trades por semana.
- **Límite**: +/- 30% vs promedio histórico.

### B. Level Composition Drift
- **Métrica**: Distribución % de trades por `sweep_level` (London vs Asia vs PDHL).
- **Límite**: Desviación material en el mix (>15% de cambio en pesos).

### C. Directional Drift
- **Métrica**: Ratio Long/Short.
- **Límite**: Desviación del ratio histórico base (p.ej. si era 50/50 y pasa a 80/20).

### D. Performance Distribution Drift
- **Métrica**: PnL medio por trade y Profit Factor en rolling window.
- **Límite**: Caída del PF por debajo de 1.0 sostenida o desviación de 2 sigma del expectancy.

### E. Blocking Profile Drift
- **Métrica**: % de señales bloqueadas por News o Guards.
- **Límite**: Incremento material en la tasa de bloqueo.

## Criterios de Readiness para el Tribunal
Un comparador se considera **READY** si:
1. Ha sido validado contra pseudo-forward histórico sin disparar falsos positivos masivos.
2. Detecta perturbaciones inyectadas de forma deliberada (frecuencia o dirección).
3. Produce outputs JSON legibles por el orquestador institucional.
