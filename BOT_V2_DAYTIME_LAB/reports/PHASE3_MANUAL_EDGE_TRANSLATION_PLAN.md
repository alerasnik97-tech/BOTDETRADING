# PHASE3_MANUAL_EDGE_TRANSLATION_PLAN

## 1. Patrón Maestro: El "NY Sweep Reclaim"
El objetivo es codificar la transición de liquidez de H1 a 3M. 
- **Setup:** Sweep de H1 (Asia/Londres/Día Anterior).
- **Confirmación:** 3M CHoCH (Fractal Objective) + FVG Displacement.
- **Validación:** El bot debe ser "ciego" antes de las 08:00 y después de las 11:30 NY.

## 2. Definiciones Programables (Zero Discretion)
- **H1 Sweep:** Mecha cruza el nivel H1 y el cuerpo cierra por debajo (para shorts) o encima (para longs) en H1 o M15.
- **3M CHoCH:** Quiebre del último fractal (High/Low de 3 velas) en 3M tras el sweep.
- **3M FVG:** Gap entre mecha de vela 1 y vela 3 en un desplazamiento de 3 velas tras el sweep.

## 3. Matriz de Ejecución
Se probarán sistemáticamente:
- **Niveles:** Asia (H/L), Londres (H/L), PDH, PDL.
- **Entradas:** Reclaim simple, CHoCH, FVG, IFVG.
- **Timeframes:** M15, M5, M3, M1 (siendo M3 la prioridad basada en manual).

## 4. Filtros Institucionales
- **Spread:** Bloqueo si spread > 1.5 pips.
- **Noticias:** ±30 min de Red Folders.
- **Rollover:** Bloqueo total 17:00-19:00 NY.

## 5. Meta de Performance
Alcanzar un **PF > 1.35** y una **Expectancy > 0.15R** en 11 años de data certificada, reduciendo la brecha con el 1.88 PF manual mediante selectividad algorítmica.
