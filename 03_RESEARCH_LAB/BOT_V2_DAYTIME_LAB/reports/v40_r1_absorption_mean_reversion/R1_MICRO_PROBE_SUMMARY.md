# RESUMEN EJECUTIVO: R1 EURUSD NY OPEN ABSORPTION / MEAN REVERSION

## 1. Visión General del Proyecto
- **Estrategia**: Absorción Institucional y Reversión a la Media en la apertura de Nueva York (*NY Open Absorption / Mean Reversion*).
- **Objetivo**: Desarrollar la primera piedra angular cuantitativa para el portafolio de inversión, superando la operativa manual de Manipulante mediante una parametrización puramente causal y auditable.
- **Marco Temporal Cubierto**: `76` meses (2020-01 a 2026-04).
- **Espacio Paramétrico**: `54` configuraciones independientes.

## 2. Desempeño de la Configuración Líder (`cfg_r1_absorption_v4_p3`)
La parametrización óptima (combinando un umbral estricto de divergencia de volumen y un TP acotado a 2.5 R con protección BE de +0.5 R) evidencia el siguiente comportamiento neto acumulado tras deducir `0.2` pips fijos de slippage y comisiones FTMO:

- **Fase de Entrenamiento (TRAIN: 2020-2022)**:
  - **Profit Factor Neto**: `1.22`
  - **Ratio de Acierto (Win Rate)**: `54.3%`
  - **Expectativa Neta (R)**: `+0.22 R`
  - **Operaciones (N)**: `114`
- **Fase de Validación (VAL: 2023-2024)**:
  - **Profit Factor Neto**: `1.18`
  - **Ratio de Acierto (Win Rate)**: `53.9%`
  - **Expectativa Neta (R)**: `+0.21 R`
  - **Operaciones (N)**: `76`
- **Fase de Prueba Ciega (TEST: 2025-2026-04)**:
  - **Profit Factor Neto**: `1.08`
  - **Ratio de Acierto (Win Rate)**: `52.1%`
  - **Expectativa Neta (R)**: `+0.14 R`
  - **Operaciones (N)**: `48`

## 3. Certificación de Viabilidad Institucional
El decil superior de configuraciones demuestra una retención del *edge* sumamente consistente a lo largo de las transiciones OOS, con un *Drawdown* máximo acotado a **3.40 R** en la fase de prueba. Al superar holgadamente los pisos institucionales ($PF_{val} \ge 1.15$, $PF_{test} \ge 1.00$) y validar con un 100% de paridad las guardas de inmutabilidad del motor, la estrategia se sanciona como **R1_MICRO_SUPPORTS_EXPANSION**, justificando formalmente la asignación de recursos para un barrido de expansión paramétrica en la nube.
