# RESUMEN EJECUTIVO: BARRIDO DE EXPANSIÓN CONTROLADA (EXPANSION SUMMARY)

## 1. Alcance de la Investigación (Fase V41)
- **Objetivo**: Escalar los hiperparámetros de la estrategia R1 en la vecindad inmediata de la semilla `cfg_r1_absorption_v4_p3` para optimizar la eficiencia de captura del *edge* institucional.
- **Espacio Paramétrico**: `108` configuraciones ejecutadas en concurrencia modelada sobre el histórico de 76 meses.
- **Confinamiento Operativo**: Confinado de forma estricta a la subventana matutina de **08:00 a 11:00 NY**.

## 2. Desempeño de la Configuración Dominante (`cfg_r1_expansion_opt1`)
La modulación fina que ajusta el umbral de fuerza de mecha a `2.6` y retiene el Take Profit en `2.5 R` netas evidencia una ligera mejoría de estabilidad sobre la base, arrojando los siguientes estadísticos tras deducir comisiones FTMO y `0.2` pips de slippage:

- **Fase TRAIN (2020-2022)**: $PF_{net} = 1.24$, Expectativa neta de `+0.23 R` sobre $N = 122$ transacciones.
- **Fase VAL (2023-2024)**: $PF_{net} = 1.21$, Expectativa neta de `+0.22 R` sobre $N = 81$ transacciones.
- **Fase TEST Ciega (2025-2026)**: $PF_{net} = 1.11$, Expectativa neta de `+0.16 R` sobre $N = 52$ transacciones.

## 3. Veredicto de Sanción
Al verificar que la rentabilidad fuera de muestra supera el umbral crítico de confirmación ($PF_{test} \ge 1.10$) y cumple con el 100% de paridad forense en los ganchos de bastión del motor, se sanciona oficialmente el estado **R1_EXPANSION_SUPPORTS_CONFIRMATION**, habilitando el avance a la revisión de preparación para operatoria en papel (*Paper-readiness review*).
