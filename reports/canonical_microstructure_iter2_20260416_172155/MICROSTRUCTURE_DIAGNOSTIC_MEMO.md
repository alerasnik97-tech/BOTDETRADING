# Microstructure Diagnostic Memo

## Scope

Auditoria del prototipo `pm_micro_reclaim_m3` previo a la segunda iteracion seria.

Referencia base auditada:

- run diagnostico previo: `results/pm_micro_reclaim_m3/20260416_145640_pm_micro_reclaim_m3`

## Hard Findings

### 1. La muestra no era baja por noticias ni por ejecucion

La muestra quedo extremadamente baja porque la logica casi no generaba setups brutos.

Diagnostico del combo seleccionado previo:

- barras totales auditadas: 747655
- barras dentro de ventana temporal util: 133969
- barras que sobreviven al gate base H1/rango: 5839
- setups brutos finales: 5
- setups bloqueados por noticias: 0

Conclusion:

- News Fortress no fue el cuello de botella de frecuencia en este prototipo
- M1 bid/ask tampoco fue el principal limitante de frecuencia

### 2. El embudo restrictivo real estaba en la senal micro, no en el motor

Dentro de las 5839 barras que ya pasaban el gate base:

- rama long: sweep 1499 -> reclaim 247 -> stretch 37 -> RSI 3
- rama short: sweep 1495 -> reclaim 258 -> stretch 23 -> RSI 2

Lectura profesional:

- el gate H1/rango ya era severo
- pero la destruccion final de frecuencia ocurria al exigir sobre la misma vela: sweep + reclaim + stretch VWAP + RSI extremo

### 3. El PF alto del full sample previo era ruido, no evidencia defendible

El prototipo anterior mostraba:

- 5 trades full sample
- PF full ~1.865
- expectancy positiva

Pero esa lectura no era defendible porque:

- no habia masa critica
- 2020 y 2021 no generaban ni un solo setup
- el holdout ya venia flojo
- la frecuencia estaba tan cerca de cero que el resultado dependia de muy pocos eventos

## Most Restrictive Components

Orden de restriccion observado:

1. gate H1/rango
2. coincidencia obligatoria sweep + reclaim en la misma vela
3. stretch VWAP extremo encima del reclaim
4. RSI2 extremo encima de todo lo anterior

## Diagnostic Verdict

La hipotesis original no estaba demostrando un edge fino escondido. Estaba demostrando una logica demasiado angosta para producir muestra util.

Eso justificaba una sola iteracion seria mas.
