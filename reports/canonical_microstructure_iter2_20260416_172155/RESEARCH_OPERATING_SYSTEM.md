# Research Operating System

## Objective

Este documento define como se investiga dentro de este repo sin caer en strategy spam, sobreoptimizacion ni perdida de contexto.

## Research Workflow

### 1. Start From Evidence

Antes de abrir una linea nueva:

- leer `CURRENT_STATE_OF_LAB.md`
- revisar `RISK_PROTOCOL.md`
- verificar si la familia ya fue probada o rechazada
- confirmar que la nueva hipotesis no sea solo un renombre de algo muerto

### 2. Open Only A Small Number Of Lines

- solo una direccion principal activa
- como maximo un challenger secundario
- si no hay razon fuerte para un challenger, no se crea

### 3. Define An Experiment Contract

Todo experimento serio debe fijar por escrito:

- hipotesis exacta
- timeframes usados
- ventana horaria exacta
- reglas exactas de entrada
- reglas exactas de salida
- stop hard
- break-even y time-exit si aplican
- politica de noticias
- benchmark de comparacion

Si alguno de estos puntos queda ambiguo, la linea no esta lista para correrse.

## Validation Standard

Cada experimento debe reportar como minimo:

- numero de trades
- PF
- expectancy
- max drawdown
- retorno por anio
- frecuencia mensual
- comportamiento frente a noticias
- robustez por anio
- sensibilidad leve a parametros
- comparacion honesta contra benchmark previo

## Promotion Logic

La taxonomia operativa del repo sigue esta logica:

- `HARD_REJECT`: muerta, no se rescata
- `SOFT_REJECT`: puede dejar aprendizaje lateral, no producto
- `PASS_MINIMUM`: sobreviviente de investigacion, no candidato a real
- `STRONG_CANDIDATE`: robusta bajo exigencia dura y precision
- `LIVE_CANDIDATE`: solo despues de auditoria completa, news fortress confiable y consenso tecnico serio

Pasar la minima no significa haber encontrado un bot aceptable.

## Kill Rules

Una linea debe matarse sin romanticismo si ocurre cualquiera de estas condiciones:

- PF no defendible
- expectancy no defendible
- OOS flojo
- holdout flojo
- frecuencia trivial que maquilla resultados
- dependencia visible de uno o dos trades
- necesidad de tuning fino para no colapsar
- dependencia de una ventana de noticias mal implementada

## Anti-Overfitting Rules

- no rescatar estrategias muertas con iteraciones cosmeticas
- no cambiar muchas cosas a la vez
- no reoptimizar sobre validacion u holdout
- no usar una mejora pequena como justificacion para seguir indefinidamente
- no aceptar un edge que desaparece bajo precision o bajo noticias bien bloqueadas

## Benchmark Policy

- `zscore_mean_reversion_pm` permanece como benchmark historico del stack viejo
- todo experimento nuevo debe compararse contra ese benchmark y contra la ultima linea activa relevante
- "mejor que el benchmark" no alcanza si ambos siguen siendo economicamente flojos

## Current Active Direction

La direccion principal vigente para este sistema operativo es:

- precision PM / microestructura / exposicion corta / news fortress PM-safe

Direccion secundaria:

- ninguna activa

## What Merits Real Or Funded Consideration

Solo merece pasar a real/fondeo una linea que:

- sea totalmente objetiva y reproducible
- tenga stop hard y proteccion real frente a noticias
- muestre robustez suficiente entre anios y periodos
- tenga muestra suficiente
- no dependa de un solo regimen
- no dependa de fallas del dataset o del motor

## Current Practical Rule

Si el siguiente ciclo de precision no mejora muestra y consistencia de forma seria, la respuesta profesional no es abrir cinco waves nuevas. La respuesta profesional es recomendar un pivot estructural mas fuerte o abandonar la linea investigada.
