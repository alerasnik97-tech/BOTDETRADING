# Research Operating System

## Objective

Este documento define como se investiga dentro de este repo sin caer en:

- strategy spam
- sobreoptimizacion
- nostalgia por lineas muertas
- subjetividad disfrazada

## Current Operating Rule

La regla operativa vigente del laboratorio es:

- `risk-first`
- `primitives first`
- `one line at a time`

Hoy eso significa:

- AM Research Habilitado: `READY_FOR_STRICT_AM_RESEARCH`
- `8:00 NY` ABIERTO para Quantitative Research
- baseline horario seguro: `08:00-16:30 NY` (sujeto a kill zones de noticias)
- `17:00-19:00 NY` no se trata como extension neutral por degradacion de ejecucion
- la compuerta `AM final` ha sido auditada y aprobada
- ICT solo via primitives objetivas
- `SMT con DXY` bloqueado hasta tener fuente local auditable

## Research Workflow

### 1. Start From Evidence

Antes de abrir una linea nueva:

- leer `CURRENT_STATE_OF_LAB.md`
- leer `RISK_PROTOCOL.md`
- verificar si la familia ya fue rechazada
- verificar si la nueva idea cambia de verdad la estructura del problema

### 2. Define The Experiment Contract

Ninguna linea se corre sin dejar por escrito:

- hipotesis exacta
- primitives usadas
- timeframe y HTF exactos
- ventana horaria exacta
- entrada exacta
- salida exacta
- hard stop exacto
- politica de noticias
- benchmark de comparacion

### 3. Prefer Primitives Over Narratives

Si la idea necesita frases como:

- "parece sweep"
- "parece CHoCH"
- "parece order block"
- "se nota intencion"

entonces la idea todavia no esta lista para este repo.

Primero se define la primitive.
Despues se prueba la strategy.

### 3B. Manual Edge Is Evidence, Not Template

La operativa manual del usuario puede justificar una nueva linea **solo** si se cumple todo esto:

- se usa como evidencia de dominio, no como receta
- se elimina toda lectura visual no codificable
- se reemplaza cada decision blanda por una regla dura equivalente
- se abre una sola hipotesis, no varias

Quedan fuera por defecto:

- divergencias `DXY` discrecionales
- "se ve limpio"
- switching manual entre `M3 / M5 / M15`
- `CHoCH` narrativo
- segunda bala discrecional

### 4. Open Only A Small Number Of Lines

- una sola direccion principal activa
- como maximo un challenger secundario
- si no hay una hipotesis realmente distinta, no se abre nada

## Validation Standard

Todo experimento serio debe reportar como minimo:

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

La taxonomia operativa del repo queda asi:

- `HARD_REJECT`: linea muerta, no se rescata
- `SOFT_REJECT`: deja aprendizaje, no producto
- `DIAGNOSTIC_ONLY`: aporta estructura o falsacion, no edge
- `PROMISING_CANDIDATE`: supera muestra, PF, expectancy y consistencia
- `LIVE_CANDIDATE`: solo despues de auditoria integral y riesgo blindado

## Kill Rules

Una linea se mata sin romanticismo si ocurre cualquiera de estas condiciones:

- PF no defendible
- expectancy no defendible
- validacion floja
- holdout flojo
- frecuencia miserable
- dependencia de uno o dos trades
- mejora solo cosmetica al mover filtros
- dependencia de noticias mal implementadas

## Anti-Overfitting Rules

- no rescatar lineas muertas con narrativas nuevas
- no cambiar 20 cosas juntas
- no reoptimizar sobre validacion u holdout
- no vender `inf PF` con `1 trade`
- no vender muestra minima como hallazgo

## ICT-Specific Rule

Cuando el tema sea ICT:

- primero se construyen primitives
- despues se testean esas primitives
- despues se corren 2 o 3 setups atomicos
- si esos setups mueren, no se sigue agregando capas por entusiasmo

Casos ya confirmados:

- `IFVG + H1 regime` ya recibio una auditoria seria y tambien fue cerrado
- `pm_volatility_squeeze_retest_m5` tambien fue auditado con muestra suficiente y fue cerrado
- `am_opening_drive_reversal` tambien fue auditado con muestra util y fue cerrado por colapso en validation / holdout
- `am_opening_range_expansion_retest` tambien fue auditado con muestra amplia y fue cerrado por economia negativa en development / validation / holdout

## Benchmark Policy

- `zscore_mean_reversion_pm` sigue como benchmark historico
- el benchmark no se usa como excusa para bajar el estandar
- si el nuevo setup es mejor que un benchmark malo pero sigue siendo malo, se rechaza igual

## Current Active Direction

No hay una estrategia promotable activa hoy.

La direccion de trabajo activa es:

- AM Research habilitado tras cerrar coverage local de `retail sales m/m` y `unemployment claims`
- extender el baseline profesional a `08:00-16:30 NY` usando el motor de News Fortress v3
- conservar e instrumentar primitives ICT objetivas
- congelar la alpha `EURUSD AM` tras tres familias rechazadas
- considerar cerrada tambien la contingencia `EURUSD H1/M15 liquidity horizon`
- tratar `USDJPY` como frente principal solo despues de cerrar readiness pair-specific
- no reabrir `ict_ifvg_repricing_pm`
- no reabrir `pm_volatility_squeeze_retest_m5`
- no abrir otra linea `PM-safe` si solo cambia thresholds sobre causalidades ya derrotadas
- mantener autorizada solo una traduccion manual->objetiva para `EURUSD AM`, no una reapertura indiscriminada de familias AM

## What Merits Real Or Funded Consideration

Solo merece pasar a real o fondeo una linea que:

- sea totalmente objetiva y reproducible
- tenga hard stop y proteccion real frente a noticias
- muestre robustez suficiente entre anios
- tenga muestra suficiente
- no dependa de un solo regimen
- no dependa de una sola narrativa ICT

## Practical Rule For The Next Cycle

Si la siguiente hipotesis no mejora muestra y defendibilidad de forma real, la conclusion profesional no es seguir apilando conceptos.
La conclusion profesional es admitir que esa familia, en esa ventana y bajo ese contrato de riesgo, no esta entregando edge.

## News Fortress Expansion Rule

No se abre expansion horaria por intuicion.

Para reabrir `08:00 NY` debian cumplirse primero estas condiciones (TODAS CUMPLIDAS):
- dataset AM aprobado y auditado: **CUMPLIDO**
- coverage seria de familias `08:15`, `08:30` y `10:00`: **CUMPLIDO**
- gap DST Europa-USA resuelto para eventos EUR: **CUMPLIDO**
- fail-closed verificado en los runners operativos: **CUMPLIDO**

Estado final de esa compuerta:
- arquitectura profesionalizada y auditada forensemente.
- `ecb press conference` cerrada con DST explicito.
- `retail sales m/m` y `unemployment claims` cerrados via manifest auditado.
- `morning_challenge_runner` operativo y habilitado.
- **VEREDICTO: READY_FOR_STRICT_AM_RESEARCH**

## Meta-Decision Outcome

La meta-decision entre:

- una ultima hipotesis `PM-safe` materialmente distinta
- o seguir empujando `AM-grade safety`

ya fue tomada y ejecutada.

Resultado:

- gano primero la ruta `PM-safe distinct hypothesis`
- la hipotesis elegida fue `pm_volatility_squeeze_retest_m5`
- la linea murio con muestra suficiente y economia negativa
- despues se empujo la compuerta AM hasta cerrar todos los gaps locales
- hoy AM esta oficialmente habilitado para investigacion

Consecuencia operativa:

- no abrir otra linea `PM-safe` cosmetica
- AM habilitado tras verificacion forense de sincronizacion total
- la primera linea AM `am_silver_bullet_ny` ya fue rechazada por muestra inutil
- la iteracion `am_silver_bullet_ny_v2` ya fue rechazada por economia negativa con muestra amplia
- `am_opening_drive_reversal` ya fue rechazada: familia distinta, muestra util, economia no estable fuera de muestra
- `am_opening_range_expansion_retest` ya fue rechazada: familia de continuation con muestra suficiente, pero economia estructuralmente negativa incluso en development
- la decision vigente es `NO_GO_STOP_AM_AND_PREPARE_STRUCTURAL_PIVOT`
- el siguiente paso no es `am_silver_bullet_ny_v3`, ni `am_opening_drive_reversal_v2`, ni mini-variantes de `am_opening_range_expansion_retest`, ni una cuarta familia `EURUSD AM` improvisada
- el siguiente paso profesional es un pivot estructural, manteniendo la infraestructura AM y dejando congelada la alpha inmediata sobre `EURUSD`
- la ruta principal recomendada de ese pivot es `instrument-first, USDJPY-first`
- la ruta secundaria recomendada era `EURUSD H1/M15 horizon pivot` solo como contingencia
- esa contingencia ya fue ejecutada via `eurusd_h1_liquidity_sweep_m15` y tambien quedo cerrada
- consecuencia: no corresponde abrir otra variante `EURUSD liquidity sweep` sin una nueva auditoria estructural
- ese audit de readiness ya se corrio y dejo veredicto `USDJPY NOT READY YET — FIX READINESS FIRST`
- por lo tanto, el paso correcto ahora no es una primera familia `USDJPY`, sino cerrar gaps de Fortress pair-specific, defaults/runners y tests antes de disenar alpha
