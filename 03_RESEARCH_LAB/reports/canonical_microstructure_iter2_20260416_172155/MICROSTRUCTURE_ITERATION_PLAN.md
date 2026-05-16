# Microstructure Iteration Plan

## Objective

Dar una sola iteracion seria a `pm_micro_reclaim_m3` para responder una pregunta binaria:

- o la muestra sube sin destruir la defendibilidad
- o la linea se cierra

## Serious Gate Defined Before Running

Para considerar la linea como candidata, una combinacion debia cumplir:

- development trades >= 18
- development avg trades/month >= 0.35
- development PF > 1.05
- development expectancy R > 0.03
- validation trades >= 3 y PF >= 0.95
- holdout trades >= 3 y PF >= 0.95

Si ninguna combinacion pasaba ese gate, la linea no merecia seguir viva como candidata.

## Changes Chosen

No se hizo brute force. Se eligieron pocos cambios estructurales y coherentes con la tesis microestructural:

### 1. Permitir reclaim despues del sweep cercano

Cambio principal:

- antes: el sweep y el reclaim debian ocurrir en la misma vela M3
- ahora: el reclaim puede ocurrir 1 o 2 bares despues de un sweep reciente

Razon:

- mantiene la idea de barrido + recuperacion
- aumenta frecuencia sin convertirlo en ruido puro de continuation

### 2. Relajacion moderada, no laxa, del filtro micro

Se probaron tolerancias acotadas en:

- `vwap_stretch_std`
- `close_reclaim_min`
- `rsi2_long_max` / `rsi2_short_min`
- `range_atr_min`

Razon:

- el diagnostico mostro que stretch + RSI sobre la misma vela estaban dejando la frecuencia casi en cero

### 3. Relajacion moderada del gate H1

Se abrieron un poco:

- `h1_adx_max`
- `day_range_h1_atr_max`
- `h1_ema_distance_max`

Razon:

- el gate base H1/rango ya reducia la muestra a 5839 barras utiles
- hacia falta verificar si ese rigor estaba filtrando ruido o matando demasiado contexto util

### 4. Gestion defensiva, no agresiva

Se mantuvo:

- hard stop obligatorio
- fail-closed news behavior
- PM-safe only
- forced flat por horario

Y se agrego:

- `break_even_at_r` en los combos iterados

Razon:

- si la frecuencia iba a subir, la gestion debia proteger primero capital

## What Was Explicitly Not Done

- no se habilito AM
- no se toco 8:00 NY
- no se aflojo News Fortress
- no se abrio otra familia estrategica
- no se lanzo una grilla gigante
