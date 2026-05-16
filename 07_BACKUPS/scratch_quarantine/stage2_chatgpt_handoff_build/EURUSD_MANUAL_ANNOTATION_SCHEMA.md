# EURUSD Manual Annotation Schema

## Propósito

Taxonomía cerrada y disciplinada para capturar explícitamente la lógica de entrada manual del usuario en una muestra curada de trades. El objetivo es construir un bridge entre el edge manual y una futura hipótesis 100% programable.

## Reglas de Uso

1. Solo usar categorías definidas en este esquema
2. No inventar nuevas categorías durante la anotación
3. Si no aplica, usar `none_unclear`
4. Mantener comentarios cortos y objetivos
5. Ser honesto sobre la incertidumbre

---

## A. Fuente Principal de Liquidez Atacada

¿Qué nivel de liquidez fue el trigger principal del trade?

**Opciones:**
- `previous_day_high` - Barrido/ataque al high del día anterior
- `previous_day_low` - Barrido/ataque al low del día anterior
- `asia_high` - Barrido/ataque al high de sesión Asia
- `asia_low` - Barrido/ataque al low de sesión Asia
- `london_high` - Barrido/ataque al high de sesión London
- `london_low` - Barrido/ataque al low de sesión London
- `none_unclear` - No se identifica fuente clara o no aplica

---

## B. Tipo de Trigger Principal

¿Qué patrón técnico triggered la entrada?

**Opciones:**
- `sweep_reclaim` - Barrido de nivel + reclaim (vuelta adentro)
- `sweep_displacement` - Barrido de nivel + displacement fuerte
- `continuation_after_break` - Continuación después de ruptura estructural
- `reversal_after_sweep` - Reversión después de barrido de liquidez
- `breakout_from_compression` - Breakout desde compresión/rango
- `none_unclear` - No se identifica patrón claro o no aplica

---

## C. Tipo de Confirmación Visual

¿Qué confirmación visual consideró válida el usuario?

**Opciones:**
- `close_back_inside` - Cierre vela vuelta adentro del nivel
- `strong_displacement_bar` - Vela de displacement fuerte y claro
- `structure_break` - Ruptura clara de estructura
- `reclaim_then_go` - Reclaim y continuación inmediata
- `immediate_rejection` - Rechazo inmediato del nivel
- `none_unclear` - No se identifica confirmación clara o no aplica

---

## D. Contexto Operativo

¿En qué contexto de sesión operó el trade?

**Opciones:**
- `london_open_drive` - Drive de apertura London
- `london_continuation` - Continuación de sesión London
- `london_reversal` - Reversión durante sesión London
- `pre_ny_transition` - Transición pre-apertura NY
- `early_ny_followthrough` - Followthrough early NY
- `none_unclear` - No se identifica contexto claro o no aplica

---

## E. Motivo Principal de Entrada

¿Cuál fue el motivo principal según el usuario?

**Opciones:**
- `liquidity` - Ataque a liquidez específica
- `displacement` - Displacement fuerte como trigger
- `reclaim` - Reclaim de nivel como trigger
- `time_window` - Ventana temporal específica
- `confluence` - Confluencia de múltiples factores
- `none_unclear` - No se identifica motivo claro o no aplica

---

## F. Calidad Subjetiva Percibida

Evaluación subjetiva de la calidad del setup.

**Opciones:**
- `A` - Setup de alta calidad, muy claro
- `B` - Setup de calidad media, razonable
- `C` - Setup de baja calidad, dudoso

---

## G. Comentario Corto Libre

Comentario breve opcional para añadir contexto no capturado por las categorías anteriores.

**Reglas:**
- Máximo 200 caracteres
- Objetivo y conciso
- Evitar storytelling emocional

---

## Ejemplo de Anotación Completa

```
trade_id: 181989080
A. Fuente de liquidez: asia_low
B. Tipo de trigger: sweep_reclaim
C. Confirmación visual: close_back_inside
D. Contexto operativo: london_continuation
E. Motivo principal: liquidity
F. Calidad: A
G. Comentario: Barrido Asia low 5:15am, reclaim 5:18am, displacement fuerte
```

---

## Notas Técnicas

- Esta taxonomía está diseñada para ser analizable estadísticamente
- Cada categoría es mutuamente excluyente dentro de su grupo
- Los campos son cortos y disciplinados para evitar overfitting
- El objetivo es capturar el núcleo de la lógica de entrada, no todos los detalles
