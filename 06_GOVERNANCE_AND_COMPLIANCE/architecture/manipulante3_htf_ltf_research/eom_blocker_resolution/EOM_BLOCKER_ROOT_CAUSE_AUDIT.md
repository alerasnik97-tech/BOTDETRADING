# EOM BLOCKER ROOT CAUSE AUDIT

Estado auditado: MANIPULANTE3_BLOCKED_BY_DATA_OR_ENGINE

Blocker fisico anterior: EOM artificial = 76.

## 1. Significado de EOM en el runner

En el runner, EOM representa una posicion que no salio por TP, SL, BE o TIME normal dentro de la ventana de ticks entregada al motor. El motor devuelve cierre en el ultimo tick disponible de esa ventana.

EOM no es automaticamente un cierre valido. Solo puede ser valido si la ventana fisica cubre el final previsto de la posicion o si se demuestra fin real de datos.

## 2. EOM valido

EOM valido solo si:
- `eom_type = SESSION_FORCED_EXIT` y `actual_tick_window_end >= intended_position_end`.
- `eom_type = REAL_DATA_END` y se demuestra que el fin de datos es real para el periodo.
- `eom_type = NO_EOM` cuando el trade salio por TP, SL, BE o TIME normal.

## 3. EOM artificial

EOM artificial si:
- La ventana de ticks esta incompleta.
- Hay recorte silencioso.
- Faltan ticks necesarios para cubrir la vida prevista de la posicion.
- Un limite fijo de ticks determina la validez.
- `actual_tick_window_end < intended_position_end`.
- Hay deadline subsegundo mal tratado.
- Hay riesgo timezone/DST o ventana temporal incorrecta.

## 4. Donde se generaron los 76 EOM artificiales

Fuente fisica auditada:
`03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/reports/v38_manipulante3_htf_ltf/maximum_confirmation_rerun/MAX_CONFIRMATION_TRADES.csv`

Distribucion anterior:
- CFG_001 TEST 0.0: 9
- CFG_001 TEST 0.2: 9
- CFG_001 TRAIN 0.0: 3
- CFG_001 TRAIN 0.2: 3
- CFG_001 VAL 0.0: 1
- CFG_001 VAL 0.2: 1
- CFG_002 TRAIN 0.0: 1
- CFG_002 TRAIN 0.2: 1
- CFG_003 TEST 0.0: 6
- CFG_003 TEST 0.2: 6
- CFG_003 TRAIN 0.0: 7
- CFG_003 TRAIN 0.2: 7
- CFG_003 VAL 0.0: 3
- CFG_003 VAL 0.2: 3
- CFG_004 TEST 0.0: 1
- CFG_004 TEST 0.2: 1
- CFG_005 TEST 0.0: 4
- CFG_005 TEST 0.2: 4
- CFG_005 TRAIN 0.0: 3
- CFG_005 TRAIN 0.2: 3

Total: 76.

## 5. Inclusion y distorsion

Los EOM artificiales anteriores estaban incluidos en las metricas principales.

Impacto fisico de esos 76 cierres:
- `sum_net_r = +22.3709R`
- `avg_net_r = +0.2943539474R`
- positivos: 52
- no positivos: 24
- max: +1.8140R
- min: -0.6734R

Conclusion: el blocker invalida la decision anterior como decision final limpia porque los EOM artificiales entraban en PF, expectancy y R total.

## 6. Root cause tecnico

El caso dominante observado fue una ventana de ejecucion demasiado estricta contra un `intended_position_end` con precision subsegundo. En muestras auditadas, el ultimo tick real quedaba milisegundos antes del deadline previsto y el motor marcaba EOM artificial aunque la cobertura era casi completa.

Tambien habia riesgo operativo en cierres cerca de fin de mes porque el runner anterior cargaba el archivo mensual para ejecucion; si la posicion necesitaba ticks del mes siguiente, la ventana fisica podia quedar cortada aunque la estrategia no hubiera pedido ese recorte.

`gate6_mini_runner.py` ya no usa truncado `head(N)` para la ventana de posicion. El riesgo activo estaba en la confirmacion fisica: clasificacion/inclusion de EOM y cobertura de ventana.

## 7. Validaciones especificas

- Configs afectadas: CFG_001, CFG_002, CFG_003, CFG_004, CFG_005.
- Fases afectadas: TRAIN, VAL, TEST.
- Slippage afectado: 0.0 y 0.2.
- Afecta mas a ganadoras: si, 52 de 76 artificiales eran positivas.
- Puede distorsionar PF: si.
- Puede invalidar resultado anterior: si, como decision final limpia.
- Problema de cobertura de ticks: si, por ventana incompleta o deadline subsegundo.
- Problema de ventana temporal: si, en ejecucion y posibles cruces de mes.
- Problema de rollover/forced exit: no confirmado como root cause principal.
- Problema de entry window: no confirmado como root cause principal.
- Problema de stop/limit fill: no confirmado como root cause principal.

## 8. Regla final aplicada

Un trade con EOM artificial no puede entrar en metricas principales. Si no puede simularse con ventana completa, se excluye y se reporta. Si la exclusion deja N insuficiente, la decision pasa a INCONCLUSIVE/BLOCKED, no a RED forzado.

## 9. Resultado despues del fix

Rerun EOM-fixed:
- trades totales: 1895
- trades incluidos en metricas: 1615
- trades excluidos: 280
- artificial EOM total: 280
- artificial EOM en metricas: 0
- exclusion ACTUAL_BEFORE_INTENDED: 276
- exclusion ARTIFICIAL_TRUNCATION: 4
- independent verify: match exacto YES

Conclusion despues del fix:
El blocker de integridad de metricas quedo corregido: no hay EOM artificial incluido en PF, expectancy, winrate, DD o R total. La decision no puede ser RED por razonamiento optimista porque el mejor candidato por VAL queda con N_test insuficiente para lectura final robusta.
