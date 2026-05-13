# EOM BLOCKER FIX PLAN

Objetivo: corregir integridad EOM sin cambiar la estrategia ni sus parametros.

Plan aplicado:
1. Congelar configs top 5 ya precomprometidas.
2. Auditar el runner anterior y `gate6_mini_runner.py`.
3. Crear un helper unico de integridad EOM para clasificar artificialidad e inclusion metrica.
4. Reforzar tests especificos de EOM.
5. Reejecutar solo CFG_002, CFG_005, CFG_004, CFG_001, CFG_003.
6. Recalcular metricas desde trades con `included_in_metrics = true`.
7. Bloquear decision si `artificial_eom_in_metrics > 0` o si independent verify no coincide.

No se cambia:
- Parametros.
- Entradas.
- Stop.
- Take profit.
- BE.
- Sesiones.
- Filtros de direccion.
- Seleccion de configs.

Fix tecnico permitido:
- Agregar cobertura de ticks de ejecucion suficiente para resolver la vida prevista de la posicion.
- Incluir el mes siguiente solo para resolver posiciones ya abiertas, no para generar senales.
- Excluir de metricas todo trade con EOM artificial o ventana incompleta.

