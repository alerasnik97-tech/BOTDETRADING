# MANIPULANTE - RESUMEN PARA OPERAR

## Que esperar
MANIPULANTE no gana por tener un winrate alto. Gana por PF, RR, BE rapido y control de perdida. El WR oficial historico es 32.53%, porque los BE cuentan como non-win.

## Como se prende
Usar START_MANIPULANTE.bat. Si ya esta prendido, no iniciar otro runner.

## Como leer STATUS
- OK - BOT ACTIVO: runner vivo y gates principales OK.
- BLOQUEADO - BOT ACTIVO PERO NO OPERA: esta vivo, pero una regla bloquea.
- BLOQUEADO - AUTOTRADING DESHABILITADO: MT5 no permite ordenes automaticas.
- DUPLICADO - LIMPIAR RUNNERS: hay mas de un runner.
- PELIGRO - NO APAGAR PC: hay riesgo operativo o posicion abierta.

## Cuando no tocar nada
No tocar nada si hay noticia, data gate bloqueado, AutoTrading bloqueado, duplicado o posicion abierta.

## Mala racha
La racha non-win maxima fue 14. Puede incluir muchos BE, no necesariamente SL puros. La racha de SL puros es mucho menor.

## Que no cambiar
No cambiar TP 1.4R, BE 0.4R, BF70, max 1 trade/dia ni News/Data gates.

## Si sale duplicado
No operar. Limpiar runners duplicados y volver a STATUS.

## Si AutoTrading esta bloqueado
No forzar ordenes. Revisar el boton Trading algoritmico en MT5. El bot debe quedar bloqueado.

## Antes de apagar PC
Confirmar STATUS: OPERACION ABIERTA NO y SEGURO APAGAR PC SI. Viernes aplica hard close 16:55 NY.
