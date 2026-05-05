# PHASE50X — TIME_EXIT POLICY SOURCE LOCK

## Política Operativa Detectada
- **Horario de Cierre**: `19:45` NY (Hora Local Nueva York).
- **Dependencia de DST**: Sí, la regla se basa en la zona horaria `America/New_York`, por lo que se ajusta automáticamente al horario de verano/invierno (Daylight Saving Time).
- **Buffer de Seguridad**: Existe un buffer de 15 minutos antes del `daily_shutdown` (20:00 NY) y 15 minutos antes del rollover bancario (17:00 NY no es el cierre, el cierre es 19:45 NY).
- **Alcance**: Aplica a todos los trades abiertos que no hayan tocado TP/SL/BE antes de las 19:45 NY.
- **Estado Operativo**: **ACTIVO** en demo/forward (`ftmo_trial_only: true`).

## Policy Lock (Canónica de Auditoría)
Para todas las auditorías de MANIPULANTE a partir de PHASE50X:
1. **Regla Primaria**: Se respetan los niveles de `Take Profit` (1.4R), `Stop Loss` (1.0R) y `Break Even` (0.4R) si se alcanzan antes de las 19:45 NY.
2. **Regla de Cierre Forzado**: Si a las 19:45:00 NY el trade sigue abierto, se cierra al primer tick ejecutable disponible.
3. **Precedencia**: La regla operativa 19:45 NY **anula** cualquier `exit_time` histórico posterior o distinto en los datasets de investigación.
4. **Ejecución**: LONG sale al Bid, SHORT sale al Ask.
5. **Lookahead**: Prohibido usar información posterior a las 19:45 NY para trades cerrados por esta regla.
