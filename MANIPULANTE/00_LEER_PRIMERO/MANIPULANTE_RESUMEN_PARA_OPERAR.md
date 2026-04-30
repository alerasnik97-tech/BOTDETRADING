# MANIPULANTE - RESUMEN PARA OPERAR (OFICIAL)

Este es el manual rapido para la operacion diaria del bot oficial **MANIPULANTE**.

## 1. Conceptos Clave
- **MANIPULANTE** es el bot oficial. No se llama Phase 25 (esa fue su auditoria).
- El objetivo es la preservacion de capital y el crecimiento estadistico.
- No gane por suerte; gana por repeticion de reglas validadas.

## 2. Ciclo Diario de Trabajo
1.  **Abrir MT5**: Cuenta **FTMO-Demo**.
2.  **Activar Boton**: El icono de "Trading algoritmico" en MT5 debe estar en **VERDE**.
3.  **Encender**: Doble clic en `START_MANIPULANTE.bat`. Dejar la ventana abierta mientras el bot trabaja.
4.  **Verificar**: Abrir `STATUS_MANIPULANTE.bat`. Debe decir `ESTADO: OK - BOT LISTO`, `BOT DETENIDO` o un bloqueo claro.
5.  **Cerrar**: Ejecutar `STOP_MANIPULANTE.bat` antes de cerrar MT5 o apagar la PC.

## 2.1 START simple e idempotente
- Si el bot esta apagado, `START_MANIPULANTE.bat` lo prende.
- Si estaba detenido por `STOP_BOT.txt`, START lo reactiva de forma segura solo despues de confirmar cuenta FTMO-Demo y sin posicion abierta.
- Si ya estaba prendido, START no duplica runners y muestra `BOT YA ESTA PRENDIDO`.
- Si hay una operacion abierta, START no limpia `STOP_BOT.txt` y bloquea con `PELIGRO - OPERACION ABIERTA`.
- Si detecta cuenta real, Exness o una cuenta que no sea FTMO Demo, START no inicia.

## 3. Estados de Alerta en STATUS
- **BLOQUEADO - NOTICIAS**: El bot no operara hasta que pase el riesgo de noticias.
- **AUTOTRADING DESHABILITADO**: El boton de MT5 esta en ROJO. El bot no podra enviar ordenes.
- **PELIGRO - NO APAGAR PC**: Hay una posicion abierta. Si apaga la PC, la posicion quedara sin gestion de BE/Cierre forzado.
- **DUPLICADO**: Hay mas de un runner. Use STOP para limpiar y reinicie uno solo.
- **BOT DETENIDO**: `STOP_BOT.txt` esta activo y no hay runner. Use `START_MANIPULANTE.bat` para reactivar si es seguro.

## 4. Que NO se debe hacer
- **NO modificar parametros**: TP 1.4R, BE 0.4R y BF 70% son sagrados.
- **NO operar con 1%**: El riesgo oficial es 0.50% para FTMO Trial.
- **NO cerrar el runner a la fuerza**: Use siempre `STOP_MANIPULANTE.bat`.
- **NO borrar STOP_BOT a mano si hay posicion abierta**: Primero revise MT5 y STATUS.

## 5. Horarios Sagrados (NY Time)
- **Apertura Ventana**: 07:00 NY.
- **Cierre Ventana**: 16:30 NY.
- **Hard Close Diario**: 19:45 NY.
- **Hard Close Semanal**: Viernes 16:55 NY.

## 6. Que hacer si cerre la ventana de START sin querer
Si cerraste la ventana negra de `START_MANIPULANTE.bat` pero el bot seguia trabajando:

1. **Abrir STATUS**: Mira el panel de estado.
2. **Si dice BOT: ACTIVO**: No hagas nada. El bot sigue operando en segundo plano aunque hayas cerrado la ventana.
3. **Si dice LOCK VIEJO**: Significa que el bot se cerro mal y dejo un rastro.
4. **Solucion**: Ejecuta `STOP_MANIPULANTE.bat` y luego vuelve a ejecutar `START_MANIPULANTE.bat`.
5. **Seguridad**: START ahora detecta rastros viejos y los limpia automaticamente si la cuenta esta libre de operaciones.
6. **Importante**: STOP no cierra MT5, solo detiene el proceso del bot de forma segura.

---
*Para mas detalle, consulte `MANIPULANTE_BOT_OFICIAL.md`.*
