# MANIPULANTE - RESUMEN PARA OPERAR (OFICIAL)

Este es el manual rapido para la operacion diaria del bot oficial **MANIPULANTE**.

## 1. Conceptos Clave
- **MANIPULANTE** es el bot oficial. No se llama Phase 25 (esa fue su auditoria).
- El objetivo es la preservacion de capital y el crecimiento estadistico.
- No gane por suerte; gana por repeticion de reglas validadas.

## 2. Ciclo Diario de Trabajo
1.  **Abrir MT5**: Cuenta **FTMO-Demo**.
2.  **Activar Boton**: El icono de "Trading algoritmico" en MT5 debe estar en **VERDE**.
3.  **Encender**: Doble clic en `START_MANIPULANTE.bat`. Dejar la ventana abierta.
4.  **Verificar**: Abrir `STATUS_MANIPULANTE.bat`. Debe decir `ESTADO: OK - BOT LISTO`.
5.  **Cerrar**: Ejecutar `STOP_MANIPULANTE.bat` antes de cerrar MT5 o apagar la PC.

## 3. Estados de Alerta en STATUS
- **BLOQUEADO - NOTICIAS**: El bot no operara hasta que pase el riesgo de noticias.
- **AUTOTRADING DESHABILITADO**: El boton de MT5 esta en ROJO. El bot no podra enviar ordenes.
- **PELIGRO - NO APAGAR PC**: Hay una posicion abierta. Si apaga la PC, la posicion quedara sin gestion de BE/Cierre forzado.
- **DUPLICADO**: Hay mas de un runner. Use STOP para limpiar y reinicie uno solo.

## 4. Que NO se debe hacer
- **NO modificar parametros**: TP 1.4R, BE 0.4R y BF 70% son sagrados.
- **NO operar con 1%**: El riesgo oficial es 0.50% para FTMO Trial.
- **NO cerrar el runner a la fuerza**: Use siempre `STOP_MANIPULANTE.bat`.

## 5. Horarios Sagrados (NY Time)
- **Apertura Ventana**: 07:00 NY.
- **Cierre Ventana**: 16:30 NY.
- **Hard Close Diario**: 19:45 NY.
- **Hard Close Semanal**: Viernes 16:55 NY.

---
*Para mas detalle, consulte `MANIPULANTE_BOT_OFICIAL.md`.*
