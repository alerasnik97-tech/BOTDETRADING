# KILL SWITCH POLICY

El Kill Switch es el protocolo de detención inmediata ante anomalías operativas o riesgos fuera de control.

## Condiciones de Activación Inmediata
La ejecución debe detenerse (cerrar posiciones si es posible y cancelar órdenes) si ocurre:

1. **Error de Ejecución Crítico**: Excepciones en el código del runner o fallos de conexión persistentes.
2. **Orden Duplicada**: Si se detectan múltiples órdenes para la misma señal.
3. **Fuera de Horario**: Apertura de trades fuera de la ventana 07:00-17:00 NY.
4. **Exceso de Frecuencia**: Intentar abrir más de 3 trades en un solo día.
5. **Trade sin Stop Loss**: Cualquier posición abierta que no tenga SL asignado en el servidor del broker.
6. **Fallo de News Fail-Close**: Si el bot opera durante una noticia de alto impacto restringida.
7. **Símbolo Incorrecto**: Intento de operar cualquier par distinto a EURUSD.
8. **Fricción Extrema**: Slippage o spread que supere el 300% del promedio histórico.
9. **Drawdown de Incubación**: Pérdida acumulada en el forward test que supere el límite de seguridad establecido.
10. **Violación de Reglas (x3)**: Tres errores menores o violaciones de protocolo acumuladas.

## Procedimiento de Recuperación
- El bot debe entrar en estado `HALTED`.
- Se requiere un análisis de causa raíz documentado.
- **Solo el usuario** puede reactivar la operativa tras verificar la corrección del error.

## Seguridad FTMO
- El Kill Switch debe priorizar la protección de la cuenta (Daily Loss Limit) sobre cualquier otra lógica.
