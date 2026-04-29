# GLOBAL WEEKEND HARD CLOSE POLICY

**REGLA OBLIGATORIA DEL SISTEMA MANIPULANTE**

## Especificaciones
- **Nombre**: GLOBAL_HARD_CLOSE_BEFORE_MARKET_CLOSE
- **Día de Ejecución**: Viernes
- **Hora Límite**: 16:55 NY (Hora de Nueva York)
- **Alcance**: Universal (todas las cuentas, pruebas, demo, paper, prop firms).

## Instrucciones
1. NO SE MANTIENEN POSICIONES EL FIN DE SEMANA.
2. A las 16:55 NY del viernes, **CUALQUIER** operación abierta de Manipulante debe cerrarse manualmente (o por el MT5 Launcher si estuviese habilitado, pero la regla actual indica cierre manual por el trader en modo demo).
3. NO HAY EXCEPCIONES. NO HAY OVERRIDE MANUAL ("Creo que va a ir a TP el lunes"). SE CIERRA.

Esta regla elimina el riesgo de "weekend gaps" y hace a Manipulante instantáneamente compatible con las reglas más estrictas de prop firms (ej. FundedNext Stellar Lite).
