# FORWARD TEST PROTOCOL

Este protocolo define la ejecución táctica durante la fase de incubación.

## Configuración de Ejecución
- **Entorno**: Paper Trading (Simulación) o Demo Controlada. **NUNCA REAL**.
- **Duración Mínima**: 8 a 12 semanas de operación continua.
- **Riesgo por Trade**: Sugerido 0.25% o 0.50% (en simulación).
- **Límites de Frecuencia**: Máximo 3 trades/día.

## Registro Obligatorio (Shadow Ledger)
Cada señal generada por el bot debe ser registrada, independientemente de si se ejecutó o no.
- **Datos de Mercado**: Spread real observado, Slippage real.
- **Contexto**: Hora NY exacta, Noticias cercanas (+/- 30 min).
- **Lógica**: Motivo de entrada (setup), HTF Context, LTF Trigger.

## Procedimientos de Control
- **Registro de Errores**: Desconexiones, latencia alta, rechazo de órdenes.
- **Prohibición de Cambio**: No se permite modificar parámetros del bot durante las semanas de prueba.
- **Pausa Automática**: Si ocurre un error de ejecución crítico, el bot debe detenerse inmediatamente (Kill Switch).

## Monitoreo
- Revisión semanal de discrepancias entre el Backtest y el Forward Test.
- Registro de señales "no tomadas" por el usuario o por restricciones técnicas.
- Capturas de pantalla de los setups más relevantes (opcional pero recomendado).
