# PHASE 55A — OPERATIONAL PROHIBITIONS

Queda terminantemente PROHIBIDO realizar las siguientes acciones sin una nueva auditoría forense y aprobación explícita:

1.  **Operar en Cuenta REAL:** No se permite el uso de capital real en `MANIPULANTE` hasta completar la fase de reconciliación de costos forward/demo.
2.  **Uso de Broker Real (Exness/Otros):** El sistema solo está autorizado para entornos Demo o Paper-Trading.
3.  **Modificar la Estrategia:** El núcleo lógico de `MANIPULANTE` está bajo *Freeze* total. No se permiten cambios en señales ni lógica de entrada.
4.  **Optimización de Parámetros:** Queda prohibido ajustar TP, SL, BE o BF basándose en resultados recientes sin un backtest tick-by-tick del universo histórico completo.
5.  **Modificar Policy 19:45 NY:** El cierre forzado a las 19:45 NY es innegociable y obligatorio para todos los trades.
6.  **Operar sin Logging Activo:** No se debe iniciar ninguna sesión de trading si el sistema de `execution_fills.csv` no está operativo.
7.  **Evaluar Rentabilidad Real:** Queda prohibido declarar el sistema como "rentable" en el mundo real hasta que la brecha entre el modelo de ticks y la ejecución del broker sea < 0.05R.

**Vigilancia:** Cualquier violación de estas prohibiciones invalida los resultados de la auditoría Phase 50-55.
