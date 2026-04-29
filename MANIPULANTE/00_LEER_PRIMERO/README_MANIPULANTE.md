# MANIPULANTE - Trading Institucional (Phase 25)

## Uso Diario (Ciclo Seguro - 3 Botones)
1.  **Abrir MT5**: Asegúrese de estar conectado a la cuenta **FTMO-Demo**.
2.  **Poner Algo Trading en VERDE**: En MT5, presione el botón "Algo Trading" hasta que se vea verde.
3.  **Iniciar el Bot**: Ejecute `START_MANIPULANTE.bat`.
    - Esta ventana **DEBE QUEDAR ABIERTA**.
    - Si el bot ya está corriendo, avisará y no se duplicará.
4.  **Consultar Estado**: Ejecute `STATUS_MANIPULANTE.bat`.
    - Muestra el panel limpio. Puede cerrarlo cuando quiera.
5.  **Detener el Bot**: Ejecute `STOP_MANIPULANTE.bat`.
    - Es la forma segura de apagar el bot.
    - Crea una señal de parada y espera a que el bot cierre ordenadamente.
    - Si hay una posición abierta, le avisará que NO es seguro apagar.
    - Una vez detenido, puede cerrar MT5 con seguridad.

## Estados del Semaforo
- OK - BOT LISTO: Todo funcionando, monitoreando senal.
- BLOQUEADO - FUERA DE HORARIO: El bot esta activo pero fuera de su ventana operativa.
- BLOQUEADO - NOTICIAS: Operacion pausada por impacto de noticias.
- BLOQUEADO - SIN SENAL: El bot busca activamente una entrada pero no la hay.
- BLOQUEADO - AUTOTRADING: Revisar Trading algoritmico en MT5.
- PELIGRO - NO APAGAR PC: Posicion abierta. No apagar hasta que sea seguro.
- DUPLICADO - LIMPIAR RUNNERS: Mas de un bot corriendo. Limpiar procesos.
- ERROR - REVISAR SISTEMA: El proceso del bot no se detecta.

## Estructura
- `MANIPULANTE\13_FTMO_TRIAL_AUTOMATION\`: Scripts oficiales.
- `MANIPULANTE\10_LOGS_PAPER\`: Logs y latidos.
