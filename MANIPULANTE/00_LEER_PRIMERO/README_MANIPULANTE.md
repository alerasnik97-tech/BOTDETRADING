# MANIPULANTE - Trading Institucional (Phase 25)

## Uso Diario (Ciclo Seguro)
1.  **Abrir MT5**: Asegúrese de estar conectado a la cuenta **FTMO-Demo**.
2.  **Poner Algo Trading en VERDE**: En MT5, presione el botón "Algo Trading" hasta que se vea verde.
3.  **Iniciar el Bot**: Ejecute `START_MANIPULANTE.bat`.
    - **IMPORTANTE**: Esta ventana **DEBE QUEDAR ABIERTA**. Es el motor del bot.
4.  **Consultar Estado**: Ejecute `STATUS_MANIPULANTE.bat`.
    - Esta ventana es solo consulta y puede cerrarla cuando quiera.
5.  **Antes de Apagar la PC**:
    - Abra `STATUS_MANIPULANTE.bat`.
    - Confirme que el semáforo dice **🟢 BOT ACTIVO Y SEGURO**.
    - Confirme que dice **SEGURO APAGAR PC: SÍ**.
    - Si dice **🚨 NO APAGAR PC**, espere a que el bot cierre la posición o intervenga manualmente.

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
