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

## Estados del Semáforo
- 🟢 **ACTIVO Y SEGURO**: Todo funcionando, monitoreando señal.
- 🟡 **ACTIVO PERO NO OPERA**: Bloqueado por noticias, fuera de horario o sin señal.
- 🔴 **NO ESTÁ CORRIENDO**: El proceso se detuvo o no se ha iniciado.
- 🚨 **NO APAGAR PC**: Riesgo activo. Posición abierta sin confirmar flat.
- 🟣 **REVISAR**: Se detectaron runners duplicados. Ejecute una limpieza.

## Estructura
- `MANIPULANTE\13_FTMO_TRIAL_AUTOMATION\`: Scripts oficiales.
- `MANIPULANTE\10_LOGS_PAPER\`: Logs y latidos.
