# MANIPULANTE - Trading Institucional (Phase 25)

## Acceso Rápido
Para operar en la cuenta **FTMO Trial**, utilice los accesos directos en la raíz de la carpeta `MANIPULANTE`:

1.  **START_MANIPULANTE.bat**: Inicia el bot en modo continuo.
2.  **STATUS_MANIPULANTE.bat**: Abre el panel de control y veredicto de seguridad.

## Estructura de Automatización
- `MANIPULANTE\13_FTMO_TRIAL_AUTOMATION\`: Contiene los scripts oficiales de lógica operativa.
- `MANIPULANTE\10_LOGS_PAPER\`: Logs de ejecución y Heartbeat.

## Seguridad Crítica
- **Fail-Closed**: El bot no opera si no se cumplen todos los gates (Noticias, Spread, Cuenta, etc.).
- **Real Blocked**: El sistema bloquea automáticamente cualquier intento de ejecución en servidores reales o Exness.
- **PC-Off Policy**: No apague la PC si el STATUS muestra `NOT_SAFE_YET`. Espere a confirmar `SAFE_TO_TURN_OFF_PC`.
