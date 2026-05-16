# Phase 37E: MQL5 Calendar Script Exporter Automation

## Status: BLOCKED_MQL5_SCRIPT_AUTORUN_NOT_AVAILABLE

### Componentes Creados:
1.  **`MANIPULANTE_CalendarScriptExporter.mq5`**: 
    - Ubicación: `MANIPULANTE\03_MT5_DEMO_LAUNCHER\`
    - Función: Exporta noticias de hoy y la semana desde el Calendario Económico oficial de MT5.
    - Filtros: EUR/USD, Alto Impacto (HIGH).
    - Sin funciones de trading (Fail-Closed).

2.  **`phase37e_run_mql5_calendar_script.py`**:
    - Ubicación: Raíz del proyecto.
    - Función: Intenta automatizar la copia, compilación y ejecución del script en MT5.

### Bloqueo Técnico:
La compilación vía línea de comandos (`metaeditor64.exe /compile`) no generó el archivo `.ex5` en este entorno. Esto suele ocurrir por falta de rutas de inclusión (`/inc`) configuradas o permisos de ejecución del terminal desde el script.

### Fallback Fallido:
- **FMP Free**: El API de Financial Modeling Prep requiere una Key válida (la de "demo" retorna 401).

### Acción Requerida (Manual):
Para desbloquear el News Gate 100% automático:
1.  Copie `MANIPULANTE_CalendarScriptExporter.mq5` a su carpeta de `MQL5\Scripts` en MT5.
2.  Compílelo y ejecútelo manualmente una vez.
3.  El sistema de caché detectará los archivos y permitirá el estado `FTMO_TRIAL_AUTO_READY`.

---
**Validación Actual:**
- Cache Hoy: No detectado.
- Cache Semana: No detectado.
- News Gate: NO_TRADE.
- Bot Runner: `NO_TRADE_NEWS_BLOCK`.
