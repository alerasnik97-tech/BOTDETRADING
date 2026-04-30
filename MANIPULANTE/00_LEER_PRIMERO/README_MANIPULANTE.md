# MANIPULANTE - Bot de Trading Oficial #1

Bienvenido a la documentacion oficial de **MANIPULANTE**. Siga el orden de lectura para asegurar una operacion profesional y sin errores.

## Orden de Lectura Obligatoria

1.  **[MANIPULANTE_BOT_OFICIAL.md](file:///c:/Users/alera/Desktop/Bot/BOT%20DE%20TRADING%20ultimo/MANIPULANTE/00_LEER_PRIMERO/MANIPULANTE_BOT_OFICIAL.md)**: Identidad, parametros core y autoridad del bot.
2.  **[MANIPULANTE_RESUMEN_PARA_OPERAR.md](file:///c:/Users/alera/Desktop/Bot/BOT%20DE%20TRADING%20ultimo/MANIPULANTE/00_LEER_PRIMERO/MANIPULANTE_RESUMEN_PARA_OPERAR.md)**: Guia rapida para prender, monitorear y apagar el bot cada dia.
3.  **[MANIPULANTE_AUDITORIAS_REALIZADAS.md](file:///c:/Users/alera/Desktop/Bot/BOT%20DE%20TRADING%20ultimo/MANIPULANTE/00_LEER_PRIMERO/MANIPULANTE_AUDITORIAS_REALIZADAS.md)**: Evidencia historica de la robustez del sistema.
4.  **[MANIPULANTE_LECCIONES_APRENDIDAS.md](file:///c:/Users/alera/Desktop/Bot/BOT%20DE%20TRADING%20ultimo/MANIPULANTE/00_LEER_PRIMERO/MANIPULANTE_LECCIONES_APRENDIDAS.md)**: Errores evitados y conocimiento critico para el futuro.
5.  **[MANIPULANTE_FILE_MAP.md](file:///c:/Users/alera/Desktop/Bot/BOT%20DE%20TRADING%20ultimo/MANIPULANTE/00_LEER_PRIMERO/MANIPULANTE_FILE_MAP.md)**: Mapa de carpetas para saber donde encontrar cada archivo.

## Reportes de Auditoria Recientes
- **[MANIPULANTE_COST_AUD_FTMO_REPORT.md](file:///c:/Users/alera/Desktop/Bot/BOT%20DE%20TRADING%20ultimo/MANIPULANTE/14_ANALISIS/MANIPULANTE_COST_AUD_FTMO_REPORT.md)**: Impacto de comisiones y spread (Phase 38B).
- **[MANIPULANTE_DEEP_EXPLAINER_REPORT.md](file:///c:/Users/alera/Desktop/Bot/BOT%20DE%20TRADING%20ultimo/MANIPULANTE/14_ANALISIS/MANIPULANTE_DEEP_EXPLAINER_REPORT.md)**: Analisis cualitativo de la ventaja estadistica.

## Uso Diario (Ciclo Seguro - 3 Botones)
- **START_MANIPULANTE.bat**: Encender. Si `STOP_BOT.txt` estaba activo y no hay riesgo abierto, lo limpia y prende. Si ya esta prendido, no duplica.
- **STATUS_MANIPULANTE.bat**: Mirar.
- **STOP_MANIPULANTE.bat**: Apagar.

START es idempotente: tocarlo una vez o varias veces solo puede iniciar un runner, avisar que ya esta prendido, o bloquear por seguridad. Nunca debe usarse para operar cuentas reales ni Exness.

---
*MANIPULANTE: Disciplina, Estadistica y Preservacion.*
