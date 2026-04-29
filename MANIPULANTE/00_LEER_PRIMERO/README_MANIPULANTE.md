MANIPULANTE es la estrategia principal actual del proyecto. Representa Phase25 Authority con Global Weekend Hard Close. Si hay contradicción entre carpetas, MANIPULANTE y los documentos maestros tienen prioridad operativa.

# MANIPULANTE - LECTURA OBLIGATORIA

Esta carpeta contiene la ÚNICA estrategia aprobada para operar.
Cualquier otra cosa en el proyecto (`ESTRATEGIAS`, `BOT_V2_DAYTIME_LAB`) es archivo, investigación, laboratorio o "shadow comparator" y NO DEBE usarse como estrategia principal.

## Reglas Inquebrantables
1. **NO REAL TRADING**: Manipulante está en fase PAPER/DEMO ONLY.
2. **NO AUTO TRADING**: La ejecución es manual, asistida por indicadores pero con disparo manual. No conectar robots.
3. **GLOBAL WEEKEND HARD CLOSE**: Viernes 16:55 NY toda posición se cierra. NO WEEKEND HOLDING.
4. **NO MODIFICAR PARÁMETROS**: La estrategia es fija. TP 1.4R, BE 0.4R, BF 70%.

## Estructura
- **01_ESTRATEGIA_AUTORIDAD**: Parámetros, configuraciones y reglas base.
- **02_REGLAS_DE_FONDEO**: Análisis de empresas de prop trading.
- **03_MT5_DEMO_LAUNCHER**: Lanzador seguro en modo Demo.
- **04_OPERACION_DIARIA**: Runbook y checklist diario.
- **05_DUAL_LEDGER_SHADOW**: Seguimiento comparativo con la versión BE0.5.
- **06_TEMPLATES**: Archivos CSV/MD en blanco para logs.
- **07_REPORTES_CLAVE**: Índices de los reportes que avalan esta estrategia.
- **08_CHECKLISTS**: Verificaciones pre, post y de fin de semana.
- **09_COMPLIANCE**: Documentación de cumplimiento (weekend rules, prop firms).
- **10_LOGS_PAPER**: Donde debes guardar tus resultados demo.
- **11_GITHUB_SYNC_NOTES**: Registros sobre versiones en Git.

## Límites de Ejecución Manual
Por favor, lea detalladamente el documento de límites: [`BOT_V2_DAYTIME_LAB/docs/MANUAL_EXECUTION_BOUNDARY.md`](../../BOT_V2_DAYTIME_LAB/docs/MANUAL_EXECUTION_BOUNDARY.md)
"El sistema estratégico es 100% objetivo/programable. La fase manual/paper/trial es una fase temporal de ejecución asistida. El usuario no tiene permiso de alterar señales, dirección, TP, BE, SL, noticias, data mask ni gestión por criterio emocional. El trader humano actúa como operador de ejecución y registro, no como fuente discrecional de edge."
