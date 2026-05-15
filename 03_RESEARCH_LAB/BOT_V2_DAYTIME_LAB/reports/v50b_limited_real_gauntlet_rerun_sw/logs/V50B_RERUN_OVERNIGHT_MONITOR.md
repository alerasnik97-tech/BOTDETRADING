# V50B RERUN OVERNIGHT MONITOR - 2026-05-15
Fecha: 2026-05-15 00:00 (Aprox)

## Runtime
- **Hora**: 2026-05-15 00:00:00
- **PID**: 7668
- **Run ID**: 24bb295d
- **Lock existe**: NO (Carpeta `locks/` vacía, pero proceso activo y escribiendo)
- **Último archivo modificado**: `V50B_RERUN_REJECTION_AUDIT.csv` (23:58:26)
- **Último checkpoint**: F12 | F12_RERUN_0050 | 2024-04 (23:56:58)
- **Familia actual**: F12 (Posiblemente en fase de auditoría final o cerrando buffers)
- **Config actual**: F12_RERUN_0050
- **Mes actual**: 2024-04
- **Señales acumuladas**: 0 (Posiblemente buffered o grabadas en log ausente)
- **Trades acumulados**: 3411
- **Rejections acumuladas**: ~2.1 MB en CSV (Miles de rechazos auditados)

## Safety
- **TEST leakage scan**: PASSED (False positive detectado en UUID, fechas en 2020-2024 OK)
- **Core drift check**: PASSED (Git status clean)
- **Error/Memory**: SI (PID 14492 detectado como proceso muerto/zombie, PID 7668 estable en 865MB WS)

## Notas
El runner parece estar finalizando la familia F12 en el mes de Abril 2024. Se observa que no hay archivo de lock, lo cual sugiere que el runner actual podría no estar usando el sistema de locks de la versión `single_writer_runner.py` o el lock fue borrado accidentalmente. No se iniciará un segundo runner.
