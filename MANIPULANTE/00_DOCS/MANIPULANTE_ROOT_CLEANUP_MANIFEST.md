# MANIPULANTE ROOT CLEANUP MANIFEST

Este documento registra la reorganización de la carpeta raíz de MANIPULANTE para profesionalizar el uso diario y mantener una interfaz limpia.

| Archivo Original | Nueva Ubicación | Motivo | Referencias | Riesgo | Acción |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `START_MANIPULANTE.bat.bak_*` | `99_ARCHIVO_BAT_ANTIGUOS\` | Backup redundante de actualización automática de alertas. | No encontradas. | Nulo. | Movido. |
| `STATUS_MANIPULANTE.bat.bak_*` | `99_ARCHIVO_BAT_ANTIGUOS\` | Backup redundante. | No encontradas. | Nulo. | Movido. |
| `STOP_MANIPULANTE.bat.bak_*` | `99_ARCHIVO_BAT_ANTIGUOS\` | Backup redundante. | No encontradas. | Nulo. | Movido. |

## Archivos que permanecen en Raíz
- `START_MANIPULANTE.bat`: Punto de entrada único para el bot y alertas.
- `STOP_MANIPULANTE.bat`: Procedimiento de apagado seguro.
- `STATUS_MANIPULANTE.bat`: Panel de control rápido.

## Notas de Auditoría Operativa
- **Telegram**: El loop está activo y se lanza automáticamente desde `START_MANIPULANTE.bat`. No es necesaria una pestaña manual extra.
- **Seguridad**: No se han movido archivos runtime ni scripts referenciados por los lanzadores principales.
- **Estructura**: Se han creado carpetas `00_DOCS` y subdirectorios para futura documentación prolija.
