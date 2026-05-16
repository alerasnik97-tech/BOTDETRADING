# Guía de Lanzamiento Local (Entorno Demo) - MONITOREO VIVO

El entorno ahora cuenta con un sistema de **Monitoreo Vivo (Heartbeat)** para asegurar que el bot esté operando correctamente.

## 1. Cómo saber si el bot está corriendo correctamente
Para confirmar el estado de salud del sistema, realiza estas comprobaciones:

1.  **Ventana de Consola (Python):** Deberías ver líneas que comienzan con `[HEARTBEAT]` cada 5 minutos.
    - Ejemplo: `[HEARTBEAT] 09:05:00 | Demo OK | NY=09:05 | Server=11:05 | Positions=0 | Next shutdown=20:30 NY`
2.  **Archivo de Estado Vivo:** Abre `mt5_demo_executor_lab\outputs\mt5_demo_status.json`.
    - Verifica que `last_heartbeat_at` se esté actualizando.
    - Confirma que `executor_status` sea `"RUNNING"`.
    - Revisa `within_runtime_window` para saber si el bot está en horario operativo.
3.  **Logs Históricos:** Revisa `mt5_demo_executor_lab\outputs\mt5_demo_log.csv`.
    - Busca eventos de tipo `HEARTBEAT`.
    - Verifica el spread y la sincronización horaria registrada.

## 2. Auditoría de Horarios
Si tienes dudas sobre la sincronización entre tu PC, Nueva York y el servidor de MT5, consulta:
- `mt5_demo_executor_lab\outputs\mt5_time_audit.md` (Reporte legible).
- `mt5_demo_executor_lab\outputs\mt5_time_audit.json` (Datos técnicos).

## 3. Rutina de Operación
- **07:00 NY:** Iniciar con `START_MT5_DEMO_LOCAL.bat`.
- **Durante el día:** Monitorear el heartbeat ocasionalmente.
- **20:30 NY:** El sistema se detendrá solo. Confirma el cierre limpio en `mt5_demo_status.json`.

---
*Transparencia operativa total para una ejecución demo profesional.*
