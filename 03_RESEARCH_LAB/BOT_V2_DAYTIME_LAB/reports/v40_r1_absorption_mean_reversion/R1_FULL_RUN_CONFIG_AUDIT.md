# AUDITORÍA DE INCLUSIÓN PARAMÉTRICA (FULL RUN CONFIG AUDIT)

## 1. Verificación del Espacio de Búsqueda
Se certifica el uso exclusivo y no modificable de las configuraciones emanadas del Readiness Gate, documentadas en `R1_MICRO_PROBE_RUN_CONFIG.json`:
- **Total de Configuraciones Inmutables**: `54`
- **Cero Inyección Dinámica**: Prohibido agregar, quitar o alterar combinaciones de TP/SL/BE durante el barrido.

## 2. Restricciones y Costos Institucionales Activos
- **Activo Exclusivo**: `EURUSD`
- **Ventana Intradía Límite**: `07:00` a `17:00` NY.
- **Subventana de Alta Prioridad**: `08:00` a `11:00` NY (Foco NY Open).
- **Throttling Operativo**: Límite de `max_trades_per_day = 3` estricto por configuración.
- **Deslizamiento Obligatorio**: Slippage fijo de `0.2` pips.
- **Estructura de Comisiones**: Costos FTMO aplicados de forma nativa en la conversión neta.
- **Filtro Macroeconómico**: Datos e impacto de noticias (Data/News) activos.
- **Higiene OOS**: Cero selección, descarte o ajuste basado en la porción de prueba (`TEST`).
