# PRE-CLEANUP SNAPSHOT

**Fecha:** 2026-04-27
**Mandato:** Registro del estado del proyecto antes de la limpieza institucional.

## 1. Inventario de la Raíz (Resumen)
- **Documentos Sueltos (.md/.json):** ~150 archivos detectados.
- **CSVs Sueltos:** ~25 archivos detectados (incluyendo resultados de campañas y auditorías manuales).
- **ZIPs:** 2 archivos `000_PARA_CHATGPT.zip` detectados (Raíz y Laboratorio).

## 2. Carpetas Temporales Detectadas
- `temp_zip_staging`
- `_staging_final`
- `_zip_clean_1032631789`

## 3. Polución de Rutas (Hardcoded)
- **Patrón Objetivo:** `C:\Users\alera\Desktop\Bot\Bot V2`
- **Ubicación:** `BOT_V2_DAYTIME_LAB\src\`
- **Estimación de Reemplazos:** 150+ instancias en archivos Python.

## 4. Estado de Seguridad
- No se han modificado estrategias.
- No se han corrido backtests.
- La evidencia histórica está íntegra y lista para ser movida al archivo.

## 5. Veredicto Pre-Limpieza
**READY_FOR_SANitization.** El sistema está identificado y los riesgos de movimiento están controlados.
