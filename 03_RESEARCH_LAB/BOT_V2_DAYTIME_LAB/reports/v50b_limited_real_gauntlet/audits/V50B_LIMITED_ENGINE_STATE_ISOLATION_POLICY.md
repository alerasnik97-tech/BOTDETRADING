# V50B LIMITED ENGINE STATE ISOLATION POLICY

**Objetivo**: Prevenir la contaminación de resultados entre ejecuciones.

## Reglas de Aislamiento
1. **Instancia Única**: El `v50b_limited_real_runner.py` debe instanciar un nuevo `UnifiedV7Engine` al inicio de cada bucle de configuración.
2. **Reset de Throttler**: Al usar un motor nuevo, el `PositionThrottler` interno inicia en 0 trades, asegurando que el límite de 3 trades/día se aplique solo a esa configuración específica.
3. **No Persistencia**: Prohibido guardar el estado del motor en archivos temporales o bases de datos compartidas durante la ejecución del Gauntlet.
4. **ID de Trazabilidad**: Cada instancia del motor tendrá un `engine_instance_id` único (UUID o timestamp) que se registrará en cada trade y cada rechazo.

## Verificacin de Throttler
Si el log de rechazos muestra `BLOCKED_BY_THROTTLER` en el primer trade del día de una configuración, se considerará **CONTAMINACIÓN DE ESTADO** e invalidará el Gauntlet.

**Veredicto**: Poltica activa.
