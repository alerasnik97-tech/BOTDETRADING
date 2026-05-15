# V50B RERUN SINGLE-WRITER POLICY

Para garantizar la integridad fscia de los resultados, se establecen las siguientes reglas de operación:

1. **Lock Obligatorio**: Todo proceso de ejecución DEBE adquirir el archivo `locks/V50B_RERUN.lock`. Si el archivo existe, el runner ABORTARÁ de inmediato.
2. **Escritura Append-Only**: Queda prohibido cargar el archivo CSV de trades/señales en memoria para reescribirlo. La persistencia se realizarǭ exclusivamente mediante `append` a disco.
3. **Run ID y Trazabilidad**: Cada ejecución generarǭ un `run_id` (UUID corto). Cada trade y rechazo registrado incluirǭ el `run_id` y el `writer_pid` para auditora forense.
4. **Validacin de Flush**: El runner realizarǭ flujos incrementales (cada 25 configs) para asegurar que la evidencia se guarde progresivamente.
5. **Cierre Atmico**: Los reportes finales de métricas se generarǭ en archivos `.tmp` y se renombrarǭ solo tras validar que el proceso termin satisfactoriamente.

**Incumplimiento**: Cualquier violación de esta poltica invalidarǭ automǭticamente los resultados de la corrida.
