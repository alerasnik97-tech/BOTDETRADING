# R1 V49.7B CONTROLLED ?" MEMORY PLAN

**Diagnstico previo**: El crash de V49.7B se debi a una acumulacin de cachǸ de ventanas de ticks y objetos de trades en RAM sin liberacin.

## Estrategia de Mitigacin
1. **Batching por Mes**: Cargar parquets de un solo mes, procesar y liberar.
2. **Batching por Configs**: Procesar grupos de 100 configuraciones a la vez.
3. **Escritura Incremental**: No acumular listas gigantes de trades en memoria; realizar flush a disco tras cada mes/batch.
4. **Garbage Collection Explicito**: Llamar a `gc.collect()` tras el procesamiento de cada mes denso (ej. 2020-03).
5. **No Caching de Ventanas**: En lugar de guardar miles de slices de DataFrames, realizar el slice en tiempo real (just-in-time) durante la ejecucin del motor.
6. **Limitacin de RAM**: Supervisar el uso de memoria por log.
