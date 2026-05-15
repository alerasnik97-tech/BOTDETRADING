# V50B RERUN SINGLE-WRITER REQUIREMENTS

Para evitar la repeticin de la Race Condition, la infraestructura de ejecución debe actualizarse con las siguientes garantas:

1. **Un solo Runner activo**: Implementar un `run.lock` en la raz del proyecto. Si existe, ningun otro runner puede iniciar.
2. **Escritura Atmica**: El runner debe escribir en un archivo temporal (`.tmp`) y realizar un rename/move al archivo final solo cuando la fase esté garantizada.
3. **Checkpoints Append-Only**: Cambiar la lgica de "Cargar todo el CSV -> Añadir -> Sobrescribir" por una lgica de `append` directo a disco (sin cargar el estado previo en memoria).
4. **Deteccin de Colisin**: El runner debe registrar su `PID` en los metadatos de cada bloque de trades.
5. **Aislamiento de Output**: Si se requiere paralelismo, cada proceso debe escribir en su propio archivo (ej: `trades_PID.csv`) y un post-procesador debe unificarlos al final.

**Objetivo**: Cero colisiones en IO.
