# ORACLE_DEPLOYMENT_PLAN

1. **Fase de Preparación (Local)**:
   - Selección de la estrategia y parámetros.
   - Creación del paquete cloud en carpeta externa.
   - Generación de hashes de verificación.

2. **Fase de Transferencia**:
   - Uso de `scp` o `rsync` para subir el paquete.
   - Verificación de hashes en destino.

3. **Fase de Ejecución**:
   - Lanzamiento en `tmux`.
   - Monitoreo remoto vía logs (opcionalmente enviando pings de estado a un webhook seguro si se desea, pero sin secretos).

4. **Fase de Cierre**:
   - Parada controlada del runner.
   - Compresión de resultados.
   - Descarga y verificación local.
   - Destrucción o limpieza de la instancia si no se usará en breve.
