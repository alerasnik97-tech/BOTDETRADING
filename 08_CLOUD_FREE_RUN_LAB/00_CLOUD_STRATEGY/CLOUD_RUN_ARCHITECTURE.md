# CLOUD_RUN_ARCHITECTURE

## Flujo de Trabajo
1. **Local**: Desarrollo, tests unitarios y generación de `CLOUD_PACKAGE`.
2. **Transferencia**: Subida del paquete (runner + config + dataset reducido) a la nube.
3. **Ejecución**: Lanzamiento del proceso en la nube con `checkpoint` activado.
4. **Monitoreo**: Verificación de logs y estados parciales.
5. **Descarga**: Bajada de outputs a `10_CLOUD_OUTPUT_INBOX`.
6. **Auditoría**: Validación local de resultados cloud antes de integrarlos al `03_RESEARCH_LAB`.

## Componentes Críticos
- **Causal Runner**: El mismo motor que corre localmente.
- **Checkpoint Logic**: Guardado de estado cada N iteraciones o períodos.
- **Fail-Safe**: Detención automática ante errores o límites de recursos.
- **Manifest**: Archivo que describe qué se está corriendo y bajo qué condiciones.
