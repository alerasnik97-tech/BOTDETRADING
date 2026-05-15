# V50B SINGLE-WRITER DESIGN SPEC

## Arquitectura de Seguridad IO

### 1. Sistema de Lock (Mutual Exclusion)
- **Archivo**: `V50B_RUNNER.lock` en la raz de la fase.
- **Contenido**: JSON con `{ "pid": int, "start_time": iso8601, "run_id": string }`.
- **Regla**: Si el archivo existe, el runner debe abortar de inmediato con error `LOCKED_BY_ANOTHER_PROCESS`.

### 2. Escritura Append-Only (Integridad)
- **MǸtodo**: Uso de `mode='a'` en Python (`to_csv(..., mode='a', header=not exists)`).
- **Ventaja**: No se carga el CSV en memoria. Se evitan regresiones de conteo por carga de estado viejo.

### 3. Run Isolation
- **Run ID**: Cada ejecución genera un UUID corto.
- **Traceability**: Cada fila escrita debe incluir el `run_id` para auditora post-procesamiento.

### 4. Atomic Write (Reportes Finales)
- **Proceso**: Escribir reporte en `.tmp`, verificar integridad, y realizar `os.replace()` al nombre final.

**Objetivo**: Garantizar que 1 trade escrito = 1 trade persistido.
