# KAGGLE_V49_7C_FAILURE_RECOVERY

## Escenarios de Fallo

### 1. Sesión de Kaggle expirada
- **Síntoma**: Desconexión y pérdida de la consola.
- **Acción**: Reiniciar notebook, cargar checkpoint y continuar.

### 2. Error de Memoria (OOM)
- **Síntoma**: Crash del kernel de Python.
- **Acción**: Reducir el número de procesos en paralelo (si se usan) o limpiar caché entre configuraciones.

### 3. CSV Truncado o Corrupto
- **Síntoma**: El archivo de resultados no se puede leer o tiene filas incompletas.
- **Acción**: Borrar la última entrada del CSV de resultados y reiniciar desde el último checkpoint válido.

### 4. Duplicados detectados
- **Síntoma**: Mismas configuraciones aparecen varias veces.
- **Acción**: Ejecutar script de `duplicate_audit.py` (si existe) y limpiar el dataset de resultados antes de la integración final.

### 5. Bloqueo de Git
- **Síntoma**: Error al hacer push desde Kaggle.
- **Acción**: Usar la descarga manual (Handoff Opción B).
