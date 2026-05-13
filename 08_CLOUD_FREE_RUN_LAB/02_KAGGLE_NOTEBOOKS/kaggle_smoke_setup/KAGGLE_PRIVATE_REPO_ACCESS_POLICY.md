# KAGGLE_PRIVATE_REPO_ACCESS_POLICY

- **No escribir usuario/contraseña en notebook**: Queda terminantemente prohibido hardcodear credenciales en las celdas.
- **No pegar GitHub token en celda**: Los tokens no deben ser visibles en el código del notebook.
- **Uso de Kaggle Secrets**: Si el repo es privado, se debe usar la funcionalidad de "Secrets" de Kaggle.
- **Configuración**:
  - Crear un GitHub Personal Access Token (PAT) con permisos mínimos (read-only).
  - Guardarlo en Kaggle como un secret llamado `GH_TOKEN`.
- **Seguridad**:
  - Nunca imprimir el token (`print(token)`).
  - Nunca guardar el token en archivos persistentes en la instancia.
  - El archivo `~/.netrc` debe ser borrado al finalizar el uso para evitar que persista en el entorno.
- **Alternativa preferida**:
  - Para máxima seguridad, se recomienda subir el paquete de código y datos reducidos como un **Kaggle Dataset privado**. Esto evita la necesidad de conexión a Git durante la ejecución.

**Recomendación oficial del proyecto**:
A. Usar paquete Kaggle dataset privado para código/data reducida (Más seguro).
B. Usar GitHub clone con token solo para código si es necesario (Opción secundaria).
