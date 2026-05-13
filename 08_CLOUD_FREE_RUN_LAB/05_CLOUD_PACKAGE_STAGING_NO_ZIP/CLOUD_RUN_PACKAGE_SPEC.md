# CLOUD_RUN_PACKAGE_SPEC

Un paquete para ejecución en la nube debe incluir estrictamente:

- **Runner congelado**: El script principal (`runner.py` o similar) y sus módulos necesarios.
- **Config JSON**: Archivo de configuración específico para la corrida.
- **Requirements mínimos**: `requirements.txt` filtrado para solo lo necesario en el runner.
- **Versión Python**: Especificación de la versión compatible.
- **Scripts de resume/checkpoint**: Herramientas para recuperar el estado.
- **Dataset reducido**: Partición de datos (CSV o Parquet) autorizada para el período de prueba.
- **Manifests**: Firma digital y descripción del contenido.
- **Hash del código**: SHA256 de los archivos de lógica para asegurar integridad.
- **README de ejecución**: Instrucciones rápidas para el operador de la nube.
- **Output folder**: Carpeta vacía preparada para recibir resultados.

**PROHIBIDO INCLUIR**:
- Raw data completo.
- Parquet masivo sin necesidad.
- Entornos virtuales (`venv`).
- Carpeta `.git`.
- Cachés (`__pycache__`).
- Backups.
- Secretos y credenciales.
- ZIP oficial del proyecto.
