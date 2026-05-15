# KAGGLE_SMOKE_OBJECTIVE

El objetivo de esta fase es validar la infraestructura de Kaggle antes de proceder a corridas reales o nocturnas.

## Objetivos Técnicos:
- Verificar el clonado exitoso desde GitHub (clean-sync-branch).
- Validar el entorno de ejecución Python en Kaggle.
- Comprobar que los imports críticos de `src/v7_engine` y `src/v6_utils` funcionan correctamente.
- Validar la visibilidad de la estructura de carpetas institucional.
- Probar la generación y persistencia de outputs livianos en `/kaggle/working/`.
- Realizar un escaneo de seguridad preventivo para asegurar que no hay filtración de secretos.
