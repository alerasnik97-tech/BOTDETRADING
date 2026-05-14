# POST-ZIP GITHUB SOURCE OF TRUTH POLICY

## 1. Contexto
A partir de mayo de 2026, el proyecto migra de un flujo de trabajo basado en archivos ZIP a un modelo de "GitHub Source of Truth". Este cambio elimina la latencia de entrega, previene la desincronización de versiones y permite una auditoría directa y profesional por parte de modelos de IA avanzados (ChatGPT 5.5+).

## 2. GitHub como Única Fuente de Verdad
- El repositorio oficial en GitHub (`alerasnik97-tech/bottrading`) es la fuente definitiva del estado del proyecto.
- La rama operativa para revisión y handoff es `clean-sync-branch`.
- Cualquier validación o auditoría debe realizarse sobre el código y reportes presentes en dicha rama.

## 3. Depreciación del Archivo ZIP
- El archivo `000_PARA_CHATGPT.zip` y similares quedan depreciados para el flujo de trabajo diario.
- Ya no se generarán paquetes ZIP para la comunicación con la IA.
- El uso de ZIPs queda restringido exclusivamente a archivos históricos externos fuera del entorno de desarrollo.

## 4. Flujo de Trabajo
Cada fase o tarea debe seguir el ciclo:
1. Ejecución y generación de resultados en carpetas institucionales.
2. Auditoría física y validación local.
3. Commit local en `clean-sync-branch`.
4. Push al repositorio remoto.
5. Reporte de handoff referenciando el hash del commit.
