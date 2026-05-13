# DOCUMENTACIÓN DEL BASTIÓN DE INTEGRIDAD (PRE-COMMIT HOOK)

## 1. Propósito
Para evitar la alteración accidental o subrepticia del código fuente del motor de backtesting durante las fases de alta agilidad o re-estructuración de repositorios, se ha implementado un bastión técnico a nivel del sistema de control de versiones Git: el hook local `pre-commit`.

## 2. Mecánica de Bloqueo
Al ejecutar `git commit`, el hook evalúa los archivos en el área de preparación (staging area). Si se detecta que alguno de los archivos modificados, añadidos o eliminados pertenece a las rutas protegidas:
- `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/src/v7_engine/`
- `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/src/v6_utils/`

El hook procede a verificar la existencia física de la **Licencia de Excepción Institucional**:
`06_GOVERNANCE_AND_COMPLIANCE/engine_lockdown/APPROVED_ENGINE_CORE_CHANGE_REQUEST.md`

En caso de que dicha licencia esté ausente, el commit es abortado de forma inmediata y automática con el mensaje:
> **"ENGINE CORE IS LOCKED. Create approved change request before modifying v7_engine or v6_utils."**

## 3. Limitaciones y Advertencias de Seguridad
- **Naturaleza Local**: El hook reside exclusivamente en el directorio `.git/hooks/` de la máquina local. Por su naturaleza en Git, no se propaga al hacer clonado. Sirve como una primera línea de defensa técnica para el desarrollador/agente local y **no sustituye la revisión obligatoria de código (Pull Requests) ni la auditoría humana en el servidor**.
- **Seguridad de Secretos**: El hook incluye un chequeo preventivo para no permitir confirmaciones si se detecta un archivo con extensiones o nombres sospechosos de contener credenciales (ej. `.env`, `id_rsa`).
