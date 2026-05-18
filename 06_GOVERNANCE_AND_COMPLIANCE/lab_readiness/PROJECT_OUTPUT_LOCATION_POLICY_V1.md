# PROJECT OUTPUT LOCATION POLICY V1
**Date:** 2026-05-18
**Project:** Systematic Infrastructure Professionalization — Repository Hygiene & Location Policy
**Security Status:** ACTIVE REPOSITORY RULES — CONFORMANCE IS MANDATORY

---

## 1. Rule

Todos los entregables, reportes, análisis, esquemas o productos serios generados por cualquier agente de Inteligencia Artificial (Antigravity, Claude, Cursor, u otros) en este proyecto deben residir obligatoriamente dentro del directorio canónico del repositorio:

```
C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo
```

No se permite que archivos de investigación o documentación estratégica permanezcan de forma permanente o exclusiva en ubicaciones externas al proyecto.

---

## 2. Prohibited Canonical Locations

Queda terminantemente prohibido utilizar como fuentes de verdad o almacenamiento canónico las siguientes rutas:
*   Carpetas temporales del Escritorio (Desktop) creadas por agentes (p. ej. `AGENTE_*`).
*   Directorio de Descargas (Downloads).
*   Carpetas temporales del sistema (`%TEMP%`, `/tmp`).
*   Cualquier directorio local o en red externo al repositorio sin control de versiones.

---

## 3. Allowed Temporary Exception

Se tolera la generación de archivos temporales de salida en el Escritorio únicamente si un proceso de seguridad (Safety Gate) o limitaciones de contexto del agente lo requieren de forma imprescindible. En tales circunstancias:
1.  **Reingesta Inmediata:** Los archivos deben ser copiados y reingresados en la carpeta correspondiente del proyecto en la misma sesión de trabajo.
2.  **Manifiesto de Integridad:** Se debe generar un archivo de manifiesto CSV con los nombres de archivo, tamaños y hashes SHA-256 de origen y destino.
3.  **No Canonicidad:** Los archivos externos no deben considerarse bajo ninguna circunstancia como la fuente oficial de verdad y deberán ser ignorados por Git.

---

## 4. Canonical Locations

Los directorios internos correctos para almacenar información del proyecto son:

*   **Investigación y Catalogación de Estrategias:**
    `03_RESEARCH_LAB/strategy_research_intake/`
*   **Conocimiento Quant General y Profesionalización:**
    `03_RESEARCH_LAB/knowledge_intake/`
*   **Gobernanza y Cumplimiento de Procesos:**
    `06_GOVERNANCE_AND_COMPLIANCE/`
*   **Salidas Locales de Ejecución (No Versionadas):**
    `03_RESEARCH_LAB/research_lab/local_outputs_do_not_commit/`
*   **Código de Producción (Core Protegido):**
    `01_CORE_PRODUCTION/` *(Solo tras certificación formal del owner)*

---

## 5. Commit Policy

Para mantener un repositorio ligero, seguro y profesional, se establecen las siguientes reglas de control de versiones:
*   **Versionables:** Se permite versionar archivos livianos de gobernanza, auditoría, manifiestos, configuraciones e índices en formato Markdown (`.md`), CSV (`.csv`), JSON (`.json`) y archivos `.gitignore` locales.
*   **No Versionables:**
    *   Prohibido commitear binarios pesados (ZIPs, bases de datos, imágenes pesadas, entornos virtuales).
    *   Prohibido commitear outputs locales de corridas (archivos en `local_outputs_do_not_commit/`).
    *   Prohibido commitear raw data o datasets de mercado (ticks, archivos parquet pesados, carpetas en `05_MARKET_DATA_VAULT/`).
*   **Comandos Prohibidos:** Se prohíbe terminantemente el uso de `git add .` o `git add -A`. El staging de archivos debe realizarse archivo por archivo de forma explícita y auditable.

---

## 6. Enforcement

Cada instrucción de prompts futuros dictada al equipo de desarrollo y a agentes de IA debe especificar explícitamente rutas de destino internas y conformes a esta política, garantizando la total agilidad y seguridad operacional.
