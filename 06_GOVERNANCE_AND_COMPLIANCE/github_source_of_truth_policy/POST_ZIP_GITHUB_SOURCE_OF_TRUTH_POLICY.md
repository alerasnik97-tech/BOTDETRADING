# POST-ZIP GITHUB SOURCE OF TRUTH POLICY

## 1. Contexto
A partir de mayo de 2026, el proyecto migra de un flujo de trabajo basado en archivos ZIP a un modelo de "GitHub Source of Truth". Este cambio elimina la latencia de entrega, previene la desincronización de versiones y permite una auditoría directa y profesional por parte de modelos de IA avanzados (ChatGPT 5.5+).

## 2. GitHub como Única Fuente de Verdad
- El repositorio oficial en GitHub (`alerasnik97-tech/bottrading`) es la fuente definitiva del estado del proyecto.
- La rama canónica vigente para root-hygiene / governance pre-laboratorio es **`governance/root-hygiene-20260516`** (o la rama canónica declarada en el gate vigente). NO se debe hardcodear ninguna rama como operativa universal.
- `clean-sync-branch` queda **reclasificada como historical donor branch / unrelated orphan history**, SIN ancestro común con la línea canónica y **ya NO es la fuente de verdad operativa**.
- Cualquier integración desde `clean-sync-branch` se hace SOLO por: curated file-level port; cherry-pick/patch bajo owner approval. **Nunca** direct merge; **nunca** `--allow-unrelated-histories`; **nunca** force push; **nunca** tocar `main` sin autorización.
- Cualquier validación o auditoría debe realizarse sobre el código y reportes presentes en la rama canónica vigente.

## 3. Depreciación del Archivo ZIP
- El archivo `000_PARA_CHATGPT.zip` y similares quedan depreciados para el flujo de trabajo diario.
- Ya no se generarán paquetes ZIP para la comunicación con la IA.
- El uso de ZIPs queda restringido exclusivamente a archivos históricos externos fuera del entorno de desarrollo.

## 4. Flujo de Trabajo
Cada fase o tarea debe seguir el ciclo:
1. Ejecución y generación de resultados en carpetas institucionales.
2. Auditoría física y validación local.
3. Commit local en la **rama canónica vigente** (actualmente `governance/root-hygiene-20260516`), o en una rama derivada de ella para trabajo curado. NUNCA commitear directamente en `clean-sync-branch` como flujo operativo.
4. Push non-force al repositorio remoto (sin tocar `main`).
5. Reporte de handoff referenciando: repo, rama actual, hash del commit, push status, no main, no force, no ZIP, tests, blockers.
