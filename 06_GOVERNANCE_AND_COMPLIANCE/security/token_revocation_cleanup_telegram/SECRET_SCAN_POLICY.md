# SECRET SCAN POLICY

**Objetivo**: Prevenir la filtracin de credenciales y secretos en el repositorio institucional.

## Reglas Obligatorias
1. **No Hardcoding**: Estrictamente prohibido guardar tokens, API keys, contraseİas o secretos en archivos de texto (`.md`, `.py`, `.json`, `.csv`, `.txt`, `.bak`).
2. **Environment Variables**: Todo secreto debe ser inyectado mediante variables de entorno locales o gestores de secretos (Secret Manager).
3. **Redaccin en Reportes**: Cualquier mencin a un secreto en reportes de auditora debe usar un `masked_preview` (ej: `12345:ABC...`) y nunca el valor completo.
4. **Gitignore**: El archivo `.gitignore` debe bloquear `.env`, `secrets/`, `*.pem`, `*.key` y cualquier archivo de configuracin local sensible.
5. **Pre-Commit Check**: Todo commit debe ser revisado manualmente o mediante herramientas de escaneo en busca de patrones de secretos antes de ser pusheado.

## Proceso ante Exposicin
1. **Revocacin Inmediata**: Revocar la credencial en el proveedor (ej: BotFather, GitHub, AWS).
2. **Notificacin**: Documentar la exposicin y la revocacin en el log de gobernanza.
3. **Mascara**: Reemplazar el secreto expuesto por un marcador de redaccin en el ǭrbol actual.
4. **Purge**: Evaluar la necesidad de reescribir el historial de Git (BFG Repo-Cleaner o `git-filter-repo`).
