# ZIP DEPRECATION POLICY

## 1. Estado de Depreciación
El uso de archivos ZIP como mecanismo de transferencia de información entre el usuario y la IA (Antigravity/ChatGPT) queda oficialmente finalizado.

## 2. Acciones Inmediatas
- Eliminación de `000_PARA_CHATGPT.zip` de la raíz del proyecto.
- **NO existe** un archivo ZIP-workflow. El antiguo directorio externo `BOT_ZIP_LEGACY_ARCHIVE` **ya fue eliminado y NO debe reaparecer** (ni en raíz, ni en proyecto activo, ni recreado).
- `.gitignore` bloquea cualquier intento accidental de trackear archivos ZIP (`*.zip`); esta regla NO debe debilitarse.

## 3. Excepciones
- **No hay** ZIP workflow operativo. No se generan ZIPs para handoff con la IA (GitHub es la fuente de verdad).
- ZIPs históricos solo pueden residir, **local-only y gitignored**, en `07_BACKUPS/` o en cuarentena, y **únicamente si el owner lo autoriza** explícitamente. Nunca en la raíz ni en el proyecto activo.
