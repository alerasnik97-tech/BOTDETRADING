# PRE-COMMIT SECRET SCAN PLAN

**Estado**: **PROPOSED**

## Objetivo
Implementar un hook de pre-commit que escanee los archivos en staging (`staged files`) antes de permitir el commit local.

## Patrones a Bloquear
- **Telegram Bot Token**: `[0-9]{8,12}:[a-zA-Z0-9_-]{35}`
- **Generic API Key**: `(?i)api_key.*=.*['"][a-zA-Z0-9]{20,}['"]`
- **GitHub Token**: `ghp_[a-zA-Z0-9]{36}`
- **Kaggle JSON**: `{"username":".*","key":".*"}`

## Implementacin Sugerida
Se propone utilizar un script de Python simple `.git/hooks/pre-commit` o configurar la herramienta `pre-commit` con `detect-secrets`.

**Accin Inmediata**: Mientras se configura el hook automático, el agente Antigravity realizarǭ un escaneo manual de cada archivo agregado al staging mediante `grep_search`.

**Veredicto**: Saneamiento manual activo; automatizacin en planificacin.
