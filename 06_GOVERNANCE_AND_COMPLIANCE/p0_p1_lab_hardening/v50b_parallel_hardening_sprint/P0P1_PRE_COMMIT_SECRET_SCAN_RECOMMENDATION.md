# P0/P1 Pre-commit Secret Scan Recommendation
Fecha: 2026-05-14

## Propuesta Técnica
Implementar un control preventivo para evitar la subida accidental de secretos al repositorio GitHub, protegiendo la integridad de la rama `clean-sync-branch`.

## Componentes Recomendados
1. **Filtro de Archivos Staged**: Escaneo automático de archivos en el index de Git antes del commit.
2. **Bloqueo por Patrón**:
    - Denegar cualquier archivo con extensión `.env`.
    - Denegar tokens de Telegram (`[0-9]{8,}:[a-zA-Z0-9_-]{30,}`).
    - Denegar strings que coincidan con `GH_TOKEN`, `API_KEY` o `KAGGLE_KEY`.
3. **Escaneo de Extensiones Sensibles**: Bloquear `.bak`, `.old`, `.tmp` si contienen patrones sospechosos.

## Plan de Implementación (Post-V50B)
- Crear un script `04_INFRASTRUCTURE_ENGINEERING/security/pre_commit_scanner.py`.
- Integrar como un Git Hook local (opcional) o como un paso obligatorio en el manual de "Audit Readiness".
- **IMPORTANTE**: No implementar hooks automáticos durante la corrida V50B para evitar bloqueos accidentales de los runners activos.

## Conclusión
La seguridad reactiva (scans periódicos) debe evolucionar hacia una seguridad proactiva (pre-commit) para cumplir con el estándar institucional V7.
