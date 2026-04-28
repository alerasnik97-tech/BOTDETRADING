
# STRATEGY AUTHORITY MAP

| Estrategia | Estado | Fase | Fuente de Verdad |
| :--- | :--- | :--- | :--- |
| **Phase 22 High WR** | **CANDIDATO FORWARD** | 23 | `BOT_V2_DAYTIME_LAB\configs\phase22_forward_demo_config.json` |
| **Phase 18 Baseline** | **CERTIFIED BASELINE** | 18 | `BOT_V2_DAYTIME_LAB\src\phase18_*.py` |
| **Phase 20 Balanced** | **ARCHIVED BENCHMARK** | 21 | `ARCHIVE_SUPERSEDED` |
| **Phase 19** | **INVALIDATED** | 19 | `ARCHIVE_SUPERSEDED` |

## Jerarquía Operativa
1. **News Fortress**: Autoridad de Bloqueo (Si News dice BLOCK, no se opera).
2. **Data Quality Mask**: Autoridad de Integridad (Si Mask dice BLOCK, no se opera).
3. **Phase 22 Logic**: Autoridad de Ejecución.

## Restricciones
- No se permiten modificaciones de parámetros fuera de la Phase 23 Repaired.
- No se permite el uso de Phase 19 como base para nuevas ideas.
- El despliegue es **LOCAL DEMO ONLY**.
