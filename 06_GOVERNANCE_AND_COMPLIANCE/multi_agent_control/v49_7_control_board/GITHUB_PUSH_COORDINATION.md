# GITHUB PUSH COORDINATION — V49.7

## Protocolo de Sincronización
Para evitar conflictos de Git en un entorno multi-agente, se establece el siguiente protocolo:

1. **Branch Única:** Solo se opera en `clean-sync-branch`.
2. **Prohibición de Main:** El push directo a `main` está desactivado para todos los agentes automáticos.
3. **Frecuencia de Pull:** Antes de cualquier commit, el agente DEBE realizar un `git pull origin clean-sync-branch` para integrar cambios de otros agentes.
4. **Scope Limitado:** Los agentes solo deben añadir (`git add`) archivos dentro de su carpeta permitida.
5. **Mensajes de Commit:** Deben incluir el prefijo del agente (ej. `[research]`, `[governance]`, `[cloud]`).

## Estado de Sincronización Actual
- **Branch:** `clean-sync-branch`
- **Último Push Coordinado:** 2026-05-14 09:06 (Governance)
- **Próximo Agente en Cola:** Research Agent (A1)

## Resolución de Conflictos
Si un agente detecta un conflicto de merge:
1. **NO intentar resolución automática.**
2. **Abortar el push.**
3. **Reportar:** `MULTI_AGENT_CONFLICT_RISK` en el Control Board.
4. **Esperar intervención manual o del Governance Agent.**
