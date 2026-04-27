# GITHUB / VPS SYNC PLAN

Estrategia de sincronización para mantener la VPS actualizada de forma segura.

## Flujo de Trabajo (Local)
1. Desarrollar/Corregir en la rama `chore/github-clean-sync`.
2. Verificar que no haya secretos (`git status`).
3. Realizar `git push origin chore/github-clean-sync`.

## Flujo de Trabajo (VPS)
1. Realizar `git pull` desde la rama `chore/github-clean-sync`.
2. Ejecutar `vps_preflight_check.ps1` para validar que los cambios no rompieron el entorno.
3. Actualizar `requirements.txt` si hubo cambios.
4. Reiniciar el servicio de monitoreo (Forward Demo).

## Reglas de Oro
- **NUNCA** hacer merge de `main` a la VPS sin auditar.
- **NUNCA** usar la VPS para resolver conflictos de Git. Los conflictos se resuelven en local y se suben limpios.
- Si detectas que Git intenta descargar archivos `.csv` pesados, aborta la sincronización.
