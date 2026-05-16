# PROMPT: APLICAR HIGIENE DE RAMAS GITHUB (POST-APROBACIÓN)

Actuá como Codex GPT-5.5 Max en modo GitHub safety-first cleanup specialist.

OBJETIVO: ejecutar la limpieza de ramas aprobada en `BRANCH_HYGIENE_AUDIT_REPORT.md`.

REGLAS:
- NO tocar main.
- NO force push.
- NO borrar ramas con PR activo (salvo el #4 superseded).
- NO borrar ramas sin tag previo si son de archivo.

ACCIONES:

1. CREAR TAGS DE ARCHIVO:
git tag archive/clean-sync-20260516 clean-sync-branch
git push origin archive/clean-sync-20260516

2. BORRAR RAMAS SUPERSEDED (REMOTO):
git push origin --delete governance/engine-base-preflight-fix-20260516
git push origin --delete research/f06-evidence-rebuild-foundation-20260515
git push origin --delete research/push-test-20260515

3. BORRAR RAMAS SUPERSEDED (LOCAL):
git branch -D governance/engine-base-preflight-fix-20260516
git branch -D research/f06-evidence-rebuild-foundation-20260515
git branch -D research/push-test-20260515

4. CERRAR PR SUPERSEDED:
# Solo si gh está disponible, si no, informar al usuario.
gh pr close 4 --comment "Superseded by PR #5 (V2)"

5. VERIFICAR:
git fetch origin --prune
git branch -a
gh pr list

6. REPORTE FINAL:
Confirmar ramas borradas y estado del repositorio.
