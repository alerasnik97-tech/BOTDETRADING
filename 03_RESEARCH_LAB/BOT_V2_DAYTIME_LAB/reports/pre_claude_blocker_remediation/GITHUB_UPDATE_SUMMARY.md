# GITHUB UPDATE SUMMARY

**Fecha:** 2026-05-15
**Branch:** `research/pre-claude-blocker-remediation-20260515`
**Commit msg:** `governance: freeze V50B evidence after extreme audit blockers`

---

## 1. Qué incluye este audit pack (allowlist)

- `PRE_CLAUDE_FREEZE_NOTICE.md`
- `FORENSIC_VERIFICATION_REPORT.md`
- `SUPERSEDED_CERTIFICATIONS.md`
- `FULL_EVIDENCE_REBUILD_PLAN.md`
- `GITHUB_UPDATE_SUMMARY.md`
- Addendums livianos (banner SUPERSEDED) en `FULL_RERUN_TRAIN_ONLY_REPORT.md`
  y `COST_HARDENING_REPORT.md` (contenido original intacto).

## 2. Qué NO incluye (deliberadamente excluido)

- Raw data, tick data, parquet, `V50B_RERUN_TRADES.csv`, ZIPs, lock files,
  `.venv`, caches, `__pycache__`, credenciales.
- Ningún CSV pesado de evidencia inválida.

## 3. Mensaje institucional del PR

- Claude audit detectó **BLOCKED_MAJOR_RISK**.
- NO V50C. NO validation. NO holdout. NO 2025/2026.
- F06 / F08 / F12 **no certificadas**.
- Certificación previa **superseded** (no borrada).
- Plan de reconstrucción de evidencia agregado.
- Sin raw data. Sin trades pesados. Sin ZIP.
- GitHub es la entrega principal.

## 4. Base del branch (decisión documentada)

`PRE_CLAUDE_FREEZE_NOTICE.md` y este pack se construyen sobre la rama
publicada `research/v50b-cost-hardening-clean-20260515` (= `origin/main` +
1 commit ya pusheado), **no** sobre `main` local.

**Razón:** la verificación forense detectó que `main` local está **42
commits adelante de `origin/main`** (contenido no revisado/no pusheado) y
que los reportes addendados sólo existen en la rama clean. Basar el pack en
la rama publicada: (a) mantiene el push mínimo y seguro, (b) evita publicar
42 commits locales no revisados, (c) preserva los reportes addendados, (d)
produce un PR limpio. Decisión alineada con la prioridad #1 (seguridad).

## 5. Estado de herramientas

- `gh` CLI: **NO instalado** → el PR no puede crearse por CLI.
  Acción: crear el PR manualmente en
  `https://github.com/alerasnik97-tech/bottrading` → "Compare & pull request"
  para `research/pre-claude-blocker-remediation-20260515` contra la base
  deseada. Título/cuerpo en §3.

## 6. Limitación de higiene pendiente (no resuelta por este pack)

El repositorio remoto **ya** contiene data pesada/sensible en historia
(raw tick ~298 MB, M3 2020-2026, ZIP 226 MB, `.bundle.lock` 396 MB) y 5
árboles `BOT_V2_DAYTIME_LAB` duplicados. Este audit pack **no** introduce
nada de eso, pero **tampoco lo limpia**. La limpieza de historia / air-gap
del holdout es trabajo aparte, listado en `FULL_EVIDENCE_REBUILD_PLAN.md`
y requiere aprobación explícita (implica history rewrite).
