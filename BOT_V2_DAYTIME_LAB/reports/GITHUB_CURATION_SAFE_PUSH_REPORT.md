# GITHUB CURATION SAFE PUSH REPORT

- timestamp_utc: 2026-04-30T11:31:17.391912+00:00
- verdict: GITHUB_SAFE_PUSH_COMPLETE_WITH_WARNINGS
- branch: main
- remote: https://github.com/alerasnik97-tech/bottrading.git

## 1. Lo mas importante
Se sube solo la superficie liviana y util: docs principales, MANIPULANTE/ROCKI livianos, Phase41/42 relevante ya versionado o reportado, auditoria de curaduria y .gitignore endurecido. `000_PARA_CHATGPT.zip` se conserva local pero se elimina del tracking de Git.

## 2. Archivos incluidos
### documentacion
- `00_READ_THIS_FIRST.md`
- `01_CURRENT_PROJECT_STATUS.json`
- `01_CURRENT_PROJECT_STATUS.md`
- `02_STRATEGY_AUTHORITY_MAP.json`
- `02_STRATEGY_AUTHORITY_MAP.md`
- `BOT_V2_DAYTIME_LAB/outputs/github_curation_safe_push/git_add_plan.json`
- `BOT_V2_DAYTIME_LAB/outputs/github_curation_safe_push/git_add_plan.md`
- `BOT_V2_DAYTIME_LAB/outputs/github_curation_safe_push/github_curation_preflight.json`
- `BOT_V2_DAYTIME_LAB/outputs/github_curation_safe_push/github_curation_preflight.md`
- `BOT_V2_DAYTIME_LAB/outputs/github_curation_safe_push/github_exclude_manifest.txt`
- `BOT_V2_DAYTIME_LAB/outputs/github_curation_safe_push/github_include_manifest.txt`
- `BOT_V2_DAYTIME_LAB/outputs/github_curation_safe_push/inventory_summary.json`
- `BOT_V2_DAYTIME_LAB/outputs/github_curation_safe_push/manifest_summary.json`
- `BOT_V2_DAYTIME_LAB/outputs/github_curation_safe_push/pre_commit_review.json`
- `BOT_V2_DAYTIME_LAB/outputs/github_curation_safe_push/pre_commit_review.md`
- `BOT_V2_DAYTIME_LAB/outputs/github_curation_safe_push/secret_scan_report.json`
- `BOT_V2_DAYTIME_LAB/outputs/github_curation_safe_push/secret_scan_report.md`
- `CHANGELOG.md`
- `CLOUD_WORKFLOW.md`
- `README.md`
### MANIPULANTE
- `MANIPULANTE/00_LEER_PRIMERO/MANIPULANTE_REPLAY_FORWARD_RESUMEN.md`
- `MANIPULANTE/14_ANALISIS/MANIPULANTE_HYBRID_REPLAY_FORWARD_AUDIT_REPORT.md`
- `MANIPULANTE/15_FORWARD_DEMO_SCORECARD/promotion_gate/MANIPULANTE_PROMOTION_GATE_TO_REAL.md`
- `MANIPULANTE/15_FORWARD_DEMO_SCORECARD/stress_tests/phase42_operational_stress_tests.csv`
- `MANIPULANTE/15_FORWARD_DEMO_SCORECARD/stress_tests/phase42b_operational_stress_tests_16.csv`
### ROCKI_AM
- `ROCKI_AM/01_ORIGEN_HISTORICO/ROCKI_AM_SOURCE_MANIFEST.csv`
- `ROCKI_AM/03_METRICAS/ROCKI_AM_METRICS_SUMMARY.csv`
### codigo_fuente
- ninguno en este commit
### reportes
- `BOT_V2_DAYTIME_LAB/reports/GITHUB_CURATION_SAFE_PUSH_REPORT.json`
- `BOT_V2_DAYTIME_LAB/reports/GITHUB_CURATION_SAFE_PUSH_REPORT.md`
- `BOT_V2_DAYTIME_LAB/reports/PHASE41_MANIPULANTE_HYBRID_REPLAY_FORWARD_AUDIT_REPORT.json`
- `BOT_V2_DAYTIME_LAB/reports/PHASE41_MANIPULANTE_HYBRID_REPLAY_FORWARD_AUDIT_REPORT.md`
### scorecards
- `BOT_V2_DAYTIME_LAB/outputs/github_curation_safe_push/full_file_inventory.csv`
- `BOT_V2_DAYTIME_LAB/outputs/github_curation_safe_push/github_exclude_manifest.csv`
- `BOT_V2_DAYTIME_LAB/outputs/github_curation_safe_push/github_include_manifest.csv`
### otros
- `.gitignore`

## 3. Archivos removidos del tracking
- `000_PARA_CHATGPT.zip` (5639517 bytes): remove cached zip from Git; keep local

## 4. Archivos excluidos
- raw/tick/heavy data
- secrets/local credentials
- MT5/broker local files
- logs/caches/temp/vendor
- zips and backups
- old duplicate/unselected reports
- Phase41 source candidates with whitespace hygiene warnings left local

### Source candidates left local
- `BOT_V2_DAYTIME_LAB/src/phase41_compare_replay.py`: left local: pre-commit whitespace hygiene warning; no source edit performed in curation pass
- `BOT_V2_DAYTIME_LAB/src/phase41_excel.py`: left local: pre-commit whitespace hygiene warning; no source edit performed in curation pass
- `BOT_V2_DAYTIME_LAB/src/phase41_manipulante_hybrid_replay.py`: left local: pre-commit whitespace hygiene warning; no source edit performed in curation pass
- `BOT_V2_DAYTIME_LAB/src/phase41_metrics.py`: left local: pre-commit whitespace hygiene warning; no source edit performed in curation pass

## 5. Secret Scan
- pass: True
- real findings: 0
- path keyword findings reviewed: 16

## 6. Archivos pesados
- >10 MB detectados: 300
- >25 MB detectados: 56
- accion: excluidos del commit; ningun >25 MB incluido.

## 7. .gitignore
- actualizado: si
- backup local: `.gitignore.phase_github_curation_backup`
- agregado/cubierto: env/key/pem/token/secret/password/credentials, caches, raw/tick, archives, MT5/Terminal/MetaQuotes/logs.

## 8. Seguridad
- no real: true
- no Exness: true
- no MT5 touched: true
- no trading: true
- no strategy changes: true
- no git add dot: true

## 9. Commit / Push
- commit message: `Curate and push essential BOT project files`
- hash: pending hasta crear commit
- push: pending hasta crear commit

## 10. Warnings
- Workspace had many unrelated pre-existing modified/untracked files; commit is selective only.
- 000_PARA_CHATGPT.zip remains local but is removed from Git tracking because it is duplicative.
- Secret scan found keyword/path false positives only; local sensitive provider configs remain excluded.
- Four Phase41 source candidates were left local because pre-commit whitespace check failed; no source code was edited in this curation pass.
