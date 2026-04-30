# Pre Commit Review

- timestamp_utc: 2026-04-30T11:31:17.391912+00:00
- staged_count: 36
- staged_non_deleted_count: 35
- addition_size_mb: 8.919
- pass: True

## Git Diff Cached Stat
```
 .gitignore                                         |    36 +-
 000_PARA_CHATGPT.zip                               |   Bin 5639517 -> 0 bytes
 00_READ_THIS_FIRST.md                              |    19 +-
 01_CURRENT_PROJECT_STATUS.json                     |    28 +-
 01_CURRENT_PROJECT_STATUS.md                       |    19 +-
 02_STRATEGY_AUTHORITY_MAP.json                     |    29 +-
 02_STRATEGY_AUTHORITY_MAP.md                       |    16 +-
 .../full_file_inventory.csv                        | 35749 +++++++++++++++++++
 .../github_curation_safe_push/git_add_plan.json    |    30 +
 .../github_curation_safe_push/git_add_plan.md      |    88 +
 .../github_curation_preflight.json                 |  1642 +
 .../github_curation_preflight.md                   |   486 +
 .../github_exclude_manifest.csv                    | 33898 ++++++++++++++++++
 .../github_exclude_manifest.txt                    |   513 +
 .../github_include_manifest.csv                    |    62 +
 .../github_include_manifest.txt                    |    61 +
 .../inventory_summary.json                         |    22 +
 .../manifest_summary.json                          |    99 +
 .../pre_commit_review.json                         |   160 +
 .../github_curation_safe_push/pre_commit_review.md |    93 +
 .../secret_scan_report.json                        |  3511 ++
 .../secret_scan_report.md                          |    34 +
 .../reports/GITHUB_CURATION_SAFE_PUSH_REPORT.json  |   111 +
 .../reports/GITHUB_CURATION_SAFE_PUSH_REPORT.md    |    95 +
 ...PULANTE_HYBRID_REPLAY_FORWARD_AUDIT_REPORT.json |     8 +
 ...NIPULANTE_HYBRID_REPLAY_FORWARD_AUDIT_REPORT.md |    19 +
 CHANGELOG.md                                       |     2 +-
 CLOUD_WORKFLOW.md                                  |     6 +-
 .../MANIPULANTE_REPLAY_FORWARD_RESUMEN.md          |    19 +
 ...NIPULANTE_HYBRID_REPLAY_FORWARD_AUDIT_REPORT.md |    40 +
 .../MANIPULANTE_PROMOTION_GATE_TO_REAL.md          |     8 +-
 .../phase42_operational_stress_tests.csv           |     6 +
 .../phase42b_operational_stress_tests_16.csv       |    17 +
 README.md                                          |     6 +-
 .../ROCKI_AM_SOURCE_MANIFEST.csv                   |     6 +
 ROCKI_AM/03_METRICAS/ROCKI_AM_METRICS_SUMMARY.csv  |    16 +
 36 files changed, 76870 insertions(+), 84 deletions(-)
```

## Staged Files
- M `.gitignore`
- D `000_PARA_CHATGPT.zip`
- M `00_READ_THIS_FIRST.md`
- M `01_CURRENT_PROJECT_STATUS.json`
- M `01_CURRENT_PROJECT_STATUS.md`
- M `02_STRATEGY_AUTHORITY_MAP.json`
- M `02_STRATEGY_AUTHORITY_MAP.md`
- A `BOT_V2_DAYTIME_LAB/outputs/github_curation_safe_push/full_file_inventory.csv`
- A `BOT_V2_DAYTIME_LAB/outputs/github_curation_safe_push/git_add_plan.json`
- A `BOT_V2_DAYTIME_LAB/outputs/github_curation_safe_push/git_add_plan.md`
- A `BOT_V2_DAYTIME_LAB/outputs/github_curation_safe_push/github_curation_preflight.json`
- A `BOT_V2_DAYTIME_LAB/outputs/github_curation_safe_push/github_curation_preflight.md`
- A `BOT_V2_DAYTIME_LAB/outputs/github_curation_safe_push/github_exclude_manifest.csv`
- A `BOT_V2_DAYTIME_LAB/outputs/github_curation_safe_push/github_exclude_manifest.txt`
- A `BOT_V2_DAYTIME_LAB/outputs/github_curation_safe_push/github_include_manifest.csv`
- A `BOT_V2_DAYTIME_LAB/outputs/github_curation_safe_push/github_include_manifest.txt`
- A `BOT_V2_DAYTIME_LAB/outputs/github_curation_safe_push/inventory_summary.json`
- A `BOT_V2_DAYTIME_LAB/outputs/github_curation_safe_push/manifest_summary.json`
- A `BOT_V2_DAYTIME_LAB/outputs/github_curation_safe_push/pre_commit_review.json`
- A `BOT_V2_DAYTIME_LAB/outputs/github_curation_safe_push/pre_commit_review.md`
- A `BOT_V2_DAYTIME_LAB/outputs/github_curation_safe_push/secret_scan_report.json`
- A `BOT_V2_DAYTIME_LAB/outputs/github_curation_safe_push/secret_scan_report.md`
- A `BOT_V2_DAYTIME_LAB/reports/GITHUB_CURATION_SAFE_PUSH_REPORT.json`
- A `BOT_V2_DAYTIME_LAB/reports/GITHUB_CURATION_SAFE_PUSH_REPORT.md`
- A `BOT_V2_DAYTIME_LAB/reports/PHASE41_MANIPULANTE_HYBRID_REPLAY_FORWARD_AUDIT_REPORT.json`
- A `BOT_V2_DAYTIME_LAB/reports/PHASE41_MANIPULANTE_HYBRID_REPLAY_FORWARD_AUDIT_REPORT.md`
- M `CHANGELOG.md`
- M `CLOUD_WORKFLOW.md`
- A `MANIPULANTE/00_LEER_PRIMERO/MANIPULANTE_REPLAY_FORWARD_RESUMEN.md`
- A `MANIPULANTE/14_ANALISIS/MANIPULANTE_HYBRID_REPLAY_FORWARD_AUDIT_REPORT.md`
- M `MANIPULANTE/15_FORWARD_DEMO_SCORECARD/promotion_gate/MANIPULANTE_PROMOTION_GATE_TO_REAL.md`
- A `MANIPULANTE/15_FORWARD_DEMO_SCORECARD/stress_tests/phase42_operational_stress_tests.csv`
- A `MANIPULANTE/15_FORWARD_DEMO_SCORECARD/stress_tests/phase42b_operational_stress_tests_16.csv`
- M `README.md`
- A `ROCKI_AM/01_ORIGEN_HISTORICO/ROCKI_AM_SOURCE_MANIFEST.csv`
- A `ROCKI_AM/03_METRICAS/ROCKI_AM_METRICS_SUMMARY.csv`

## Excluded After Check
- `BOT_V2_DAYTIME_LAB/src/phase41_compare_replay.py`: left local: pre-commit whitespace hygiene warning; no source edit performed in curation pass
- `BOT_V2_DAYTIME_LAB/src/phase41_excel.py`: left local: pre-commit whitespace hygiene warning; no source edit performed in curation pass
- `BOT_V2_DAYTIME_LAB/src/phase41_manipulante_hybrid_replay.py`: left local: pre-commit whitespace hygiene warning; no source edit performed in curation pass
- `BOT_V2_DAYTIME_LAB/src/phase41_metrics.py`: left local: pre-commit whitespace hygiene warning; no source edit performed in curation pass

## Safety Checks
- sensitive filename issues: 0
- content secret hits: 0
- files >25 MB: 0
- archive additions: 0
- raw/tick data issues: 0
- MT5 account/local issues: 0
