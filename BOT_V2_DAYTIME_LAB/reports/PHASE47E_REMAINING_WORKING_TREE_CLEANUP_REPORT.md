# PHASE47E - Remaining Working Tree Cleanup Report

Generated at: 2026-04-30T21:43:22-03:00

## 1. Lo mas importante

Phase47D quedo commiteado y pusheado. El working tree sigue sucio y no esta listo para Phase47A porque quedan cambios protegidos, Telegram pendiente, runtime/logs/backups, riesgo MT5/order-router y muchos unknowns.

## 2. Veredicto final exacto

WORKING_TREE_HAS_PROTECTED_REMAINS

## 3. Estado Git actual

- Branch: main
- Ultimo commit: c91cc47 Fix Manipulante MT5 reopen stop behavior
- git status --short lines antes del reporte: 2343
- Tracked modified/deleted antes del reporte: 27
- Untracked explicitos antes del reporte: 2342
- Staged: 0

## 4. Checks ejecutados

- Phase46 local: exit 0, GITHUB_CI_READY_WITH_WARNINGS.
- Phase47A guard: exit 1, LAB_ISOLATION_GUARD_FAIL_PROTECTED_CHANGE, 46 protected changes reportados por el guard.

## 5. Clasificacion por categoria

- PHASE47A_LAB_ISOLATION_READY_CANDIDATE: 12
- TELEGRAM_AUTO_ALERTS_PENDING_CANDIDATE: 9
- PHASE46_GENERATED_OR_NOISE: 7
- PHASE47B_47C_47D_REPORTS: 4
- RUNTIME_LOGS_DATA_DO_NOT_COMMIT: 9
- SECRETS_LOCAL_CONFIG_DO_NOT_COMMIT: 0
- BACKUPS_DO_NOT_COMMIT: 10
- PROTECTED_OPERATIONAL_CHANGE_REQUIRES_SEPARATE_DECISION: 28
- ORDER_ROUTER_OR_MT5_EXECUTION_RISK: 11
- UNKNOWN_REQUIRES_MANUAL_REVIEW: 2279

## 6. Telegram/alerts

Estado: PENDING_LOCAL_CHANGES_REQUIRES_SEPARATE_REVIEW.

- M BOT_V2_DAYTIME_LAB/src/phase45_run_alert_check.py
- M BOT_V2_DAYTIME_LAB/src/phase45_telegram_sender.py
- M MANIPULANTE/16_OBSERVABILITY/alerts/README_ALERTS.md
- M MANIPULANTE/16_OBSERVABILITY/alerts/alerts_config.example.json
- M MANIPULANTE/START_MANIPULANTE.bat
- ?? MANIPULANTE/16_OBSERVABILITY/alerts/START_ALERTS_LOOP_MANIPULANTE.bat
- ?? MANIPULANTE/16_OBSERVABILITY/alerts/STATUS_ALERTS_LOOP_MANIPULANTE.bat
- ?? MANIPULANTE/16_OBSERVABILITY/alerts/STOP_ALERTS_LOOP_MANIPULANTE.bat
- ?? reports/MANIPULANTE_AUTO_ALERTS_START_STOP_REPORT.md

Base ya commiteada: b446885 Phase45 Manipulante free alerts Telegram email foundation. El auto-start/loop actual sigue local y debe revisarse en una fase separada, sin mezclar Phase47A.

## 7. Phase47A

Estado: ARTIFACTS_PRESENT_BUT_BLOCKED_BY_PROTECTED_REMAINS.

- ?? BOT_V2_DAYTIME_LAB/reports/PHASE47A_LAB_ISOLATION_AND_MANIPULANTE_PROTECTION_REPORT.json
- ?? BOT_V2_DAYTIME_LAB/reports/PHASE47A_LAB_ISOLATION_AND_MANIPULANTE_PROTECTION_REPORT.md
- ?? BOT_V2_DAYTIME_LAB/src/phase47a_lab_isolation_guard.py
- ?? LAB_STRATEGIES/EURUSD_DAYTIME/README_EURUSD_DAYTIME_LAB.md
- ?? LAB_STRATEGIES/EURUSD_DAYTIME/_templates/strategy_config_template.json
- ?? LAB_STRATEGIES/EURUSD_DAYTIME/_templates/strategy_research_template.md
- ?? LAB_STRATEGIES/EURUSD_DAYTIME/correlation/.gitkeep
- ?? LAB_STRATEGIES/EURUSD_DAYTIME/reports/.gitkeep
- ?? LAB_STRATEGIES/EURUSD_DAYTIME/shared/.gitkeep
- ?? LAB_STRATEGIES/EURUSD_DAYTIME/strategies/.gitkeep
- ?? LAB_STRATEGIES/README_LAB_STRATEGIES.md
- ?? PROJECT_ZONES_AND_BRANCHING_RULES.md

Los JSON de Phase47A validan, pero el guard falla por remanentes externos. El reporte Phase47A conserva conteos de su momento original y no fue actualizado en esta fase.

## 8. Runtime/logs/data fuera de commit

- ?? BOT_V2_DAYTIME_LAB/src/BOT_V2_DAYTIME_LAB/outputs/phase35_final_real_readiness_audit/dry_run_order_simulation/phase35_dry_run_orders.csv
- ?? MANIPULANTE/10_LOGS_PAPER/dry_run_decisions/2026-04-29_decisions.csv
- ?? MANIPULANTE/10_LOGS_PAPER/ftmo_trial_bot/decisions.csv
- ?? MANIPULANTE/10_LOGS_PAPER/ftmo_trial_bot/heartbeat.json
- ?? MANIPULANTE/10_LOGS_PAPER/ftmo_trial_bot/heartbeat.txt
- ?? MANIPULANTE/10_LOGS_PAPER/ftmo_trial_bot/quick_status.txt
- ?? MANIPULANTE/16_OBSERVABILITY/alerts/runtime/alerts_loop.last_heartbeat.json
- ?? MANIPULANTE/16_OBSERVABILITY/daily/latest_health_snapshot.json
- ?? results_REHEARSAL/SCBI_FORWARD_TELEMETRY_TRACE.csv

## 9. Backups fuera de commit

- ?? BOT_V2_DAYTIME_LAB/reports/PHASE46_GITHUB_CI_SAFETY_TESTS_REPORT.json.bak_phase47a_lab_isolation_20260430_205314
- ?? BOT_V2_DAYTIME_LAB/reports/PHASE46_GITHUB_CI_SAFETY_TESTS_REPORT.md.bak_phase47a_lab_isolation_20260430_205314
- ?? BOT_V2_DAYTIME_LAB/reports/PHASE47D_MIXED_CHANGES_NOT_COMMITTED.patch.txt
- ?? BOT_V2_DAYTIME_LAB/src/phase37ze_quick_status_panel.py.bak_auto_alerts_20260430_175900
- ?? BOT_V2_DAYTIME_LAB/src/phase45_run_alert_check.py.bak_auto_alerts_20260430_175900
- ?? MANIPULANTE/START_MANIPULANTE.bat.bak_auto_alerts_20260430_175900
- ?? MANIPULANTE/STATUS_MANIPULANTE.bat.bak_auto_alerts_20260430_175900
- ?? MANIPULANTE/STOP_MANIPULANTE.bat.bak_auto_alerts_20260430_175900
- ?? legacy_archive_2026/000_PARA_CHATGPT_BACKUP_20260421_211526.zipbak
- ?? legacy_archive_2026/_test.zipbak

## 10. Riesgo MT5/order execution

- M MANIPULANTE/09_COMPLIANCE/MT5_LIVE_NEWS_ADAPTER/MANIPULANTE_CalendarBootstrapEA.mq5
- M mt5_demo_executor_lab/mt5_order_router.py
- ?? BOT_V2_DAYTIME_LAB/reports/PHASE36R_37A_ORDER_SEND_REPAIR_MICRO_REAL_GATE_REPORT.json
- ?? BOT_V2_DAYTIME_LAB/reports/PHASE36R_37A_ORDER_SEND_REPAIR_MICRO_REAL_GATE_REPORT.md
- ?? MANIPULANTE/09_COMPLIANCE/live_news_cache/mql5_bootstrap/ftmo_news_gate_status.json
- ?? MANIPULANTE/09_COMPLIANCE/live_news_cache/mql5_bootstrap/ftmo_news_today.json
- ?? MANIPULANTE/09_COMPLIANCE/live_news_cache/mql5_bootstrap/ftmo_news_week.json
- ?? mt5_demo_executor_lab/mt5_order_router.py.phase36r_backup
- ?? mt5_demo_executor_lab/outputs/mt5_demo_log.csv
- ?? mt5_demo_executor_lab/outputs/mt5_demo_telemetry.csv
- ?? phase37e_run_mql5_calendar_script.py

## 11. Protected operational changes

Total expanded protected marks: 59. Muestra de cambios que requieren decision separada:
- M MANIPULANTE/04_OPERACION_DIARIA/MANIPULANTE_DAILY_RUNBOOK.md
- M MANIPULANTE/04_OPERACION_DIARIA/MANIPULANTE_KILL_SWITCH.md
- M MANIPULANTE/12_MICRO_REAL_READINESS/MICRO_REAL_KILL_SWITCH.md
- M MANIPULANTE/12_MICRO_REAL_READINESS/MICRO_REAL_POSITION_SIZE_POLICY.md
- D MANIPULANTE/STATUS_TECNICO_MANIPULANTE.bat
- ?? BOT_V2_DAYTIME_LAB/reports/PHASE36_LIVE_NEWS_MT5_DRYRUN_READINESS_REPORT.json
- ?? BOT_V2_DAYTIME_LAB/reports/PHASE36_LIVE_NEWS_MT5_DRYRUN_READINESS_REPORT.md
- ?? BOT_V2_DAYTIME_LAB/src/phase45b_tests.py
- ?? MANIPULANTE/08_CHECKLISTS/CHECKLIST_LIVE_NEWS_GATE.md
- ?? MANIPULANTE/09_COMPLIANCE/API_LIVE_NEWS_PROVIDER/API_NEWS_PROVIDER_POLICY.md
- ?? MANIPULANTE/09_COMPLIANCE/API_LIVE_NEWS_PROVIDER/API_NEWS_PROVIDER_README.md
- ?? MANIPULANTE/09_COMPLIANCE/API_LIVE_NEWS_PROVIDER/API_NEWS_PROVIDER_SETUP.md
- ?? MANIPULANTE/09_COMPLIANCE/API_LIVE_NEWS_PROVIDER/api_news_provider_config.example.json
- ?? MANIPULANTE/09_COMPLIANCE/LIVE_NEWS_FORTRESS_POLICY.md
- ?? MANIPULANTE/09_COMPLIANCE/live_news_cache/2026-04-29_news_gate_status.json
- ?? MANIPULANTE/09_COMPLIANCE/live_news_cache/2026-04-29_news_today.json
- ?? MANIPULANTE/09_COMPLIANCE/live_news_cache/2026-04-29_news_week.json
- ?? MANIPULANTE/09_COMPLIANCE/live_news_cache/api_provider/2026-04-29_api_news_gate_status.json
- ?? MANIPULANTE/09_COMPLIANCE/live_news_cache/api_provider/2026-04-29_api_news_today.json
- ?? MANIPULANTE/09_COMPLIANCE/live_news_cache/api_provider/2026-04-29_api_news_week.json
- ?? MANIPULANTE/09_COMPLIANCE/live_news_cache/api_provider/2026-04-30_api_news_gate_status.json
- ?? MANIPULANTE/09_COMPLIANCE/live_news_cache/api_provider/2026-04-30_api_news_today.json
- ?? MANIPULANTE/09_COMPLIANCE/live_news_cache/api_provider/2026-04-30_api_news_week.json
- ?? MANIPULANTE/09_COMPLIANCE/live_news_fortress_config.json
- ?? MANIPULANTE/09_COMPLIANCE/news_automation_status.json
- ?? MANIPULANTE/12_MICRO_REAL_READINESS/MICRO_REAL_ACTIVATION_PROTOCOL.md
- ?? MANIPULANTE/12_MICRO_REAL_READINESS/NO_REAL_UNTIL_CONFIRMATION.md
- ?? MANIPULANTE/13_FTMO_TRIAL_AUTOMATION/STOP_BOT.txt

## 12. Unknowns

Total unknowns: 2279. Muestra:
- M legacy_archive_2026/_audit_dest_extras.ps1
- M reports/canonical_context_master_20260416_163108/AGENTS.md
- M reports/canonical_context_master_20260416_163108/RESEARCH_OPERATING_SYSTEM.md
- M reports/canonical_context_master_20260416_163108/RISK_PROTOCOL.md
- M reports/canonical_microstructure_iter2_20260416_172155/AGENTS.md
- M reports/canonical_microstructure_iter2_20260416_172155/RESEARCH_OPERATING_SYSTEM.md
- M reports/canonical_microstructure_iter2_20260416_172155/RISK_PROTOCOL.md
- M research_lab/README.md
- ?? BOT_V2_DAYTIME_LAB/ZIP_UPLOAD_IDENTITY_MARKER.md
- ?? BOT_V2_DAYTIME_LAB/configs/phase24_forward_demo_candidate_config.json
- ?? BOT_V2_DAYTIME_LAB/configs/phase24_forward_demo_candidate_config_hash.txt
- ?? BOT_V2_DAYTIME_LAB/configs/phase25_forward_demo_candidate_config.json
- ?? BOT_V2_DAYTIME_LAB/configs/phase25_forward_demo_candidate_config_hash.txt
- ?? BOT_V2_DAYTIME_LAB/configs/prop_firm_rules_config.json
- ?? BOT_V2_DAYTIME_LAB/docs/CANONICAL_ZIP_CHECKLIST.md
- ?? BOT_V2_DAYTIME_LAB/docs/CANONICAL_ZIP_OPERATING_STANDARD.md
- ?? BOT_V2_DAYTIME_LAB/docs/PHASE25_DAILY_RUNBOOK.md
- ?? BOT_V2_DAYTIME_LAB/docs/PHASE25_FORWARD_DEMO_PROTOCOL.md
- ?? BOT_V2_DAYTIME_LAB/docs/PHASE25_FORWARD_REVIEW_CRITERIA.md
- ?? BOT_V2_DAYTIME_LAB/docs/PHASE25_KILL_SWITCH_POLICY.md
- ?? BOT_V2_DAYTIME_LAB/docs/PHASE26B_DATA_ACQUISITION_REQUIREMENTS_2015_2019.md
- ?? BOT_V2_DAYTIME_LAB/docs/PHASE26B_DATA_CERTIFICATION_CHECKLIST_2015_2019.md
- ?? BOT_V2_DAYTIME_LAB/docs/PHASE32C_FUNDEDNEXT_CHECKOUT_VERIFICATION_CHECKLIST.md
- ?? BOT_V2_DAYTIME_LAB/docs/PHASE32C_FUNDEDNEXT_KILL_SWITCH.md
- ?? BOT_V2_DAYTIME_LAB/docs/PHASE32C_FUNDEDNEXT_PRE_TRADE_CHECKLIST.md
- ?? BOT_V2_DAYTIME_LAB/docs/PHASE32C_FUNDEDNEXT_STELLAR_LITE_OPERATIONAL_RULEBOOK.md
- ?? BOT_V2_DAYTIME_LAB/docs/PHASE32C_FUNDEDNEXT_WEEKEND_POLICY.md
- ?? BOT_V2_DAYTIME_LAB/docs/PHASE32_DAILY_RUNBOOK.md
- ?? BOT_V2_DAYTIME_LAB/docs/PHASE32_DUAL_LEDGER_PROTOCOL.md
- ?? BOT_V2_DAYTIME_LAB/docs/PHASE32_FTMO_PAPER_EVALUATION_PLAN.md
- ?? BOT_V2_DAYTIME_LAB/docs/PHASE32_KILL_SWITCH_POLICY.md
- ?? BOT_V2_DAYTIME_LAB/docs/PHASE32_REVIEW_CRITERIA.md
- ?? BOT_V2_DAYTIME_LAB/docs/PHASE32_RISK_POLICY.md
- ?? BOT_V2_DAYTIME_LAB/reports/MANIPULANTE_DAILY_FORWARD_REVIEW_20260430.md
- ?? BOT_V2_DAYTIME_LAB/reports/PHASE36S_LIVE_NEWS_LOT_FEASIBILITY_REPORT.json
- ?? BOT_V2_DAYTIME_LAB/reports/PHASE36S_LIVE_NEWS_LOT_FEASIBILITY_REPORT.md
- ?? BOT_V2_DAYTIME_LAB/reports/PHASE37B_FTMO_TRIAL_NEWS_SIGNAL_FINALIZATION_REPORT.json
- ?? BOT_V2_DAYTIME_LAB/reports/PHASE37B_FTMO_TRIAL_NEWS_SIGNAL_FINALIZATION_REPORT.md
- ?? BOT_V2_DAYTIME_LAB/reports/PHASE37C_FULL_AUTO_FTMO_TRIAL_BOOTSTRAP_REPORT.json
- ?? BOT_V2_DAYTIME_LAB/reports/PHASE37C_FULL_AUTO_FTMO_TRIAL_BOOTSTRAP_REPORT.md
- ?? BOT_V2_DAYTIME_LAB/reports/PHASE37D_FTMO_TRIAL_API_NEWS_SIGNAL_AUTO_REPORT.json
- ?? BOT_V2_DAYTIME_LAB/reports/PHASE37D_FTMO_TRIAL_API_NEWS_SIGNAL_AUTO_REPORT.md
- ?? BOT_V2_DAYTIME_LAB/reports/PHASE37_FTMO_SWING_FREE_TRIAL_AUTOMATION_REPORT.json
- ?? BOT_V2_DAYTIME_LAB/reports/PHASE37_FTMO_SWING_FREE_TRIAL_AUTOMATION_REPORT.md
- ?? BOT_V2_DAYTIME_LAB/src/BOT_V2_DAYTIME_LAB/outputs/phase35_final_real_readiness_audit/dry_run_order_simulation/phase35_dry_run_order_simulation.json
- ?? BOT_V2_DAYTIME_LAB/src/BOT_V2_DAYTIME_LAB/outputs/phase35_final_real_readiness_audit/dry_run_order_simulation/phase35_dry_run_order_simulation.md
- ?? BOT_V2_DAYTIME_LAB/src/BOT_V2_DAYTIME_LAB/outputs/phase35_final_real_readiness_audit/risk_lot_audit/phase35_lot_scenarios.csv
- ?? BOT_V2_DAYTIME_LAB/src/BOT_V2_DAYTIME_LAB/outputs/phase35_final_real_readiness_audit/risk_lot_audit/phase35_risk_lot_audit.json
- ?? BOT_V2_DAYTIME_LAB/src/BOT_V2_DAYTIME_LAB/outputs/phase35_final_real_readiness_audit/risk_lot_audit/phase35_risk_lot_audit.md
- ?? BOT_V2_DAYTIME_LAB/src/phase36_exness_lot_validator.py
- ?? BOT_V2_DAYTIME_LAB/src/phase36_exness_symbol_gate.py
- ?? BOT_V2_DAYTIME_LAB/src/phase36_live_data_quality_gate.py
- ?? BOT_V2_DAYTIME_LAB/src/phase36_live_news_fortress.py
- ?? BOT_V2_DAYTIME_LAB/src/phase36_manipulante_dry_run_engine.py
- ?? BOT_V2_DAYTIME_LAB/src/phase36_readiness_orchestrator.py
- ?? BOT_V2_DAYTIME_LAB/src/phase36_server_time_validator.py
- ?? BOT_V2_DAYTIME_LAB/src/phase36_time_gate_validator.py
- ?? BOT_V2_DAYTIME_LAB/src/phase36r_37a_micro_real_gate_orchestrator.py
- ?? BOT_V2_DAYTIME_LAB/src/phase36s_live_news_lot_feasibility_orchestrator.py
- ?? BOT_V2_DAYTIME_LAB/src/phase36s_lot_feasibility_100usd.py

La lista completa por categoria queda en el JSON Phase47E.

## 13. Propuesta de secuencia

1. Revisar Telegram/alerts como Phase47F separada y decidir commit selectivo o correccion.
2. Mantener fuera runtime/logs/data/backups y revisar .gitignore minimo.
3. Revisar cambios protegidos y order-router/MT5 risk antes de cualquier commit adicional.
4. Cuando el working tree este controlado, commitear Phase47A solo con sus 12 archivos candidatos.

## 14. .gitignore sugerido

Sugerido para revision, no aplicado en esta fase:
- __pycache__/
- *.pyc
- *.pid
- *.lock
- *.sqlite
- *.db
- *.bak
- *.bak_*
- *.zipbak
- .env
- .env.*
- **/runtime/
- **/logs/
- **/alert_state.json
- **/alerts_loop.last_heartbeat.json
- **/bot_events.jsonl
- **/telemetry*.jsonl

## 15. Confirmaciones

- No se hizo git add.
- No se hizo commit.
- No se hizo push.
- No se hizo reset.
- No se hizo clean.
- No se hizo stash.
- No se toco estrategia.
- No se abrio MT5.
- No se ejecutaron ordenes.
- No se imprimieron secretos.

## 16. Siguiente paso unico

Revisar y aislar Telegram/alerts como Phase47F antes de intentar el commit limpio de Phase47A.
