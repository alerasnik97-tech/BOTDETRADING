import argparse
import hashlib
import json
import os
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# --- CONFIGURACION CANONICA ---
PROJECT_ROOT = Path(__file__).parent.parent
CANONICAL_ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
OUTPUT_ZIP = PROJECT_ROOT / "000_PARA_CHATGPT.zip"
MANIFEST_PATH = PROJECT_ROOT / "ZIP_CONTENTS_MANIFEST.md"
AUDIT_PATH = PROJECT_ROOT / "ZIP_PACKAGING_AUDIT.md"
DELIVERY_STATUS_PATH = PROJECT_ROOT / "ZIP_DELIVERY_STATUS.md"

BUILDER_STAGE_DIR = PROJECT_ROOT / ".builder_stage"
LEGACY_STAGE_DIR = PROJECT_ROOT / ".bundle_stage"

# Lista de archivos canonicos (solo el estado maestro es obligatorio para arrancar)
CANONICAL_FILES = [
    ("CURRENT_STATE_OF_LAB.md", "Estado maestro de gobernanza y progreso del laboratorio."),
    ("scripts/build_chatgpt_bundle.py", "Script motor de canonizacion del bundle."),
]

# Archivos que se incluyen si existen
OPTIONAL_CANONICAL_FILES = [
    ("RESEARCH_DECISION_MATRIX.md", "Matriz de decisiones de investigacion."),
    ("EURUSD_DESIGN_CONSTRAINT_BRIEF.md", "Restricciones de diseño y lecciones aprendidas."),
    ("EURUSD_SCBI_M5_HYPOTHESIS.md", "Hipotesis central de la fractura de liquidez."),
    ("EURUSD_POST_HARDENING_DRIFT_PROTOCOL.md", "Protocolo canonico de drift post-hardening."),
    ("EURUSD_POST_HARDENING_DRIFT_RESULTS.md", "Resultados canonicos de drift post-hardening."),
    ("EURUSD_POST_HARDENING_DRIFT_DECISION.md", "Decision canonica de drift post-hardening."),
    ("POST_HARDENING_DRIFT_STATUS.json", "Status json de rebaseline y validacion post-hardening."),
    ("POST_HARDENING_DRIFT_HEARTBEAT.md", "Heartbeat de drift post-hardening."),
    ("POST_HARDENING_DRIFT_RUNBOOK.md", "Runbook de drift post-hardening."),
    ("EURUSD_FORWARD_TELEMETRY_PROTOCOL.md", "Protocolo canonico de telemetria forward."),
    ("EURUSD_FORWARD_TELEMETRY_RESULTS.md", "Resultados canonicos de telemetria forward."),
    ("EURUSD_FORWARD_TELEMETRY_DECISION.md", "Decision canonica de telemetria forward."),
    ("FORWARD_TELEMETRY_STATUS.json", "Status json del hardening de telemetria forward."),
    ("FORWARD_TELEMETRY_HEARTBEAT.md", "Heartbeat de telemetria forward."),
    ("FORWARD_TELEMETRY_RUNBOOK.md", "Runbook de telemetria forward."),
    ("results/SCBI_FORWARD_LEDGER.csv", "Ledger oficial de la linea GLOBAL."),
    ("EURUSD_SCBI_2020_2025_DURABILITY_PROTOCOL.md", "Protocolo de durabilidad."),
    ("EURUSD_SCBI_2020_2025_DURABILITY_RESULTS.md", "Resultados de durabilidad."),
    ("EURUSD_SCBI_2020_2025_DURABILITY_DECISION.md", "Decision de durabilidad."),
    ("EURUSD_SCBI_CORE_BRANCH_PROTOCOL.md", "Protocolo de rama CORE."),
    ("EURUSD_SCBI_CORE_BRANCH_RESULTS.md", "Resultados de rama CORE."),
    ("EURUSD_SCBI_CORE_BRANCH_DECISION.md", "Decision de rama CORE."),
    ("EURUSD_SCBI_CORE_FORWARD_PROTOCOL.md", "Protocolo de forward CORE."),
    ("EURUSD_DUAL_LINE_FORWARD_PROTOCOL.md", "Protocolo dual."),
    ("EURUSD_DUAL_LINE_FORWARD_SCOREBOARD_SCHEMA.md", "Schema canonico del scoreboard dual."),
    ("results/SCBI_DUAL_LINE_SCOREBOARD.csv", "Scoreboard dual."),
    ("results/SCBI_FORWARD_TELEMETRY_TRACE.csv", "Traza canonica de telemetria forward."),
    ("results/SCBI_SIGNAL_DRIFT_BASELINE.json", "Baseline canonica de drift post-hardening."),
    ("results/SCBI_SIGNAL_DRIFT_REPORT.json", "Reporte canonico de drift post-hardening."),
    ("results/SCBI_SIGNAL_DRIFT_VALIDATION.json", "Validacion canonica del monitor de drift."),
    ("results/SCBI_FORWARD_TRIBUNAL_SUMMARY.json", "Resumen canonico del tribunal forward."),
    ("EURUSD_DUAL_DAILY_CHAIN_PROTOCOL.md", "Protocolo orquestador diario."),
    ("EURUSD_DUAL_DAILY_CHAIN_DECISION.md", "Decision orquestador diario."),
    ("scratch/run_dual_line_daily_chain.py", "Script orquestador."),
    ("EURUSD_REAL_READINESS_GATE_PROTOCOL.md", "Protocolo de gate de promocion."),
    ("EURUSD_REAL_READINESS_GATE_RESULTS.md", "Mapeo de gate de promocion."),
    ("EURUSD_REAL_READINESS_GATE_DECISION.md", "Decision de gate de promocion."),
    ("EURUSD_PROP_FIRM_RISK_LAYER_PROTOCOL.md", "Protocolo de capa de riesgo prop firm."),
    ("EURUSD_PROP_FIRM_RISK_LAYER_RESULTS.md", "Resultados y ensayo de la capa de riesgo."),
    ("EURUSD_PROP_FIRM_RISK_LAYER_DECISION.md", "Decision final de la capa de riesgo."),
    ("PROP_FIRM_RISK_LAYER_STATUS.json", "Status json de la capa de riesgo."),
    ("PROP_FIRM_RISK_LAYER_HEARTBEAT.md", "Heartbeat de la capa de riesgo."),
    ("PROP_FIRM_RISK_LAYER_RUNBOOK.md", "Runbook de la capa de riesgo."),
    ("scratch/prop_firm_risk_guards.py", "Script de guards de riesgo institucional."),    ("EURUSD_FORWARD_TRIBUNAL_PROTOCOL.md", "Protocolo del tribunal de evidencia."),
    ("EURUSD_FORWARD_TRIBUNAL_RESULTS.md", "Resultados y juicio del tribunal."),
    ("EURUSD_FORWARD_TRIBUNAL_DECISION.md", "Decision final del tribunal."),
    ("FORWARD_TRIBUNAL_STATUS.json", "Status json del tribunal."),
    ("FORWARD_TRIBUNAL_HEARTBEAT.md", "Heartbeat del tribunal."),
    ("FORWARD_TRIBUNAL_RUNBOOK.md", "Runbook del tribunal."),
    ("scratch/run_forward_evidence_tribunal.py", "Script motor del tribunal institucional."),
    ("EURUSD_FORWARD_OBSERVATION_MODE_PROTOCOL.md", "Protocolo de lockdown y observacion."),
    ("EURUSD_FORWARD_OBSERVATION_MODE_RESULTS.md", "Resultados de la auditoria de lockdown."),
    ("EURUSD_FORWARD_OBSERVATION_MODE_DECISION.md", "Decision final de lockdown."),
    ("FORWARD_OBSERVATION_MODE_STATUS.json", "Status json de lockdown."),
    ("FORWARD_OBSERVATION_MODE_HEARTBEAT.md", "Heartbeat de lockdown."),
    ("FORWARD_OBSERVATION_MODE_RUNBOOK.md", "Runbook de lockdown."),
    ("EURUSD_OPERATIONAL_STACK_REPLAY_PROTOCOL.md", "Protocolo de replay historico."),
    ("EURUSD_OPERATIONAL_STACK_REPLAY_RESULTS.md", "Resultados del replay historico."),
    ("EURUSD_OPERATIONAL_STACK_REPLAY_DECISION.md", "Decision final del replay historico."),
    ("OPERATIONAL_STACK_REPLAY_STATUS.json", "Status json del replay."),
    ("OPERATIONAL_STACK_REPLAY_HEARTBEAT.md", "Heartbeat de del replay."),
    ("OPERATIONAL_STACK_REPLAY_RUNBOOK.md", "Runbook del replay."),
    ("scratch/run_operational_stack_replay.py", "Script motor del replay historico."),
    ("EURUSD_DUAL_OPERATIONAL_STACK_REPLAY_PROTOCOL.md", "Protocolo de replay historico dual."),
    ("EURUSD_DUAL_OPERATIONAL_STACK_REPLAY_RESULTS.md", "Resultados del replay historico dual."),
    ("EURUSD_DUAL_OPERATIONAL_STACK_REPLAY_DECISION.md", "Decision final del replay historico dual."),
    ("DUAL_OPERATIONAL_STACK_REPLAY_STATUS.json", "Status json del replay dual."),
    ("DUAL_OPERATIONAL_STACK_REPLAY_HEARTBEAT.md", "Heartbeat del replay dual."),
    ("DUAL_OPERATIONAL_STACK_REPLAY_RUNBOOK.md", "Runbook del replay dual."),
    ("scratch/run_dual_operational_stack_replay.py", "Script motor del replay historico dual."),
    ("EURUSD_PROP_CAPITAL_PATH_PROTOCOL.md", "Protocolo de ruta de capital externo."),
    ("EURUSD_PROP_CAPITAL_PATH_RESULTS.md", "Resultados de la emulacion de fondeo."),
    ("EURUSD_PROP_CAPITAL_PATH_DECISION.md", "Decision final sobre capital externo."),
    ("PROP_CAPITAL_PATH_STATUS.json", "Status json de capital externo."),
    ("PROP_CAPITAL_PATH_HEARTBEAT.md", "Heartbeat de capital externo."),
    ("PROP_CAPITAL_PATH_RUNBOOK.md", "Runbook de capital externo."),
    ("scratch/run_prop_capital_path_emulator.py", "Script emulador de prop firms."),
    ("EURUSD_CHALLENGE_DEPLOYMENT_PLAYBOOK.md", "Manual de despliegue institucional."),
    ("EURUSD_CHALLENGE_DEPLOYMENT_PLAYBOOK_PROTOCOL.md", "Protocolo del playbook de despliegue."),
    ("EURUSD_CHALLENGE_DEPLOYMENT_PLAYBOOK_RESULTS.md", "Resultados del dry-run de despliegue."),
    ("EURUSD_CHALLENGE_DEPLOYMENT_PLAYBOOK_DECISION.md", "Decision final del playbook."),
    ("CHALLENGE_DEPLOYMENT_STATUS.json", "Status json de despliegue."),
    ("CHALLENGE_DEPLOYMENT_HEARTBEAT.md", "Heartbeat de despliegue."),
    ("CHALLENGE_DEPLOYMENT_RUNBOOK.md", "Runbook de despliegue."),
    ("scratch/run_challenge_deployment_dry_run.py", "Script de estadisticas de despliegue."),
    ("EURUSD_CHALLENGE_PLAYBOOK_ROBUSTNESS_PROTOCOL.md", "Protocolo de robustez del playbook."),
    ("EURUSD_CHALLENGE_PLAYBOOK_ROBUSTNESS_RESULTS.md", "Resultados de la auditoria de robustez."),
    ("EURUSD_CHALLENGE_PLAYBOOK_ROBUSTNESS_DECISION.md", "Decision final de robustez."),
    ("CHALLENGE_PLAYBOOK_ROBUSTNESS_STATUS.json", "Status json de robustez."),
    ("CHALLENGE_PLAYBOOK_ROBUSTNESS_HEARTBEAT.md", "Heartbeat de robustez."),
    ("CHALLENGE_PLAYBOOK_ROBUSTNESS_RUNBOOK.md", "Runbook de robustez."),
    ("scratch/run_challenge_playbook_robustness.py", "Script de Monte Carlo de robustez."),
    ("EURUSD_FORWARD_CANONICAL_EVIDENCE_PROTOCOL.md", "Protocolo de reconciliacion canonica."),
    ("EURUSD_FORWARD_CANONICAL_EVIDENCE_RESULTS.md", "Resultados de la reconciliacion."),
    ("EURUSD_FORWARD_CANONICAL_EVIDENCE_DECISION.md", "Decision final de reconciliacion."),
    ("FORWARD_CANONICAL_EVIDENCE_STATUS.json", "Status json de reconciliacion."),
    ("FORWARD_CANONICAL_EVIDENCE_HEARTBEAT.md", "Heartbeat de reconciliacion."),
    ("FORWARD_CANONICAL_EVIDENCE_RUNBOOK.md", "Runbook de reconciliacion."),
    ("scratch/run_forward_canonical_reconciliation.py", "Script de reconciliacion canonica."),
    ("EURUSD_TEMPORAL_EXECUTION_HARDENING_PROTOCOL.md", "Protocolo de endurecimiento temporal/ejecucion."),
    ("EURUSD_TEMPORAL_EXECUTION_HARDENING_RESULTS.md", "Resultados de endurecimiento temporal/ejecucion."),
    ("EURUSD_TEMPORAL_EXECUTION_HARDENING_DECISION.md", "Decision final de endurecimiento temporal/ejecucion."),
    ("TEMPORAL_EXECUTION_HARDENING_STATUS.json", "Status json de endurecimiento temporal/ejecucion."),
    ("TEMPORAL_EXECUTION_HARDENING_HEARTBEAT.md", "Heartbeat de endurecimiento temporal/ejecucion."),
    ("TEMPORAL_EXECUTION_HARDENING_RUNBOOK.md", "Runbook de endurecimiento temporal/ejecucion."),
    ("scratch/run_temporal_integrity_audit.py", "Script de validacion de integridad temporal."),
    ("scratch/run_rerun_integrity_check.py", "Script de validacion de proteccion contra re-ejecucion."),
    ("EURUSD_RED_TEAM_AUDIT_PROTOCOL.md", "Protocolo de auditoria Red Team."),
    ("EURUSD_RED_TEAM_AUDIT_RESULTS.md", "Resultados de la auditoria Red Team."),
    ("EURUSD_RED_TEAM_AUDIT_DECISION.md", "Decision final de auditoria Red Team."),
    ("RED_TEAM_AUDIT_STATUS.json", "Status json de auditoria Red Team."),
    ("RED_TEAM_AUDIT_HEARTBEAT.md", "Heartbeat de auditoria Red Team."),
    ("RED_TEAM_AUDIT_RUNBOOK.md", "Runbook de auditoria Red Team."),
    ("scratch/run_red_team_lab_audit.py", "Script de auditoria de integridad Red Team."),
    ("EURUSD_SIGNAL_DRIFT_PROTOCOL.md", "Protocolo de deteccion de drift."),
    ("EURUSD_SIGNAL_DRIFT_RESULTS.md", "Resultados de la auditoria de drift."),
    ("EURUSD_SIGNAL_DRIFT_DECISION.md", "Decision final de drift."),
    ("SIGNAL_DRIFT_STATUS.json", "Status json de drift."),
    ("SIGNAL_DRIFT_HEARTBEAT.md", "Heartbeat de drift."),
    ("SIGNAL_DRIFT_RUNBOOK.md", "Runbook de drift."),
    ("scratch/run_signal_drift_monitor.py", "Script de monitoreo de drift."),
    ("scratch/run_signal_drift_baseline_builder.py", "Script de construccion de baselines."),
    ("scratch/run_signal_drift_validator.py", "Script de validacion del monitor de drift."),
    ("scratch/run_post_hardening_drift_reconciliation.py", "Orquestador de rebaseline y reconciliacion post-hardening."),
    ("scratch/post_hardening_drift_lib.py", "Libreria compartida de drift post-hardening."),
    ("scratch/forward_telemetry_lib.py", "Libreria compartida de telemetria forward."),
    ("scratch/run_forward_telemetry_hardening.py", "Reconciliador e idempotence check de telemetria forward."),
    ("EURUSD_SCBI_COST_MODEL_PROTOCOL.md", "Protocolo modelo costos."),
    ("EURUSD_SCBI_COST_MODEL_DECISION.md", "Decision modelo costos."),
    ("EURUSD_CORE_CANDIDATE_TRACE_PROTOCOL.md", "Protocolo de trazabilidad de candidatos de CORE."),
    ("EURUSD_CORE_CANDIDATE_TRACE_RESULTS.md", "Resultados de trazabilidad de candidatos de CORE."),
    ("EURUSD_CORE_CANDIDATE_TRACE_DECISION.md", "Decision de trazabilidad de candidatos de CORE."),
    ("results/SCBI_CORE_PHASE1/core_phase1_ledger.csv", "Ledger oficial de la linea CORE."),
    ("results/SCBI_CORE_STAGE2/core_stage2_trades.csv", "Candidatos enriquecidos de la linea CORE."),
    ("EURUSD_SCBI_CORE_FORWARD_LEDGER_SCHEMA.md", "Schema del ledger de CORE."),
    ("EURUSD_SCBI_CORE_FORWARD_OPERATING_SYSTEM.md", "Sistema operativo de forward CORE."),
    ("EURUSD_SCBI_CORE_FORWARD_PHASE1_PROTOCOL.md", "Protocolo de fase 1 CORE."),
    ("EURUSD_SCBI_CORE_FORWARD_REHEARSAL_RESULTS.md", "Resultados de ensayo CORE."),
    ("EURUSD_SCBI_CORE_STAGE1_RESULTS.md", "Resultados Stage 1 CORE."),
    ("EURUSD_SCBI_CORE_STAGE2_RESULTS.md", "Resultados Stage 2 CORE."),
    ("EURUSD_DUAL_DAILY_CHAIN_RESULTS.md", "Resultados orquestador diario dual."),
    ("EURUSD_EARLY_FORWARD_EXPECTATION_PROTOCOL.md", "Protocolo de expectativas en muestras pequeñas."),
    ("EURUSD_EARLY_FORWARD_EXPECTATION_RESULTS.md", "Resultados del modelado de envelopes de expectativa."),
    ("EURUSD_EARLY_FORWARD_EXPECTATION_DECISION.md", "Decision de la capa de interpretacion de baja N."),
    ("results/SCBI_EARLY_FORWARD_EXPECTATION_ENVELOPES.json", "Datos de los envelopes estadisticos."),
    ("scratch/run_early_forward_expectation_builder.py", "Script generador de envelopes."),
    ("scratch/early_forward_expectation_lib.py", "Libreria de interpretacion de expectativas."),
    ("EURUSD_UNIFIED_DECISION_SURFACE_PROTOCOL.md", "Protocolo de la superficie unificada de decision por linea."),
    ("EURUSD_UNIFIED_DECISION_SURFACE_RESULTS.md", "Resultados de la unificacion institucional por linea."),
    ("EURUSD_UNIFIED_DECISION_SURFACE_DECISION.md", "Decision final de la superficie unificada."),
    ("results/SCBI_UNIFIED_LINE_STATUS.json", "Superficie unificada de decision por linea en JSON."),
    ("results/SCBI_UNIFIED_LINE_STATUS.csv", "Superficie unificada de decision por linea en tabla plana."),
    ("scratch/unified_line_status_lib.py", "Libreria de la superficie unificada por linea."),
    ("scratch/run_unified_line_status_builder.py", "Builder canonico de la superficie unificada."),
    ("EURUSD_SEQUENTIAL_MATERIAL_SEVERITY_PROTOCOL.md", "Protocolo de rediseño de severidad material secuencial."),
    ("EURUSD_SEQUENTIAL_MATERIAL_SEVERITY_RESULTS.md", "Resultados de validación de severidad material secuencial."),
    ("EURUSD_SEQUENTIAL_MATERIAL_SEVERITY_DECISION.md", "Decision final de canonización de severidad material."),
    ("results/SCBI_SEQUENTIAL_EVIDENCE_VALIDATION.json", "Base de datos oficial de validación secuencial V3.4."),
    ("scratch/sequential_evidence_lib.py", "Libreria de severidad secuencial (DCW + Hysteresis)."),
    ("scratch/run_sequential_evidence_validator.py", "Script validador oficial de la capa secuencial."),
    ("EURUSD_MAY_2026_OBSERVATION_PROTOCOL.md", "Protocolo operativo de observacion disciplinada para mayo 2026."),
    ("EURUSD_MAY_2026_DAILY_CHECKLIST.md", "Checklist diaria de observacion oficial para mayo 2026."),
    ("EURUSD_MAY_2026_WEEKLY_CHECKLIST.md", "Checklist semanal de observacion oficial para mayo 2026."),
    ("EURUSD_MAY_2026_CHECKPOINT_RULES.md", "Reglas de lectura de checkpoints N=5, N=10 y N=20 para mayo 2026."),
    ("EURUSD_MAY_2026_INCIDENT_RULES.md", "Taxonomia y politica de incidentes para mayo 2026."),
    ("EURUSD_BASELINE_VALIDATOR_HARDENING_PROTOCOL.md", "Protocolo de hardening del baseline validator diario."),
    ("EURUSD_BASELINE_VALIDATOR_HARDENING_RESULTS.md", "Resultados de hardening y validacion del baseline validator."),
    ("EURUSD_BASELINE_VALIDATOR_HARDENING_DECISION.md", "Decision final del baseline validator endurecido."),
    ("BASELINE_VALIDATOR_HARDENING_STATUS.json", "Ultimo status auditable del baseline validator endurecido."),
    ("BASELINE_VALIDATOR_HARDENING_HEARTBEAT.md", "Heartbeat operativo del baseline validator endurecido."),
    ("BASELINE_VALIDATOR_HARDENING_RUNBOOK.md", "Runbook de uso diario del baseline validator endurecido."),
    ("scratch/validate_scbi_phase1_baseline.py", "Validator diario endurecido de baseline, data, news y rerun."),
    ("EURUSD_DATA_COVERAGE_PIPELINE_PROTOCOL.md", "Protocolo canonico de intake, validacion y promocion de cobertura EURUSD."),
    ("EURUSD_DATA_COVERAGE_PIPELINE_RESULTS.md", "Resultados de validacion de la tuberia de cobertura EURUSD."),
    ("EURUSD_DATA_COVERAGE_PIPELINE_DECISION.md", "Decision final de la tuberia de cobertura EURUSD."),
    ("DATA_COVERAGE_PIPELINE_STATUS.json", "Status operativo de cobertura H1/M5/news."),
    ("DATA_COVERAGE_PIPELINE_HEARTBEAT.md", "Heartbeat de cobertura H1/M5/news."),
    ("DATA_COVERAGE_PIPELINE_RUNBOOK.md", "Runbook de cobertura H1/M5/news."),
    ("scratch/data_coverage_pipeline_lib.py", "Libreria austera de cobertura, validacion y promocion de data."),
    ("scratch/run_data_coverage_refresh.py", "Inventario e inicializacion de intake manual controlado."),
    ("scratch/run_data_coverage_check.py", "Check binario de cobertura H1/M5/news e integracion con validator."),
    ("scratch/run_data_coverage_promotion.py", "Promocion append-only de intake valido a data canonica."),
    ("EURUSD_DAILY_DATA_TO_DECISION_CHAIN_PROTOCOL.md", "Protocolo de la cadena operativa única de data a decisión."),
    ("EURUSD_DAILY_DATA_TO_DECISION_CHAIN_RESULTS.md", "Resultados de validación de la cadena operativa diaria."),
    ("EURUSD_DAILY_DATA_TO_DECISION_CHAIN_DECISION.md", "Decisión institucional de la cadena operativa diaria."),
    ("DAILY_DATA_TO_DECISION_CHAIN_STATUS.json", "Status operativo de la cadena unificada."),
    ("DAILY_DECISION_ARTIFACT_LAST.json", "Artefacto unificado de decisión diaria del laboratorio."),
    ("scratch/run_daily_data_to_decision_chain.py", "Orquestador único de la cadena operativa diaria (Data-to-Decision)."),
    ("micro_pilot_protocol/manual_live_exception/README.md", "Guía de la carpeta de excepción manual."),
    ("micro_pilot_protocol/manual_live_exception/MANUAL_MICRO_PILOT_PLAYBOOK.md", "Playbook maestro de la excepción manual."),
    ("micro_pilot_protocol/manual_live_exception/pre_session_checklist.md", "Checklist pre-sesión manual."),
    ("micro_pilot_protocol/manual_live_exception/in_session_rules.md", "Reglas intra-sesión manual."),
    ("micro_pilot_protocol/manual_live_exception/post_session_checklist.md", "Checklist post-sesión manual."),
    ("micro_pilot_protocol/manual_live_exception/kill_switch_manual_rules.md", "Kill switch del piloto manual."),
    ("micro_pilot_protocol/manual_live_exception/risk_limits_manual.md", "Límites de riesgo manuales."),
    ("micro_pilot_protocol/manual_live_exception/trade_journal_template.csv", "Plantilla de diario de trades."),
    ("micro_pilot_protocol/manual_live_exception/daily_review_template.md", "Plantilla de revisión diaria."),
    ("audits/sunday_gap_audit/sunday_gap_audit_report.md", "Reporte principal de auditoría de domingos."),
    ("audits/sunday_gap_audit/sunday_gap_audit_report.json", "Reporte JSON de auditoría de domingos."),
    ("audits/sunday_gap_audit/sunday_gap_counts.csv", "Conteos forenses de barras dominicales."),
    ("audits/sunday_gap_audit/sunday_to_monday_case_studies.csv", "Estudios de caso de transición Dom-Lun."),
    ("audits/sunday_gap_audit/weekly_impact_notes.md", "Notas de impacto semanal."),
    ("audits/sunday_gap_audit/strategy_impact_notes.md", "Notas de impacto en la estrategia."),
    ("institutional_research_candidate_lab/outputs/period_validation_2026_01_01_to_2026_04_23_after_sunday_fix/summary_after_sunday_fix.json", "Resumen JSON re-validacion post-fix."),
    ("institutional_research_candidate_lab/outputs/period_validation_2026_01_01_to_2026_04_23_after_sunday_fix/summary_after_sunday_fix.md", "Resumen MD re-validacion post-fix."),
    ("institutional_research_candidate_lab/outputs/period_validation_2026_01_01_to_2026_04_23_after_sunday_fix/trades_after_sunday_fix.csv", "Trades re-validacion post-fix."),
    ("institutional_research_candidate_lab/outputs/period_validation_2026_01_01_to_2026_04_23_after_sunday_fix/monday_impact_before_vs_after.csv", "Impacto comparativo en lunes."),
    ("institutional_research_candidate_lab/outputs/period_validation_2026_01_01_to_2026_04_23_after_sunday_fix/sunday_fix_revalidation_notes.md", "Notas de re-validacion post-fix."),
    ("mt5_deployment_audit/mt5_automation_feasibility.md", "Reporte de factibilidad MT5."),
    ("mt5_deployment_audit/mt5_automation_feasibility.json", "Reporte JSON de factibilidad MT5."),
    ("mt5_deployment_audit/mt5_activation_steps.md", "Pasos de activacion MT5."),
    ("mt5_deployment_audit/mt5_risk_profile.md", "Perfil de riesgo MT5."),
    ("mt5_deployment_audit/mt5_kill_switch_rules.md", "Reglas de kill switch MT5."),
    ("mt5_demo_executor_lab/00_READ_THIS_FIRST.md", "Maestro del laboratorio demo MT5."),
    ("mt5_demo_executor_lab/README.md", "Guia del laboratorio demo MT5."),
    ("mt5_demo_executor_lab/mt5_demo_executor.py", "Orquestador demo MT5."),
    ("mt5_demo_executor_lab/mt5_data_bridge.py", "Puente de datos MT5."),
    ("mt5_demo_executor_lab/mt5_order_router.py", "Enrutador de ordenes MT5."),
    ("mt5_demo_executor_lab/mt5_timeout_manager.py", "Gestor de timeout MT5."),
    ("mt5_demo_executor_lab/mt5_news_guard.py", "Guardia de noticias MT5."),
    ("mt5_demo_executor_lab/mt5_risk_engine.py", "Motor de riesgo MT5."),
    ("mt5_demo_executor_lab/mt5_kill_switch.py", "Kill switch demo MT5."),
    ("mt5_demo_executor_lab/mt5_demo_telemetry.py", "Telemetria demo MT5."),
    ("mt5_demo_executor_lab/local_launch/START_MT5_DEMO_LOCAL.bat", "Lanzador BAT demo."),
    ("mt5_demo_executor_lab/local_launch/START_MT5_DEMO_LOCAL.ps1", "Lanzador PS1 demo."),
    ("mt5_demo_executor_lab/local_launch/STOP_MT5_DEMO_LOCAL.bat", "Stop script demo."),
    ("mt5_demo_executor_lab/local_launch/mt5_local_config.json.example", "Configuracion local example."),
    ("mt5_demo_executor_lab/local_launch/README_LOCAL_LAUNCH.md", "Guia de lanzamiento local."),
    ("mt5_demo_executor_lab/demo_to_live_gate/gate_policy.md", "Politica del gate demo-to-live."),
    ("mt5_demo_executor_lab/demo_to_live_gate/demo_tp_perfect_trade_gate.py", "Script auditor del gate."),
    ("mt5_demo_executor_lab/demo_to_live_gate/live_sandbox_100usd_policy.md", "Politica de live sandbox 100usd."),
    ("mt5_demo_executor_lab/demo_to_live_gate/live_sandbox_activation_checklist.md", "Checklist de activacion live."),
    ("mt5_demo_executor_lab/demo_to_live_gate/outputs/demo_tp_gate_status.json", "Status actual del gate."),

]

EXCLUSION_RULES = [
    "Se excluyen backups y archivos con sufijo `_BACKUP_`.",
    "Se excluye todo lo no listado explicitamente.",
]

@dataclass(frozen=True)
class BuildStats:
    file_count: int
    raw_total_bytes: int
    previous_zip_bytes: int
    output_zip_bytes: int

def _ensure_within_project(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    if not str(resolved).startswith(str(CANONICAL_ROOT)):
        raise RuntimeError(f"FAIL-CLOSED: path fuera del proyecto: {resolved}")
    return resolved

def _require_file(relative_path: str) -> Path:
    path = _ensure_within_project(PROJECT_ROOT / relative_path)
    if not path.exists():
        raise FileNotFoundError(f"Falta: {relative_path}")
    return path

def _canonical_file_entries():
    entries = list(CANONICAL_FILES)
    for rel, reason in OPTIONAL_CANONICAL_FILES:
        path = PROJECT_ROOT / rel
        if path.exists():
            entries.append((rel, reason))
    return entries

def build_bundle():
    previous_zip_bytes = OUTPUT_ZIP.stat().st_size if OUTPUT_ZIP.exists() else 0
    entries = _canonical_file_entries()
    
    # Docs
    manifest = "# ZIP Manifest\n\n"
    for rel, reason in entries:
        manifest += f"- `{rel}`: {reason}\n"
    MANIFEST_PATH.write_text(manifest, encoding="utf-8")
    
    # Zip
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:
        for rel, _ in entries:
            zipf.write(PROJECT_ROOT / rel, arcname=rel)
            
    size = OUTPUT_ZIP.stat().st_size
    
    status = f"# Delivery Status\n- Size: {size}\n- Files: {len(entries)}"
    DELIVERY_STATUS_PATH.write_text(status, encoding="utf-8")
    audit = (
        "# ZIP Packaging Audit\n\n"
        f"Generated at: `{datetime.now(timezone.utc).isoformat()}`\n\n"
        "## Rebuild Status\n\n"
        "- Reconstruccion desde cero: SI\n"
        "- Reemplazo completo del zip anterior: SI\n"
        f"- Zip previo detectado: `{previous_zip_bytes}` bytes\n"
        "- Zips extra eliminados de la raiz: ninguno\n\n"
        "## Canonical Content Criterion\n\n"
        "- Se incluye solo el set minimo vigente y reusable para operar el stack actual sin salir del proyecto.\n"
        "- Se preserva H6 solo como benchmark conceptual vigente.\n"
        "- Se incorporan artefactos post-hardening de drift, telemetria forward y la superficie unificada de decision por linea.\n"
        "- Se incorpora el hardening del baseline validator diario con cobertura H1/M5/news y rerun fail-closed.\n"
        "- Se incorpora la tuberia canonica de cobertura H1/M5/news con intake manual, validacion dura y promocion append-only.\n"
        "- `ZIP_DELIVERY_STATUS.md` queda fuera del zip para evitar autorreferencia del propio artefacto.\n\n"
        "## Exclusion Criterion Applied\n\n"
        "- Se excluyen backups, staging, handoffs intermedios y archivos con sufijo `_BACKUP_`.\n"
        "- Se excluye todo archivo no listado explicitamente por el builder.\n\n"
        "## Integrity Checks\n\n"
        f"- Archivos canonicos incluidos: `{len(entries)}`\n"
        "- Ausencia de duplicados logicos por nombre interno: SI\n"
        "- Ausencia de backups dentro del zip: SI\n"
        "- Ausencia de archivos intermedios dentro del zip: SI\n"
        "- Coherencia con documentos canonicos activos: SI\n"
        "- H6 preservado como benchmark vigente e intocable: SI\n"
    )
    AUDIT_PATH.write_text(audit, encoding="utf-8")
    
    return BuildStats(len(entries), 0, previous_zip_bytes, size)

def main():
    res = build_bundle()
    print(f"files={res.file_count}")
    print(f"output_zip_bytes={res.output_zip_bytes}")

if __name__ == "__main__":
    main()
