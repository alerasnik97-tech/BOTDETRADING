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
    ("results/SCBI_FORWARD_LEDGER.csv", "Ledger oficial de la linea GLOBAL."),
    ("EURUSD_SCBI_2020_2025_DURABILITY_PROTOCOL.md", "Protocolo de durabilidad."),
    ("EURUSD_SCBI_2020_2025_DURABILITY_RESULTS.md", "Resultados de durabilidad."),
    ("EURUSD_SCBI_2020_2025_DURABILITY_DECISION.md", "Decision de durabilidad."),
    ("EURUSD_SCBI_CORE_BRANCH_PROTOCOL.md", "Protocolo de rama CORE."),
    ("EURUSD_SCBI_CORE_BRANCH_RESULTS.md", "Resultados de rama CORE."),
    ("EURUSD_SCBI_CORE_BRANCH_DECISION.md", "Decision de rama CORE."),
    ("EURUSD_SCBI_CORE_FORWARD_PROTOCOL.md", "Protocolo de forward CORE."),
    ("EURUSD_DUAL_LINE_FORWARD_PROTOCOL.md", "Protocolo dual."),
    ("results/SCBI_DUAL_LINE_SCOREBOARD.csv", "Scoreboard dual."),
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
    ("scratch/prop_firm_risk_guards.py", "Script de guards de riesgo institucional."),
    ("EURUSD_FORWARD_TRIBUNAL_PROTOCOL.md", "Protocolo del tribunal de evidencia."),
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
    ("OPERATIONAL_STACK_REPLAY_HEARTBEAT.md", "Heartbeat del replay."),
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
    ("EURUSD_SCBI_COST_MODEL_PROTOCOL.md", "Protocolo modelo costos."),
    ("EURUSD_SCBI_COST_MODEL_DECISION.md", "Decision modelo costos."),
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
    
    return BuildStats(len(entries), 0, previous_zip_bytes, size)

def main():
    res = build_bundle()
    print(f"files={res.file_count}")
    print(f"output_zip_bytes={res.output_zip_bytes}")

if __name__ == "__main__":
    main()
