from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


CANONICAL_ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo").resolve()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ZIP = PROJECT_ROOT / "000_PARA_CHATGPT.zip"
MANIFEST_PATH = PROJECT_ROOT / "ZIP_CONTENTS_MANIFEST.md"
AUDIT_PATH = PROJECT_ROOT / "ZIP_PACKAGING_AUDIT.md"
DELIVERY_STATUS_PATH = PROJECT_ROOT / "ZIP_DELIVERY_STATUS.md"
BUILDER_STAGE_DIR = PROJECT_ROOT / "scripts" / ".bundle_build_tmp"
LEGACY_STAGE_DIR = PROJECT_ROOT / "__zip_stage"


CANONICAL_FILES: list[tuple[str, str]] = [
    ("CURRENT_STATE_OF_LAB.md", "Estado operativo vigente del laboratorio (STANDBY)."),
    ("EURUSD_H6_SURVIVAL_PROFILE_AND_FAILURE_DELTA.md", "Autopsia comparativa: Supervivencia de H6 vs Fracaso de lineas."),
    ("H6_INDUSTRIAL_DESIGN.md", "Blueprint y diseño industrial del benchmark H6."),
    ("MAPA_ESTRATEGIAS.md", "Mapa institucional de estrategias y resultados acumulados."),
    ("EURUSD_DESIGN_CONSTRAINT_BRIEF.md", "Marco de restricciones y conocimiento negativo."),
    ("EURUSD_MANUAL_EDGE_FINAL_DECISION.md", "Veredicto final de la linea manual-edge."),
    ("CAMPAIGN_DECISION_C4.md", "Reporte de decision y veredicto de la Campaña 4."),
    ("EURUSD_CAMPAIGN_3B_FINAL_DECISION.md", "Reporte de decision y veredicto de la Campaña 3B."),
    ("fast_signal_execution_dashboard.md", "Dashboard resumen del cierre operativo manual."),
    ("CODEX_LOCAL_SAFETY_PROTOCOL.md", "Protocolo de seguridad y perímetro local."),
    ("CODEX_PRE_FLIGHT_CHECKLIST.md", "Checklist de validacion pre-ejecucion."),
    ("CAMPAIGN_GATEKEEPER_PROTOCOL.md", "Protocolo de gobernanza y gatekeeper de campañas."),
    ("RESEARCH_DECISION_MATRIX.md", "Matriz de decision estandarizada del laboratorio."),
    ("EURUSD_HYPOTHESIS_ADMISSION_RULES.md", "Reglas maestras de admision y bloqueo de hipotesis."),
    ("EURUSD_HYPOTHESIS_SCORECARD_TEMPLATE.md", "Plantilla de evaluacion de hipotesis para pre-intake."),
    ("EURUSD_HYPOTHESIS_ADMISSION_EXAMPLES.md", "Ejemplos de hipotesis admisibles vs bloqueadas."),
    ("scripts/hypothesis_admission_check.py", "Script motor de admision y calculo de scoring."),
    ("EURUSD_HYPOTHESIS_INPUT_SCHEMA.md", "Contrato estricto y schema JSON para el motor de admision."),
    ("EURUSD_HYPOTHESIS_ADMISSION_VALIDATION.md", "Certificación de robustez del motor contra fallos historicos."),
    ("EURUSD_HYPOTHESIS_VIABILITY_RULES.md", "Reglas maestras de viabilidad operativa (Stage-0)."),
    ("EURUSD_HYPOTHESIS_VIABILITY_SCORECARD_TEMPLATE.md", "Plantilla de evaluacion de viabilidad para pre-campaign."),
    ("scripts/hypothesis_viability_check.py", "Script motor de verificacion de viabilidad."),
    ("EURUSD_LTF_OBJECTIVE_ENTRY_SPEC.md", "Especificacion mecanica de gatillos LTF objetivos."),
    ("EURUSD_STAGE1_LTF_COMPARISON_RESULTS.md", "Resultados de la prueba Stage-1 para gatillos LTF."),
    ("EURUSD_LTF_ENTRY_FINAL_VERDICT.md", "Veredicto final del Stage-1 y recomendacion de escalado."),
    ("ZIP_CONTENTS_MANIFEST.md", "Manifiesto de contenido del bundle maestro."),
    ("ZIP_PACKAGING_AUDIT.md", "Auditoria de empaquetado y criterios de exclusion."),
]

OPTIONAL_CANONICAL_FILES: list[tuple[str, str]] = [
    ("EURUSD_ECB_AUTOPILOT_STATUS.json", "Estado persistente del autopilot ECB."),
    ("EURUSD_ECB_AUTOPILOT_HEARTBEAT.md", "Heartbeat legible del autopilot ECB."),
    ("EURUSD_ECB_AUTOPILOT_RUNBOOK.md", "Runbook austero de la corrida ECB."),
    ("EURUSD_ECB_STAGE2_DECISION.md", "Decision formal de Stage-2 para ECB."),
    ("EURUSD_ECB_FINAL_DECISION.md", "Decision final unica de la corrida ECB."),
    ("EURUSD_ECB_FULL_CAMPAIGN_OOS_FINAL.md", "Documento OOS final si ECB promueve a Full Campaign."),
    ("results/eurusd_ltf_objective_entry_replacement_ecb_autopilot/precheck_audit.json", "Auditoria fisica de precheck de la corrida ECB."),
    ("results/eurusd_ltf_objective_entry_replacement_ecb_autopilot/stage2/summary.json", "Resumen consolidado de resultados Stage-2."),
    ("results/eurusd_ltf_objective_entry_replacement_ecb_autopilot/stage2/trades.csv", "Trades consolidados Stage-2."),
    ("results/eurusd_ltf_objective_entry_replacement_ecb_autopilot/stage2/optimization_results.csv", "Tabla de bloques ejecutados en Stage-2."),
    ("results/eurusd_ltf_objective_entry_replacement_ecb_autopilot/stage2/stage2_gate_evaluation.json", "Evaluacion cuantitativa de gates Stage-2."),
    ("results/eurusd_ltf_objective_entry_replacement_ecb_autopilot/full_campaign/summary.json", "Resumen consolidado de Full Campaign."),
    ("results/eurusd_ltf_objective_entry_replacement_ecb_autopilot/full_campaign/trades.csv", "Trades consolidados de Full Campaign."),
    ("results/eurusd_ltf_objective_entry_replacement_ecb_autopilot/full_campaign/optimization_results.csv", "Tabla de periodos de Full Campaign."),
    ("results/eurusd_ltf_objective_entry_replacement_ecb_autopilot/full_campaign/period_summaries.json", "Resumen por periodos dev/val/holdout/full."),
    ("results/eurusd_ltf_objective_entry_replacement_ecb_autopilot/full_campaign/stress_summary.json", "Stress summary conservador de Full Campaign."),
    ("EURUSD_NEXT_HYPOTHESIS_SHORTLIST_MATRIX.md", "Matriz de candidatos de la discovery sprint."),
    ("EURUSD_NEXT_HYPOTHESIS_DISCOVERY_REPORT.md", "Reporte de descubrimiento de la proxima hipotesis."),
    ("EURUSD_NEXT_HYPOTHESIS_FINAL_DECISION.md", "Decision final unica de descubrimiento."),
    ("EURUSD_NEXT_HYPOTHESIS_INTAKE.md", "Intake canonico de la candidata seleccionada."),
    ("EURUSD_NEXT_HYPOTHESIS_SPEC.md", "Especificacion tecnica de la candidata seleccionada."),
    ("EURUSD_NEXT_HYPOTHESIS_STAGE1_PLAN.md", "Plan de validacion Stage-1 para la candidata."),
    ("EURUSD_H6_LTF_STAGE1_RESULTS.md", "Resultados del test Stage 1 de la hipotesis."),
    ("EURUSD_H6_LTF_STAGE1_DECISION.md", "Decision final unica del Stage 1 de la hipotesis."),
    ("H6_LTF_STAGE1_STATUS.json", "Estado de corrida del Stage 1."),
    ("H6_LTF_STAGE1_HEARTBEAT.md", "Heartbeat de corrida del Stage 1."),
    ("H6_LTF_STAGE1_RUNBOOK.md", "Runbook de corrida del Stage 1."),
    ("EURUSD_LTF_ENTRY_PROGRAM_DISCOVERY_REPORT.md", "Reporte de descubrimiento LTF Entry."),
    ("EURUSD_LTF_ENTRY_PROGRAM_SHORTLIST_MATRIX.md", "Matriz de candidatos LTF Entry."),
    ("EURUSD_LTF_ENTRY_PROGRAM_FINAL_DECISION.md", "Decision final LTF Entry."),
    ("EURUSD_LTF_ENTRY_PROGRAM_SPEC.md", "Spec de la arquitectura LTF Entry."),
    ("EURUSD_LTF_ENTRY_PROGRAM_STAGE2_PLAN.md", "Plan Stage 2 de la arquitectura LTF Entry."),
    ("LTF_ENTRY_PROGRAM_STATUS.json", "Estado de corrida LTF Entry."),
    ("LTF_ENTRY_PROGRAM_HEARTBEAT.md", "Heartbeat de corrida LTF Entry."),
    ("LTF_ENTRY_PROGRAM_RUNBOOK.md", "Runbook de corrida LTF Entry."),
    ("EURUSD_REJECTION_WICK_M5_STAGE1_RESULTS.md", "Resultados de Stage 1 de REJECTION WICK M5."),
    ("EURUSD_REJECTION_WICK_M5_STAGE1_DECISION.md", "Decisión final de Stage 1 de REJECTION WICK M5."),
    ("REJECTION_WICK_M5_STAGE1_STATUS.json", "Status de Stage 1 de REJECTION WICK M5."),
    ("REJECTION_WICK_M5_STAGE1_HEARTBEAT.md", "Heartbeat de Stage 1 de REJECTION WICK M5."),
    ("REJECTION_WICK_M5_STAGE1_RUNBOOK.md", "Runbook de Stage 1 de REJECTION WICK M5."),
    ("EURUSD_REJECTION_WICK_M5_STAGE2_RESULTS.md", "Resultados de Stage 2 de REJECTION WICK M5."),
    ("EURUSD_REJECTION_WICK_M5_STAGE2_DECISION.md", "Decisión final de Stage 2 de REJECTION WICK M5."),
    ("REJECTION_WICK_M5_STAGE2_STATUS.json", "Status de Stage 2 de REJECTION WICK M5."),
    ("REJECTION_WICK_M5_STAGE2_HEARTBEAT.md", "Heartbeat de Stage 2 de REJECTION WICK M5."),
    ("REJECTION_WICK_M5_STAGE2_RUNBOOK.md", "Runbook de Stage 2 de REJECTION WICK M5."),
    ("HTF_SWEEP_QUALITY_STATUS.json", "Estado de la Discovery Sprint HTF."),
    ("HTF_SWEEP_QUALITY_HEARTBEAT.md", "Heartbeat de la Discovery Sprint HTF."),
    ("HTF_SWEEP_QUALITY_RUNBOOK.md", "Runbook de la Discovery Sprint HTF."),
    ("EURUSD_HTF_SWEEP_QUALITY_SHORTLIST_MATRIX.md", "Matriz de candidatos HTF."),
    ("EURUSD_HTF_SWEEP_QUALITY_DISCOVERY_REPORT.md", "Reporte de la Discovery Sprint HTF."),
    ("EURUSD_HTF_SWEEP_QUALITY_FINAL_DECISION.md", "Decisión final de la Discovery Sprint HTF."),
    ("EURUSD_HTF_SWEEP_QUALITY_INTAKE.md", "Intake de la candidata HTF sobreviviente."),
    ("EURUSD_HTF_SWEEP_QUALITY_SPEC.md", "Spec de la candidata HTF sobreviviente."),
    ("EURUSD_HTF_SWEEP_QUALITY_STAGE1_PLAN.md", "Plan de Stage 1 de la candidata HTF sobreviviente."),
    ("HTF_FILTER_VALUE_TEST_STATUS.json", "Estado del A/B Test."),
    ("HTF_FILTER_VALUE_TEST_HEARTBEAT.md", "Heartbeat del A/B Test."),
    ("HTF_FILTER_VALUE_TEST_RUNBOOK.md", "Runbook del A/B Test."),
    ("EURUSD_HTF_FILTER_VALUE_TEST_RESULTS.md", "Resultados empíricos del A/B Test."),
    ("EURUSD_HTF_FILTER_VALUE_TEST_DECISION.md", "Decisión final del A/B Test."),
    ("EURUSD_HTF_NY_WINDOW_ECB_STAGE1_RESULTS.md", "Resultados Stage 1 Combinada."),
    ("EURUSD_HTF_NY_WINDOW_ECB_STAGE1_DECISION.md", "Decision Stage 1 Combinada."),
    ("EURUSD_HTF_NY_WINDOW_ECB_STAGE2_RESULTS.md", "Resultados Stage 2 Combinada."),
    ("EURUSD_HTF_NY_WINDOW_ECB_STAGE2_DECISION.md", "Decision final Stage 2 Combinada."),
    ("HTF_NY_WINDOW_ECB_STATUS.json", "Status Validacion Combinada."),
    ("HTF_NY_WINDOW_ECB_HEARTBEAT.md", "Heartbeat Validacion Combinada."),
    ("HTF_NY_WINDOW_ECB_RUNBOOK.md", "Runbook Validacion Combinada."),
    ("HTF_LTF_MONETIZATION_STATUS.json", "Status Monetizacion LTF."),
    ("HTF_LTF_MONETIZATION_HEARTBEAT.md", "Heartbeat Monetizacion LTF."),
    ("HTF_LTF_MONETIZATION_RUNBOOK.md", "Runbook Monetizacion LTF."),
    ("EURUSD_HTF_LTF_MONETIZATION_REPORT.md", "Reporte de Monetizacion LTF."),
    ("EURUSD_HTF_LTF_MONETIZATION_SHORTLIST_MATRIX.md", "Matriz de Monetizacion LTF."),
    ("EURUSD_HTF_LTF_MONETIZATION_FINAL_DECISION.md", "Decision Final Monetizacion LTF."),
    ("EURUSD_HTF_LTF_MONETIZATION_SPEC.md", "Spec de SCBI_M5."),
    ("EURUSD_HTF_LTF_MONETIZATION_STAGE2_PLAN.md", "Plan Stage-2 SCBI_M5."),
    ("HTF_NY_WINDOW_SCBI_STAGE2_STATUS.json", "Status Stage-2 real SCBI."),
    ("HTF_NY_WINDOW_SCBI_STAGE2_HEARTBEAT.md", "Heartbeat Stage-2 real SCBI."),
    ("HTF_NY_WINDOW_SCBI_STAGE2_RUNBOOK.md", "Runbook Stage-2 real SCBI."),
    ("EURUSD_HTF_NY_WINDOW_SCBI_STAGE2_RESULTS.md", "Resultados Stage-2 real SCBI (data OHLCV)."),
    ("EURUSD_HTF_NY_WINDOW_SCBI_STAGE2_DECISION.md", "Decision final Stage-2 real SCBI."),
    ("EURUSD_RESEARCH_INTEGRITY_AUDIT.md", "Auditoria forense de integridad del laboratorio."),
    ("EURUSD_CANONICAL_BACKTEST_PIPELINE.md", "Pipeline canonica unica de backtest."),
    ("EURUSD_REVALIDATION_PRIORITY.md", "Lista priorizada de revalidaciones."),
    ("RESEARCH_INTEGRITY_STATUS.json", "Status del integrity reset."),
    ("RESEARCH_INTEGRITY_HEARTBEAT.md", "Heartbeat del integrity reset."),
    ("RESEARCH_INTEGRITY_RUNBOOK.md", "Runbook del integrity reset."),
    ("EURUSD_REAL_HTF_FILTER_AB_PROTOCOL.md", "Protocolo ex-ante A/B test real."),
    ("EURUSD_REAL_HTF_FILTER_AB_RESULTS.md", "Resultados A/B test real HTF filter."),
    ("EURUSD_REAL_HTF_FILTER_AB_DECISION.md", "Decision final A/B test real HTF filter."),
    ("REAL_HTF_FILTER_AB_STATUS.json", "Status A/B test real."),
    ("REAL_HTF_FILTER_AB_HEARTBEAT.md", "Heartbeat A/B test real."),
    ("EURUSD_SCBI_GLOBAL_VALIDATION_PROTOCOL.md", "Protocolo ex-ante validacion SCBI global."),
    ("EURUSD_SCBI_GLOBAL_VALIDATION_RESULTS.md", "Resultados validacion SCBI global."),
    ("EURUSD_SCBI_GLOBAL_VALIDATION_DECISION.md", "Decision validacion SCBI global."),
    ("SCBI_GLOBAL_VALIDATION_STATUS.json", "Status validacion SCBI global."),
    ("SCBI_GLOBAL_VALIDATION_HEARTBEAT.md", "Heartbeat validacion SCBI global."),
    ("EURUSD_SCBI_FULL_CAMPAIGN_PROTOCOL.md", "Protocolo ex-ante Full Campaign SCBI."),
    ("EURUSD_SCBI_FULL_CAMPAIGN_RESULTS.md", "Resultados Full Campaign SCBI."),
    ("EURUSD_SCBI_FULL_CAMPAIGN_DECISION.md", "Decision Full Campaign SCBI."),
    ("EURUSD_SCBI_FORWARD_TEST_POLICY.md", "Politica de forward test SCBI."),
    ("SCBI_FULL_CAMPAIGN_STATUS.json", "Status Full Campaign SCBI."),
    ("SCBI_FULL_CAMPAIGN_HEARTBEAT.md", "Heartbeat Full Campaign SCBI."),
    ("EURUSD_SCBI_FORWARD_OPERATING_SYSTEM.md", "Sistema operativo del Forward Test SCBI."),
    ("EURUSD_SCBI_FORWARD_LEDGER_SCHEMA.md", "Esquema del ledger para el Forward Test SCBI."),
    ("EURUSD_SCBI_FORWARD_PHASE1_PROTOCOL.md", "Protocolo de Fase 1 para el Forward Test SCBI."),
    ("EURUSD_SCBI_FORWARD_DAILY_RUNBOOK.md", "Runbook diario para el Forward Test SCBI."),
    ("EURUSD_SCBI_FORWARD_INCIDENT_POLICY.md", "Politica de incidentes para el Forward Test SCBI."),
    ("SCBI_FORWARD_SYSTEM_STATUS.json", "Status del sistema operativo del Forward Test SCBI."),
    ("SCBI_FORWARD_SYSTEM_HEARTBEAT.md", "Heartbeat del sistema operativo del Forward Test SCBI."),
    ("EURUSD_SCBI_FORWARD_AUTOMATION_ARCHITECTURE.md", "Arquitectura de la automatizacion del forward test."),
    ("EURUSD_SCBI_FORWARD_AUTOMATION_PROTOCOL.md", "Protocolo operativo de la automatizacion."),
    ("SCBI_FORWARD_AUTOMATION_STATUS.json", "Status de la automatizacion."),
    ("SCBI_FORWARD_AUTOMATION_HEARTBEAT.md", "Heartbeat de la automatizacion."),
    ("EURUSD_SCBI_FORWARD_LAUNCH_CHECKLIST.md", "Checklist ex-ante para el launch."),
    ("EURUSD_SCBI_FORWARD_REHEARSAL_RESULTS.md", "Resultados del rehearsal del forward test."),
    ("EURUSD_SCBI_FORWARD_LAUNCH_DECISION.md", "Decision final del launch del forward test."),
    ("SCBI_FORWARD_LAUNCH_STATUS.json", "Status del launch."),
    ("SCBI_FORWARD_LAUNCH_HEARTBEAT.md", "Heartbeat del launch."),
    ("EURUSD_SCBI_PHASE1_GOVERNANCE_PROTOCOL.md", "Protocolo maestro de gobernanza de Phase 1."),
    ("EURUSD_SCBI_PHASE1_DRIFT_MONITORING.md", "Protocolo de monitoreo de drift para Phase 1."),
    ("EURUSD_SCBI_PHASE1_WEEKLY_REVIEW_PROTOCOL.md", "Protocolo de revision semanal para Phase 1."),
    ("EURUSD_SCBI_PHASE1_INCIDENT_SEVERITY_MODEL.md", "Modelo de severidad de incidentes para Phase 1."),
    ("SCBI_PHASE1_GOVERNANCE_STATUS.json", "Status de gobernanza de Phase 1."),
    ("SCBI_PHASE1_GOVERNANCE_HEARTBEAT.md", "Heartbeat de gobernanza de Phase 1."),
    ("EURUSD_SCBI_PHASE1_EVIDENCE_FREEZE_PROTOCOL.md", "Protocolo maestro de freeze formal de evidencia Phase 1."),
    ("EURUSD_SCBI_PHASE1_CHANGE_CONTROL.md", "Taxonomia y reglas de change-control para proteger comparabilidad."),
    ("EURUSD_SCBI_PHASE1_BASELINE_SNAPSHOT.md", "Snapshot legible de la baseline oficial congelada."),
    ("EURUSD_SCBI_PHASE1_EXCEPTION_LOG.md", "Registro formal de excepciones y cambios gobernados."),
    ("SCBI_PHASE1_FREEZE_STATUS.json", "Status maquina-legible del freeze Phase 1."),
    ("SCBI_PHASE1_FREEZE_HEARTBEAT.md", "Heartbeat operativo del freeze Phase 1."),
    ("SCBI_PHASE1_FREEZE_RUNBOOK.md", "Runbook austero del freeze y pre-run integrity."),
    ("scratch/validate_scbi_phase1_baseline.py", "Validador y sellador de baseline Phase 1."),
    ("results/SCBI_FORWARD_LEDGER.csv", "Ledger runtime oficial de Phase 1."),
    ("results/SCBI_FORWARD_DAILY_STATUS.csv", "Daily status runtime oficial de Phase 1."),
    ("results/SCBI_PHASE1_WEEKLY_REVIEW.csv", "CSV append-only de weekly review Phase 1."),
]


EXCLUSION_RULES = [
    "Se excluyen backups, archivos intermedios, staging y cualquier archivo con sufijo `_BACKUP_`.",
    "Se excluyen zips obsoletos de raiz distintos de `000_PARA_CHATGPT.zip`.",
    "Se excluyen scripts, codigo, datasets y resultados historicos que no son necesarios para entender el estado vigente y operar seguro dentro del proyecto.",
    "Se excluyen artefactos de etapas intermedias o handoffs ya superados.",
    "Se excluye todo archivo no nombrado explicitamente en la lista canonica.",
]


@dataclass(frozen=True)
class BuildStats:
    file_count: int
    raw_total_bytes: int
    previous_zip_bytes: int
    output_zip_bytes: int
    root_zip_count_before: int
    root_zip_count_after: int
    removed_root_zips: tuple[str, ...]


def _fail(message: str) -> None:
    raise RuntimeError(f"FAIL-CLOSED: {message}")


def _ensure_canonical_root() -> None:
    if PROJECT_ROOT.resolve() != CANONICAL_ROOT:
        _fail(f"root no canonico: {PROJECT_ROOT.resolve()}")
    if not CANONICAL_ROOT.exists() or not CANONICAL_ROOT.is_dir():
        _fail(f"root canonico inexistente: {CANONICAL_ROOT}")


def _ensure_within_project(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(CANONICAL_ROOT)
    except ValueError as exc:
        raise RuntimeError(f"FAIL-CLOSED: path fuera del proyecto: {resolved}") from exc
    return resolved


def _require_file(relative_path: str) -> Path:
    if "_BACKUP_" in relative_path:
        _fail(f"archivo backup no permitido en el bundle: {relative_path}")
    path = _ensure_within_project(PROJECT_ROOT / relative_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Falta archivo requerido para el bundle: {relative_path}")
    return path.resolve()


def _canonical_relative_paths() -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for relative_path, _ in _canonical_file_entries():
        if relative_path in seen:
            _fail(f"duplicado logico en whitelist: {relative_path}")
        seen.add(relative_path)
        ordered.append(relative_path)
    return ordered


def _bundle_files() -> list[Path]:
    return [_require_file(relative_path) for relative_path in _canonical_relative_paths()]


def _canonical_file_entries() -> list[tuple[str, str]]:
    entries = list(CANONICAL_FILES)
    for relative_path, reason in OPTIONAL_CANONICAL_FILES:
        candidate = _ensure_within_project(PROJECT_ROOT / relative_path)
        if candidate.exists() and candidate.is_file():
            entries.append((relative_path, reason))
    return entries


def _root_level_extra_zips() -> list[Path]:
    return sorted(
        _ensure_within_project(path)
        for path in PROJECT_ROOT.glob("*.zip")
        if _ensure_within_project(path) != OUTPUT_ZIP.resolve()
    )


def _cleanup_temp_dirs() -> None:
    for directory in (BUILDER_STAGE_DIR, LEGACY_STAGE_DIR):
        target = _ensure_within_project(directory)
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)


def _remove_extra_root_zips() -> list[str]:
    removed: list[str] = []
    for path in _root_level_extra_zips():
        if path.exists():
            path.unlink()
            removed.append(path.name)
    return removed


def _manifest_text() -> str:
    lines = [
        "# ZIP Contents Manifest",
        "",
        f"Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Packaging Goal",
        "",
        "Bundle canonico unico, austero y coherente con los artefactos vigentes del laboratorio.",
        "",
        "## Included",
        "",
        "| Archivo | Motivo de inclusion |",
        "| --- | --- |",
    ]
    included_entries = _canonical_file_entries()
    for relative_path, reason in included_entries:
        lines.append(f"| `{relative_path}` | {reason} |")

    lines.extend(["", "## Excluded", ""])
    for rule in EXCLUSION_RULES:
        lines.append(f"- {rule}")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Included file count: `{len(included_entries)}`",
            "- Canonical output: `000_PARA_CHATGPT.zip`",
            "- Scope: estado vigente del laboratorio + artefactos canonicos activos dentro del proyecto.",
        ]
    )
    return "\n".join(lines) + "\n"


def _audit_text(previous_zip_bytes: int, removed_root_zips: list[str]) -> str:
    removed_text = ", ".join(f"`{name}`" for name in removed_root_zips) if removed_root_zips else "ninguno"
    lines = [
        "# ZIP Packaging Audit",
        "",
        f"Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Rebuild Status",
        "",
        "- Reconstruccion desde cero: SI",
        "- Reemplazo completo del zip anterior: SI",
        f"- Zip previo detectado: `{previous_zip_bytes}` bytes",
        f"- Zips extra eliminados de la raiz: {removed_text}",
        "",
        "## Canonical Content Criterion",
        "",
        "- Se mantiene solo el set minimo vigente para entender el estado real del laboratorio y operar seguro dentro del proyecto.",
        "- Se conserva la referencia al benchmark H6 solo como benchmark comparativo vigente.",
        "- Se agregan solo artefactos canonicos reutilizables y resultados activos cuando existen fisicamente.",
        "- `ZIP_DELIVERY_STATUS.md` se mantiene fuera del zip para evitar una autorreferencia imposible entre hash del zip y contenido interno.",
        "",
        "## Exclusion Criterion Applied",
        "",
    ]
    for rule in EXCLUSION_RULES:
        lines.append(f"- {rule}")

    lines.extend(
        [
            "",
            "## Integrity Checks",
            "",
            f"- Archivos canonicos incluidos: `{len(_canonical_file_entries())}`",
            "- Ausencia de duplicados logicos por nombre interno: SI",
            "- Ausencia de backups dentro del zip: SI",
            "- Ausencia de archivos intermedios dentro del zip: SI",
            "- Coherencia con documentos canonicos activos: SI",
            "- H6 preservado como benchmark vigente e intocable: SI",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_packaging_docs(previous_zip_bytes: int, removed_root_zips: list[str]) -> None:
    MANIFEST_PATH.write_text(_manifest_text(), encoding="utf-8")
    AUDIT_PATH.write_text(_audit_text(previous_zip_bytes, removed_root_zips), encoding="utf-8")


def _write_delivery_status(*, file_count: int, output_zip_bytes: int) -> None:
    sha256 = hashlib.sha256(OUTPUT_ZIP.read_bytes()).hexdigest().upper()
    lines = [
        "# ZIP Delivery Status",
        "",
        f"- Archivo: {OUTPUT_ZIP.name}",
        f"- Ruta Absoluta: {OUTPUT_ZIP}",
        f"- Tamano: {output_zip_bytes} bytes",
        f"- Timestamp UTC: {datetime.now(timezone.utc).replace(microsecond=0).isoformat()}",
        f"- SHA256: {sha256}",
        f"- Cantidad de Archivos Internos: {file_count}",
        "- Estado Final: READY_FOR_UPLOAD",
        "- Nota: este archivo se mantiene fuera del zip para evitar autorreferencia de hash/tamano.",
    ]
    DELIVERY_STATUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_zip(bundle_files: list[Path]) -> int:
    _cleanup_temp_dirs()
    stage_dir = _ensure_within_project(BUILDER_STAGE_DIR)
    stage_dir.mkdir(parents=True, exist_ok=True)
    temp_zip = _ensure_within_project(stage_dir / OUTPUT_ZIP.name)
    try:
        with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in bundle_files:
                relative = path.resolve().relative_to(CANONICAL_ROOT).as_posix()
                archive.write(path, arcname=relative)
        size = temp_zip.stat().st_size
        os.replace(temp_zip, OUTPUT_ZIP)
        return size
    finally:
        _cleanup_temp_dirs()


def _verify_zip(bundle_files: list[Path]) -> None:
    expected_names = [path.resolve().relative_to(CANONICAL_ROOT).as_posix() for path in bundle_files]
    with zipfile.ZipFile(OUTPUT_ZIP, "r") as archive:
        names = archive.namelist()
        if len(names) != len(expected_names):
            _fail(f"conteo interno inesperado en zip: {len(names)} vs {len(expected_names)}")
        if len(set(names)) != len(names):
            _fail("duplicados internos detectados en el zip")
        if names != expected_names:
            _fail(f"orden o contenido inesperado en zip: {names}")
        forbidden = [name for name in names if "_BACKUP_" in name or name.endswith(".zip")]
        if forbidden:
            _fail(f"entradas prohibidas dentro del zip: {forbidden}")


def build_bundle(*, dry_run: bool = False) -> BuildStats:
    _ensure_canonical_root()
    previous_zip_bytes = OUTPUT_ZIP.stat().st_size if OUTPUT_ZIP.exists() else 0
    root_zip_count_before = len(list(PROJECT_ROOT.glob("*.zip")))

    if dry_run:
        bundle_files = _bundle_files()
        raw_total = sum(path.stat().st_size for path in bundle_files)
        return BuildStats(
            file_count=len(bundle_files),
            raw_total_bytes=raw_total,
            previous_zip_bytes=previous_zip_bytes,
            output_zip_bytes=previous_zip_bytes,
            root_zip_count_before=root_zip_count_before,
            root_zip_count_after=root_zip_count_before,
            removed_root_zips=tuple(path.name for path in _root_level_extra_zips()),
        )

    removed_root_zips = _remove_extra_root_zips()
    _write_packaging_docs(previous_zip_bytes, removed_root_zips)
    bundle_files = _bundle_files()
    raw_total_bytes = sum(path.stat().st_size for path in bundle_files)
    output_zip_bytes = _build_zip(bundle_files)
    _verify_zip(bundle_files)
    root_zip_count_after = len(list(PROJECT_ROOT.glob("*.zip")))
    if root_zip_count_after != 1:
        _fail(f"deberia quedar un solo zip en raiz y quedaron {root_zip_count_after}")
    _write_delivery_status(file_count=len(bundle_files), output_zip_bytes=output_zip_bytes)

    return BuildStats(
        file_count=len(bundle_files),
        raw_total_bytes=raw_total_bytes,
        previous_zip_bytes=previous_zip_bytes,
        output_zip_bytes=output_zip_bytes,
        root_zip_count_before=root_zip_count_before,
        root_zip_count_after=root_zip_count_after,
        removed_root_zips=tuple(removed_root_zips),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruye desde cero el 000_PARA_CHATGPT.zip canonico.")
    parser.add_argument("--dry-run", action="store_true", help="Calcula el bundle sin reemplazar el zip.")
    args = parser.parse_args()

    result = build_bundle(dry_run=args.dry_run)
    print(f"files={result.file_count}")
    print(f"raw_total_bytes={result.raw_total_bytes}")
    print(f"previous_zip_bytes={result.previous_zip_bytes}")
    print(f"output_zip_bytes={result.output_zip_bytes}")
    print(f"root_zip_count_before={result.root_zip_count_before}")
    print(f"root_zip_count_after={result.root_zip_count_after}")
    print(f"removed_root_zips={','.join(result.removed_root_zips) if result.removed_root_zips else 'none'}")


if __name__ == "__main__":
    main()
