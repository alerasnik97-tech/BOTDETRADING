from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch.early_forward_expectation_lib import get_line_status
from scratch.forward_telemetry_lib import load_trace_frame, source_hash_summary, telemetry_snapshot_by_line
from scratch.post_hardening_drift_lib import LINE_CONFIGS, compute_metrics, load_forward_bundle

RESULTS_DIR = ROOT / "results"
SCOREBOARD_CSV = RESULTS_DIR / "SCBI_DUAL_LINE_SCOREBOARD.csv"
TRIBUNAL_JSON = RESULTS_DIR / "SCBI_FORWARD_TRIBUNAL_SUMMARY.json"
DRIFT_REPORT_JSON = RESULTS_DIR / "SCBI_SIGNAL_DRIFT_REPORT.json"
READINESS_STATUS_JSON = ROOT / "REAL_READINESS_GATE_STATUS.json"
TRIBUNAL_STATUS_JSON = ROOT / "FORWARD_TRIBUNAL_STATUS.json"
TELEMETRY_STATUS_JSON = ROOT / "FORWARD_TELEMETRY_STATUS.json"
PROP_RISK_STATUS_JSON = ROOT / "PROP_FIRM_RISK_LAYER_STATUS.json"
OUTPUT_JSON = RESULTS_DIR / "SCBI_UNIFIED_LINE_STATUS.json"
OUTPUT_CSV = RESULTS_DIR / "SCBI_UNIFIED_LINE_STATUS.csv"
SEQUENTIAL_STATUS_JSON = RESULTS_DIR / "SCBI_SEQUENTIAL_EVIDENCE_STATUS.json"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Falta fuente requerida: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_close(actual: float, expected: float, *, label: str, tol: float = 1e-6) -> None:
    if abs(actual - expected) > tol:
        raise RuntimeError(f"FAIL-CLOSED: mismatch en {label}: actual={actual} expected={expected}")


def as_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def as_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def sample_checkpoint_label(n: int) -> str:
    if n < 5:
        return "BELOW_ENVELOPE_MINIMUM"
    if n < 10:
        return "EXPECTATION_ONLY"
    if n < 20:
        return "PRE_DEMO_GATE"
    if n < 40:
        return "DEMO_GATE_WINDOW"
    return "POST_DEMO_GATE"


def expectation_note(label: str, n: int) -> str:
    notes = {
        "WITHIN_EXPECTATION_ENVELOPE": "La muestra corta cae dentro del envelope esperado.",
        "STRETCHED_BUT_STILL_NORMAL": "La muestra esta estirada, pero sigue dentro del ruido historico.",
        "EARLY_WARNING": "La muestra activa vigilancia temprana y no debe sobre-interpretarse sola.",
        "OUTSIDE_EXPECTATION_ENVELOPE": "La muestra queda fuera del envelope historico y exige revision.",
        "NOT_ENOUGH_SAMPLE": f"La capa de expectation no emite juicio con N={n}; el primer checkpoint es N=5.",
        "EXPECTATION_MODEL_NOT_RELIABLE": "La capa de expectation no pudo emitir una lectura confiable.",
    }
    return notes.get(label, "Estado de expectation sin nota definida.")


def promotion_state_from_verdict(verdict: str) -> str:
    upper = verdict.upper()
    if "DEMO_ELIGIBLE" in upper:
        return "DEMO_ELIGIBLE"
    if "PROMOTION_BLOCKED" in upper or "SUSPENDED" in upper:
        return "DEMO_BLOCKED"
    return "PAPER_ONLY"


def promotion_reason(n: int, verdict: str) -> str:
    upper = verdict.upper()
    if "DEMO_ELIGIBLE" in upper:
        return "La linea ya cumple el umbral del tribunal para pasar a demo, sujeto al gate institucional."
    if "PROMOTION_BLOCKED" in upper:
        return "El tribunal ya bloqueo una promocion y no habilita avanzar."
    if "SUSPENDED" in upper:
        return "La linea esta suspendida por riesgo y no puede promocionarse."
    if n < 20:
        return "La readiness gate sigue cerrada por muestra oficial menor a 20 trades."
    return "La linea sigue en observacion hasta nuevo gate institucional."


def build_active_risks(
    *,
    n: int,
    tribunal_verdict: str,
    drift_label: str,
    expectation_label: str,
    guard_status: str,
    guard_reason: str,
    incident_code: str,
    sequential_state: str,
    sequential_interpretation: str,
) -> list[str]:
    risks: list[str] = []
    if n < 5:
        risks.append("Muestra oficial menor a 5 trades: no hay lectura de expectation envelope.")
    elif n < 10:
        risks.append("Muestra oficial menor a 10 trades: no hay drift comparable.")
    elif n < 20:
        risks.append("Muestra oficial menor a 20 trades: sigue cerrada la puerta a demo.")

    if drift_label == "NOT_COMPARABLE_YET":
        risks.append("Drift todavia no comparable con rigor estadistico.")
    elif drift_label in {"STRUCTURAL_DRIFT", "DATA_OR_PIPELINE_DRIFT"}:
        risks.append(f"Drift activo detectado: {drift_label}.")

    if expectation_label in {"EARLY_WARNING", "OUTSIDE_EXPECTATION_ENVELOPE"}:
        risks.append(f"Expectation envelope en estado {expectation_label}.")

    if sequential_state in {"EVIDENCE_EARLY_WARNING", "EVIDENCE_MATERIALLY_UNFAVORABLE"}:
        risks.append(f"Capa secuencial en {sequential_state}.")
    if sequential_interpretation == "ESCALATE_TO_TRIBUNAL_NOTE":
        risks.append("La capa secuencial pide elevar nota institucional al proximo corte del tribunal.")

    if guard_status in {"WARNING", "FAIL"} and guard_reason:
        risks.append(f"Guard activo: {guard_reason}")
    elif incident_code:
        risks.append(f"Incidente registrado: {incident_code}")

    if tribunal_verdict.startswith("PROMOTION_BLOCKED"):
        risks.append("El tribunal mantiene la promocion bloqueada.")
    if tribunal_verdict.startswith("SUSPENDED"):
        risks.append("La linea esta suspendida por el tribunal.")

    unique: list[str] = []
    seen: set[str] = set()
    for risk in risks:
        if risk not in seen:
            unique.append(risk)
            seen.add(risk)
    return unique


def build_can_do(n: int) -> list[str]:
    actions = [
        "Seguir acumulando evidencia paper oficial dentro del stack vigente.",
        "Reejecutar scoreboard, tribunal y superficie unificada sin tocar alpha.",
        "Usar la superficie solo como lectura institucional y no como override manual.",
    ]
    if n >= 5:
        actions.append("Usar la capa de expectation como contexto de muestra corta.")
    return actions


def build_cannot_do(n: int) -> list[str]:
    actions = [
        "No promocionar manualmente la linea a demo ni a real.",
        "No vender la linea como lista para real.",
        "No reinterpretar N/PF/Exp/DD fuera de los ledgers canonicamente reconciliados.",
    ]
    if n < 10:
        actions.append("No concluir drift estructural serio con la muestra actual.")
    return actions


def next_action_for_line(n: int, tribunal_verdict: str, sequential_interpretation: str) -> str:
    upper = tribunal_verdict.upper()
    if "SUSPENDED" in upper:
        return "Resolver la causa de suspension y volver a correr el tribunal solo despues de cerrar la incidencia."
    if "PROMOTION_BLOCKED" in upper:
        return "Mantener la linea bloqueada y reunir evidencia adicional antes de cualquier revision institucional."
    if sequential_interpretation == "ESCALATE_TO_TRIBUNAL_NOTE":
        return "Registrar la advertencia secuencial en el proximo corte del tribunal y seguir acumulando evidencia oficial sin cambiar el gating."
    if n < 5:
        return "Acumular evidencia forward oficial hasta alcanzar N=5 y volver a reconstruir la superficie unificada."
    if n < 10:
        return "Acumular evidencia forward oficial hasta N=10 para habilitar la primera lectura comparable de drift."
    if n < 20:
        return "Acumular evidencia forward oficial hasta N=20 y reejecutar tribunal/readiness sin tocar alpha."
    if "DEMO_ELIGIBLE" in upper:
        return "Preparar la validacion de demo y esperar el gate institucional sin promocion manual."
    return "Mantener observacion estricta y reejecutar el tribunal en el proximo corte semanal."


def explanation_for_line(
    *,
    line_name: str,
    tribunal_verdict: str,
    n: int,
    expectation_label: str,
    guard_status: str,
    incident_code: str,
    sequential_state: str,
    sequential_confidence: Any,
    sequential_compatibility: Any,
) -> str:
    parts = [f"{line_name} queda en {tribunal_verdict}."]
    if n < 5:
        parts.append("La muestra oficial sigue demasiado corta para una lectura estadistica fuerte.")
    elif n < 20:
        parts.append("La linea acumula evidencia, pero todavia no llega al gate de demo.")
    if expectation_label == "NOT_ENOUGH_SAMPLE":
        parts.append("La capa de expectation aun no emite juicio.")
    if sequential_state:
        parts.append(
            f"La capa secuencial marca {sequential_state} con confidence={normalize_text(sequential_confidence)} y compatibility={normalize_text(sequential_compatibility)}."
        )
    if guard_status == "WARNING":
        parts.append("Existe una advertencia operativa activa en guards.")
    if incident_code:
        parts.append(f"El ultimo incidente relevante es {incident_code}.")
    return " ".join(parts)


def build_source_pointers(line_name: str, include_sequential: bool) -> list[str]:
    pointers = [
        f"results/SCBI_DUAL_LINE_SCOREBOARD.csv#{line_name}",
        f"results/SCBI_FORWARD_TRIBUNAL_SUMMARY.json#{line_name}",
        f"results/SCBI_SIGNAL_DRIFT_REPORT.json#{line_name}",
        f"results/SCBI_EARLY_FORWARD_EXPECTATION_ENVELOPES.json#{line_name}",
        f"results/SCBI_FORWARD_TELEMETRY_TRACE.csv#line={line_name}",
        f"REAL_READINESS_GATE_STATUS.json#{line_name}",
    ]
    if include_sequential:
        pointers.insert(4, f"results/SCBI_SEQUENTIAL_EVIDENCE_STATUS.json#{line_name}")
    return pointers


def build_validation_bundle(
    *,
    metrics_by_line: dict[str, dict[str, Any]],
    scoreboard_df: pd.DataFrame,
    tribunal_map: dict[str, dict[str, Any]],
    telemetry_snapshot: dict[str, dict[str, Any]],
    readiness_status: dict[str, Any],
) -> dict[str, Any]:
    validations: dict[str, Any] = {
        "status": "PASS",
        "scoreboard_vs_ledgers": {},
        "tribunal_vs_scoreboard": {},
        "telemetry_vs_live_trace": {},
        "readiness_snapshot": {},
    }

    for line_name, metrics in metrics_by_line.items():
        score_row = scoreboard_df.loc[scoreboard_df["Line"] == line_name].iloc[0]
        tribunal_row = tribunal_map[line_name]
        telemetry_row = telemetry_snapshot[line_name]

        ensure_close(float(metrics["performance_distribution"]["n"]), as_float(score_row["Sample_N"]), label=f"{line_name}.Sample_N")
        ensure_close(metrics["performance_distribution"]["pf"], as_float(score_row["PF_Forward"]), label=f"{line_name}.PF_Forward")
        ensure_close(metrics["performance_distribution"]["expectancy"], as_float(score_row["Exp_Forward"]), label=f"{line_name}.Exp_Forward")
        ensure_close(metrics["performance_distribution"]["max_dd"], as_float(score_row["Max_DD_R"]), label=f"{line_name}.Max_DD_R")
        validations["scoreboard_vs_ledgers"][line_name] = "PASS"

        ensure_close(as_float(score_row["Sample_N"]), as_float(tribunal_row["n"]), label=f"{line_name}.tribunal_n")
        ensure_close(as_float(score_row["PF_Forward"]), as_float(tribunal_row["pf"]), label=f"{line_name}.tribunal_pf")
        ensure_close(as_float(score_row["Max_DD_R"]), as_float(tribunal_row["dd"]), label=f"{line_name}.tribunal_dd")
        validations["tribunal_vs_scoreboard"][line_name] = "PASS"

        if normalize_text(score_row["Telemetry_Execution_Fidelity"]) != normalize_text(telemetry_row["execution_fidelity"]):
            raise RuntimeError(f"FAIL-CLOSED: telemetry execution mismatch en {line_name}")
        if normalize_text(score_row["Telemetry_Blocking_Fidelity"]) != normalize_text(telemetry_row["blocking_fidelity"]):
            raise RuntimeError(f"FAIL-CLOSED: telemetry blocking mismatch en {line_name}")
        if normalize_text(score_row["Telemetry_Last_Guard_Status"]) != normalize_text(telemetry_row["last_guard_status"]):
            raise RuntimeError(f"FAIL-CLOSED: telemetry guard mismatch en {line_name}")
        if normalize_text(score_row["Telemetry_Last_Incident"]) != normalize_text(telemetry_row["last_incident_code"]):
            raise RuntimeError(f"FAIL-CLOSED: telemetry incident mismatch en {line_name}")
        ensure_close(as_float(score_row["Telemetry_Lineage_Coverage"]), as_float(telemetry_row["lineage_coverage"]), label=f"{line_name}.telemetry_lineage")
        validations["telemetry_vs_live_trace"][line_name] = "PASS"

        validations["readiness_snapshot"][line_name] = readiness_status["current_promotions"].get(line_name, "UNKNOWN")

    return validations


def build_unified_surface() -> tuple[dict[str, Any], pd.DataFrame]:
    scoreboard_df = pd.read_csv(SCOREBOARD_CSV)
    tribunal_summary = read_json(TRIBUNAL_JSON)
    drift_report = read_json(DRIFT_REPORT_JSON)
    readiness_status = read_json(READINESS_STATUS_JSON)
    tribunal_status = read_json(TRIBUNAL_STATUS_JSON)
    telemetry_status = read_json(TELEMETRY_STATUS_JSON)
    prop_risk_status = read_json(PROP_RISK_STATUS_JSON)
    sequential_status = read_optional_json(SEQUENTIAL_STATUS_JSON)
    sequential_validator_decision = ""
    if sequential_status:
        sequential_validator_decision = str(sequential_status.get("validation", {}).get("validator_decision", ""))
    sequential_enabled = sequential_validator_decision == "SEQUENTIAL_EVIDENCE_LAYER_CONFIRMED"
    sequential_map = sequential_status.get("lines", {}) if sequential_enabled else {}

    trace_df = load_trace_frame()
    trace_summary = source_hash_summary()
    telemetry_snapshot = telemetry_snapshot_by_line(trace_df)
    tribunal_map = {entry["line"]: entry for entry in tribunal_summary["verdicts"]}
    metrics_by_line: dict[str, dict[str, Any]] = {}
    line_cards: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []

    for line_name, config in LINE_CONFIGS.items():
        bundle = load_forward_bundle(line_name)
        metrics = compute_metrics(bundle["standardized_trades"], config["level_order"])
        metrics_by_line[line_name] = metrics

        score_row = scoreboard_df.loc[scoreboard_df["Line"] == line_name].iloc[0]
        tribunal_row = tribunal_map[line_name]
        drift_line = drift_report["lines"][line_name]
        telemetry_line = telemetry_snapshot[line_name]

        n = int(metrics["performance_distribution"]["n"])
        pf = float(metrics["performance_distribution"]["pf"])
        expectancy = float(metrics["performance_distribution"]["expectancy"])
        max_dd = float(metrics["performance_distribution"]["max_dd"])
        expectation_label = get_line_status(
            line_name,
            n,
            {
                "pf": pf,
                "expectancy": expectancy,
                "max_dd": max_dd,
            },
        )
        tribunal_verdict = str(tribunal_row["verdict"])
        line_promotion_state = promotion_state_from_verdict(tribunal_verdict)
        guard_status = str(telemetry_line["last_guard_status"])
        guard_reason = str(telemetry_line["last_guard_reason"])
        incident_code = str(telemetry_line["last_incident_code"])
        sequential_line = sequential_map.get(line_name, {})
        sequential_current = sequential_line.get("current_state", {})
        sequential_state = str(sequential_current.get("sequential_evidence_state", ""))
        sequential_interpretation = str(sequential_current.get("recommended_interpretation_state", ""))

        active_risks = build_active_risks(
            n=n,
            tribunal_verdict=tribunal_verdict,
            drift_label=str(drift_line["verdict"]),
            expectation_label=expectation_label,
            guard_status=guard_status,
            guard_reason=guard_reason,
            incident_code=incident_code,
            sequential_state=sequential_state,
            sequential_interpretation=sequential_interpretation,
        )
        next_action = next_action_for_line(n, tribunal_verdict, sequential_interpretation)
        explanation = explanation_for_line(
            line_name=line_name,
            tribunal_verdict=tribunal_verdict,
            n=n,
            expectation_label=expectation_label,
            guard_status=guard_status,
            incident_code=incident_code,
            sequential_state=sequential_state,
            sequential_confidence=sequential_current.get("institutional_confidence_score"),
            sequential_compatibility=sequential_current.get("cumulative_compatibility_score"),
        )

        card = {
            "source_line": line_name,
            "line_status_clarity": "LINE_STATUS_CLEAR",
            "institutional_operating_state": tribunal_verdict,
            "surface_relation_to_truth": "DECISION_SURFACE_CANONICAL",
            "automation_posture": "AUTOMATION_SAFE",
            "OFFICIAL_SAMPLE_STATE": {
                "sample_n": n,
                "sample_checkpoint": sample_checkpoint_label(n),
                "last_activity_ny": metrics["last_activity_ny"] or "N/A",
                "official_ledger_source": str(config["forward_path"].relative_to(ROOT)),
                "forward_evidence_read_surface": bundle["source_path"],
            },
            "PERFORMANCE_STATE": {
                "pf_forward": pf,
                "expectancy_r": expectancy,
                "max_dd_r": max_dd,
                "scoreboard_pointer": f"results/SCBI_DUAL_LINE_SCOREBOARD.csv#{line_name}",
            },
            "DRIFT_STATE": {
                "drift_label": drift_line["verdict"],
                "drift_r": drift_line["drift_r"],
                "drift_governance_mode": drift_report["tribunal_integration_mode"],
                "drift_validation_status": drift_report["monitor_validation_verdict"],
                "drift_gate_applied": bool(tribunal_row["drift_gate_applied"]),
            },
            "EARLY_EXPECTATION_STATE": {
                "expectation_label": expectation_label,
                "note": expectation_note(expectation_label, n),
                "envelope_source": f"results/SCBI_EARLY_FORWARD_EXPECTATION_ENVELOPES.json#{line_name}",
            },
            "TELEMETRY_STATE": {
                "trace_path": trace_summary["trace_path"],
                "trace_hash": trace_summary["trace_hash"],
                "trace_rows": trace_summary["trace_rows"],
                "execution_fidelity": telemetry_line["execution_fidelity"],
                "blocking_fidelity": telemetry_line["blocking_fidelity"],
                "lineage_coverage": telemetry_line["lineage_coverage"],
                "official_event_count_live": telemetry_line["official_trace_events"],
                "official_trade_events": telemetry_line["official_trade_events"],
                "block_events": telemetry_line["block_events"],
            },
            "F_GUARD_AND_INCIDENT_STATE": {
                "guard_status": guard_status,
                "guard_reason": guard_reason,
                "incident_code": incident_code,
                "prop_risk_decision": prop_risk_status["decision"],
            },
            "READINESS_STATE": {
                "readiness_decision": readiness_status["decision"],
                "readiness_status": readiness_status["status"],
                "current_promotion_state": readiness_status["current_promotions"].get(line_name, "UNKNOWN"),
                "gate_rules": readiness_status["gates"],
            },
            "PROMOTION_STATE": {
                "promotion_state": line_promotion_state,
                "promotion_reason": promotion_reason(n, tribunal_verdict),
                "tribunal_rules": tribunal_status["rules"],
            },
            "NEXT_ACTION_STATE": {
                "next_action": next_action,
            },
            "summary": explanation,
            "active_risks": active_risks,
            "what_can_do": build_can_do(n),
            "what_cannot_do": build_cannot_do(n),
            "canonical_source_pointers": build_source_pointers(line_name, sequential_enabled),
        }
        if sequential_enabled:
            card["SEQUENTIAL_EVIDENCE_STATE"] = {
                "status_available": bool(sequential_line),
                "institutional_confidence_score": sequential_current.get("institutional_confidence_score"),
                "cumulative_compatibility_score": sequential_current.get("cumulative_compatibility_score"),
                "sequential_evidence_state": sequential_state,
                "evidence_delta_per_trade": sequential_current.get("evidence_delta_per_trade"),
                "evidence_delta_per_day": sequential_current.get("evidence_delta_per_day"),
                "low_n_caution_state": sequential_current.get("low_n_caution_state"),
                "direction_of_confidence_change": sequential_current.get("direction_of_confidence_change"),
                "recommended_interpretation_state": sequential_interpretation,
                "expected_next_trade_delta_band": sequential_current.get("expected_next_trade_delta_band"),
                "integration_posture": None if not sequential_line else sequential_line.get("integration_posture"),
                "source_pointer": f"results/SCBI_SEQUENTIAL_EVIDENCE_STATUS.json#{line_name}",
            }
        line_cards.append(card)
        flat_row = {
            "Line": line_name,
            "Line_Status_Clarity": "LINE_STATUS_CLEAR",
            "Institutional_Operating_State": tribunal_verdict,
            "Sample_N": n,
            "PF_Forward": pf,
            "Exp_Forward": expectancy,
            "Max_DD_R": max_dd,
            "Drift_Label": drift_line["verdict"],
            "Expectation_Label": expectation_label,
            "Telemetry_Execution_Fidelity": telemetry_line["execution_fidelity"],
            "Telemetry_Blocking_Fidelity": telemetry_line["blocking_fidelity"],
            "Telemetry_Lineage_Coverage": telemetry_line["lineage_coverage"],
            "Guard_Status": guard_status,
            "Incident_Code": incident_code,
            "Readiness_State": readiness_status["current_promotions"].get(line_name, "UNKNOWN"),
            "Promotion_State": line_promotion_state,
            "Next_Action": next_action,
            "Summary": explanation,
            "Active_Risks": " | ".join(active_risks),
        }
        if sequential_enabled:
            flat_row.update(
                {
                    "Sequential_Evidence_State": sequential_state,
                    "Sequential_Confidence_Score": sequential_current.get("institutional_confidence_score"),
                    "Sequential_Compatibility_Score": sequential_current.get("cumulative_compatibility_score"),
                    "Sequential_Interpretation": sequential_interpretation,
                    "Sequential_Confidence_Direction": sequential_current.get("direction_of_confidence_change"),
                }
            )
        flat_rows.append(flat_row)

    validation = build_validation_bundle(
        metrics_by_line=metrics_by_line,
        scoreboard_df=scoreboard_df,
        tribunal_map=tribunal_map,
        telemetry_snapshot=telemetry_snapshot,
        readiness_status=readiness_status,
    )

    overall = {
        "generated_at_utc": now_utc_iso(),
        "engine": "UNIFIED_LINE_STATUS_ENGINE_V1",
        "benchmark_reference": {
            "strategy": "H6_SILVER_BULLET_HYBRID",
            "role": "Benchmark conceptual congelado",
        },
        "taxonomy_outcome": {
            "line_status": "LINE_STATUS_CLEAR",
            "surface_status": "DECISION_SURFACE_CANONICAL",
            "automation_status": "AUTOMATION_SAFE",
        },
        "integration_decision": {
            "chosen_shape": "STATUS_FILE_AUTONOMO_PLUS_THIN_TABLE",
            "kept_existing_scoreboard": True,
            "reason": "El scoreboard vigente conserva la capa cuantitativa; la nueva superficie agrega lectura institucional sin romper consumidores existentes.",
        },
        "sequential_evidence_status": {
            "available": bool(sequential_status),
            "integrated_in_surface": sequential_enabled,
            "validator_decision": sequential_validator_decision or "UNAVAILABLE",
            "trace_path": "results/SCBI_SEQUENTIAL_EVIDENCE_TRACE.csv" if sequential_status else "",
            "daily_path": "results/SCBI_SEQUENTIAL_EVIDENCE_DAILY.csv" if sequential_status else "",
            "validator_summary_available": bool(sequential_status and sequential_status.get("validation", {}).get("validator_summary_available")),
            "integration_posture": "MONITOR_ONLY" if sequential_enabled else ("NOT_CANONIZED_PENDING_REFINEMENT" if sequential_status else "UNAVAILABLE"),
        },
        "lab_summary": {
            "GLOBAL": next(card for card in line_cards if card["source_line"] == "SCBI_M5_GLOBAL")["institutional_operating_state"],
            "CORE": next(card for card in line_cards if card["source_line"] == "SCBI_CORE")["institutional_operating_state"],
            "meaning": (
                "El laboratorio queda legible por linea en una sola vista derivada: arquitectura fuerte, evidencia forward corta y sin permiso de promocion."
                if not sequential_enabled
                else "El laboratorio queda legible por linea en una sola vista derivada: arquitectura fuerte, evidencia forward corta y ahora con lectura secuencial monitor-only entre checkpoints."
            ),
            "what_can_do": [
                "Seguir paper oficial y generar lectura institucional diaria/semanal.",
                "Reejecutar la superficie sin tocar alpha ni lineas activas.",
                "Usar la vista unificada como tablero de decision y no como sustituto de los ledgers.",
            ],
            "what_cannot_do": [
                "No promover GLOBAL ni CORE a demo o real con la muestra vigente.",
                "No forzar conclusiones de drift estructural con N aun corta.",
                "No contradecir tribunal, readiness ni ledgers desde la vista derivada.",
            ],
            "next_action": "Acumular evidencia forward oficial y reconstruir la superficie unificada en cada corte semanal del tribunal.",
        },
        "source_precedence": [
            "Ledgers y outputs oficiales en results/",
            "Status json vigentes",
            "Documentos protocol/results/decision",
            "Artefactos historicos de freeze solo como contexto puntual",
        ],
        "trace_summary": trace_summary,
        "telemetry_status": {
            "decision": telemetry_status["decision"],
            "status": telemetry_status["status"],
        },
        "tribunal_status": {
            "decision": tribunal_status["decision"],
            "status": tribunal_status["status"],
        },
        "readiness_status": {
            "decision": readiness_status["decision"],
            "status": readiness_status["status"],
        },
        "prop_risk_status": {
            "decision": prop_risk_status["decision"],
            "status": prop_risk_status["status"],
        },
        "validation": validation,
        "lines": line_cards,
    }

    csv_frame = pd.DataFrame(flat_rows)
    return overall, csv_frame


def write_unified_surface(surface: dict[str, Any], csv_frame: pd.DataFrame) -> None:
    OUTPUT_JSON.write_text(json.dumps(surface, indent=2, ensure_ascii=True), encoding="utf-8")
    csv_frame.to_csv(OUTPUT_CSV, index=False)
