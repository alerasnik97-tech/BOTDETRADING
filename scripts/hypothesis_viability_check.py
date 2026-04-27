import json
import os
import sys
from datetime import datetime

# Definicion de Campos Obligatorios (Stage-0 Contract)
REQUIRED_FIELDS = [
    "name", "spec_clarity_score", "technical_austerity_score",
    "fast_falsifiability_score", "compute_budget_score", "signal_purity_score",
    "is_fully_specified", "has_defined_kill_conditions",
    "has_low_variants_bloat", "dataset_is_ready"
]

VETO_MAP = {
    "is_fully_specified": "VETO_SPEC_AMBIGUITY",
    "has_defined_kill_conditions": "VETO_NO_KILL_CONDITIONS",
    "has_low_variants_bloat": "VETO_COMPLEXITY_BLOAT",
    "dataset_is_ready": "VETO_DATA_UNAVAILABLE"
}

def check_viability(data):
    """
    Evalua la viabilidad operativa de una hipotesis admitida (Stage-0).
    """
    report_lines = []
    report_lines.append("# EURUSD Hypothesis Viability Report (Stage-0)")
    report_lines.append(f"**Fecha**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Hipotesis**: {data.get('name', 'UNKNOWN')}")
    report_lines.append("-" * 30)

    # 1. Validacion de Schema (Fail-Closed)
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        msg = f"ERROR: Fallo de integridad en input Stage-0. Campos faltantes: {', '.join(missing)}"
        report_lines.append(f"> [!CAUTION]\n> {msg}")
        _write_report(data.get('name', 'UNKNOWN'), report_lines)
        return "FAIL_CLOSED", ["SCHEMA_ERROR"]

    # 2. Validacion de Vetos de Viabilidad
    triggered_vetoes = []
    for field, code in VETO_MAP.items():
        if not data[field]:
            triggered_vetoes.append(code)

    if triggered_vetoes:
        report_lines.append("### Veredicto: **BLOCKED_PRE_CAMPAIGN**")
        report_lines.append("> [!WARNING]\n> La hipotesis no es viable operativamente en su estado actual.")
        for code in triggered_vetoes:
            report_lines.append(f"- **Reason Code**: `{code}`")
        _write_report(data['name'], report_lines)
        return "BLOCKED_PRE_CAMPAIGN", triggered_vetoes

    # 3. Calificacion Operativa (Scoring)
    report_lines.append("## Calificacion de Viabilidad")
    
    score_weights = {
        "spec_clarity_score": 30,
        "technical_austerity_score": 25,
        "fast_falsifiability_score": 25,
        "compute_budget_score": 10,
        "signal_purity_score": 10
    }

    total_score = 0
    for key, weight in score_weights.items():
        val = data[key]
        total_score += min(val, weight)

    report_lines.append(f"- **Viability Score**: {total_score} / 100")
    
    # Metadata Operativa
    report_lines.append(f"- **Stage-1 Sample Target**: N={data.get('stage1_sample_size', 20)}")
    report_lines.append(f"- **Stage-1 Kill Drawdown**: {data.get('stage1_kill_drawdown', -5.0)}R")

    # 4. Veredicto Final Stage-0
    if total_score >= 80:
        verdict = "ELIGIBLE_FOR_CAMPAIGN"
    elif total_score >= 60:
        verdict = "NEEDS_TIGHTER_SPEC"
    else:
        verdict = "BLOCKED_PRE_CAMPAIGN"

    report_lines.append(f"\n## Veredicto Final: **{verdict}**")
    
    if verdict == "ELIGIBLE_FOR_CAMPAIGN":
        report_lines.append("> [!TIP]\n> La hipotesis tiene una arquitectura lista para Intake formal.")
    elif verdict == "NEEDS_TIGHTER_SPEC":
        report_lines.append("> [!NOTE]\n> Se requiere endurecer la especificacion antes de abrir la campaña.")
    else:
        report_lines.append("> [!IMPORTANT]\n> La hipotesis es demasiado ruidosa o costosa para el presupuesto actual.")

    _write_report(data['name'], report_lines)
    return verdict, []

def _write_report(name, lines):
    # Genera reporte markdown
    safe_name = name.replace(" ", "_").upper()
    filename = f"VIABILITY_REPORT_{safe_name}.md"
    content = "\n".join(lines)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Reporte generado: {filename}")

def main():
    if len(sys.argv) < 2:
        print("Uso: python scripts/hypothesis_viability_check.py <path_al_json>")
        sys.exit(1)

    json_path = sys.argv[1]
    if not os.path.exists(json_path):
        print(f"Error: Archivo no encontrado: {json_path}")
        sys.exit(1)

    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: El archivo {json_path} no es un JSON Valido.")
            sys.exit(1)

    verdict, reasons = check_viability(data)
    print(f"Veredicto: {verdict}")
    if reasons:
        print(f"Reason Codes: {reasons}")

if __name__ == "__main__":
    main()
