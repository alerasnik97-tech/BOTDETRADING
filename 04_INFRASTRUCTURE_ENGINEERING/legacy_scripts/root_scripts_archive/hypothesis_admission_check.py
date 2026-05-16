import json
import os
import sys
from datetime import datetime

# Definicion de Constantes y Reglas (Hardened)
REQUIRED_FIELDS = [
    "name", "is_programmable", "has_news_fortress", 
    "is_not_rejected_family", "friction_resistant", 
    "has_material_differentiation", "programmability_score",
    "differentiation_score", "temporal_selectivity_score",
    "austerity_score", "htf_confluence_score", "base_evidence_score"
]

VETO_CODES = {
    "is_programmable": "VETO_SUBJECTIVITY",
    "has_news_fortress": "VETO_NEWS_FORTRESS",
    "is_not_rejected_family": "VETO_REJECTED_FAMILY",
    "friction_resistant": "VETO_FRICTION",
    "has_material_differentiation": "VETO_NO_DIFFERENCE"
}

def check_hypothesis(data):
    """
    Evalua una hipotesis con validacion de schema, vetos y scoring calibrado.
    Genera un reporte markdown atómico.
    """
    report_lines = []
    report_lines.append("# EURUSD Hypothesis Evaluation Report")
    report_lines.append(f"**Fecha**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Hipotesis**: {data.get('name', 'UNKNOWN')}")
    report_lines.append("-" * 30)

    # 1. Validacion de Schema (Fail-Closed)
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        msg = f"ERROR: Fallo de validacion de schema. Faltan campos: {', '.join(missing)}"
        report_lines.append(f"> [!CAUTION]\n> {msg}")
        _write_report(data.get('name', 'UNKNOWN'), report_lines)
        return "BLOCKED_HARD", ["SCHEMA_ERROR"]

    # 2. Validacion de Vetos (Hard Gates)
    triggered_vetoes = []
    for field, code in VETO_CODES.items():
        if not data[field]:
            triggered_vetoes.append(code)

    if triggered_vetoes:
        report_lines.append("## Veredicto: BLOCKED_HARD")
        report_lines.append("> [!WARNING]\n> Violacion de Vetos Criticos Detectada.")
        for code in triggered_vetoes:
            report_lines.append(f"- **Reason Code**: `{code}`")
        _write_report(data['name'], report_lines)
        return "BLOCKED_HARD", triggered_vetoes

    # 3. Scoring y Penalizaciones
    report_lines.append("## Scoring Breakdown")
    raw_score = (
        data['programmability_score'] + data['differentiation_score'] +
        data['temporal_selectivity_score'] + data['austerity_score'] +
        data['htf_confluence_score'] + data['base_evidence_score']
    )
    report_lines.append(f"- **Raw Score**: {raw_score} / 100")

    penalties = 0
    if data.get('penalty_excess_confirmation'):
        penalties += 10
        report_lines.append("- **Penalizacion**: `PENALTY_EXCESS_CONFIRMATION` (-10 pts)")
    if data.get('penalty_wide_time_domain'):
        penalties += 15
        report_lines.append("- **Penalizacion**: `PENALTY_WIDE_TIME_DOMAIN` (-15 pts)")
    if data.get('penalty_no_fixed_sl'):
        penalties += 20
        report_lines.append("- **Penalizacion**: `PENALTY_NO_FIXED_SL` (-20 pts)")

    final_score = max(0, raw_score - penalties)
    report_lines.append(f"### **Final Score**: {final_score} / 100")

    # 4. Veredicto Final
    if final_score >= 80:
        verdict = "ADMISSIBLE_FOR_INTAKE"
    elif final_score >= 60:
        verdict = "NEEDS_REDESIGN"
    else:
        verdict = "BLOCKED_HARD"

    report_lines.append(f"\n## VEREDICTO FINAL: **{verdict}**")
    
    if verdict == "ADMISSIBLE_FOR_INTAKE":
        report_lines.append("> [!TIP]\n> La hipotesis cumple con los estandares de rigor del laboratorio.")
    elif verdict == "NEEDS_REDESIGN":
        report_lines.append("> [!NOTE]\n> La hipotesis es prometedora pero estructuralmente debil o ruidosa.")
    else:
        report_lines.append("> [!IMPORTANT]\n> Hipotesis bloqueada por baja calidad estadistica o riesgo estructural.")

    _write_report(data['name'], report_lines)
    return verdict, []

def _write_report(name, lines):
    filename = f"ADMISSION_REPORT_{name.replace(' ', '_').upper()}.md"
    content = "\n".join(lines)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Reporte generado: {filename}")

def main():
    if len(sys.argv) < 2:
        print("Uso: python scripts/hypothesis_admission_check.py <path_al_json>")
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

    verdict, reasons = check_hypothesis(data)
    print(f"Veredicto: {verdict}")
    if reasons:
        print(f"Reason Codes: {reasons}")

if __name__ == "__main__":
    main()
