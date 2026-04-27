import json
import csv
import os
import sys
from datetime import datetime

# Configuración de rutas
BASE_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
RESEARCH_DIR = os.path.join(BASE_DIR, "institutional_research_candidate_lab", "outputs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
GATE_DIR = os.path.join(BASE_DIR, "real_readiness_gate")

# Thresholds Institucionales (Sincronizados con REAL_READINESS_POLICY.md)
POLICY = {
    "min_sample_size": 1500,
    "min_pf": 2.0,
    "min_expectancy": 0.30,
    "max_drawdown": 15.0, # Valor absoluto para comparación
    "min_year_positive_ratio": 1.0,
    "max_year_profit_concentration": 0.30,
    "max_timeout_exit_rate": 0.40,
    "min_shadow_trades": 20,
    "max_shadow_drift": 0.10
}

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def evaluate_line(variant_id):
    scorecard = {
        "variant_id": variant_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "gates": {},
        "verdict": "NOT_READY",
        "blockers": [],
        "recommendations": []
    }

    # 1. Cargar Datos de Research
    research_summary = load_json(os.path.join(RESEARCH_DIR, "research_summary.json"))
    if not research_summary:
        scorecard["blockers"].append("MISSING_RESEARCH_SUMMARY")
        return scorecard

    variant_data = next((v for v in research_summary.get("top_variants", []) if v["variant_id"] == variant_id), None)
    if not variant_data:
        matrix_results = load_csv(os.path.join(RESEARCH_DIR, "research_matrix_results.csv"))
        variant_data = next((v for v in matrix_results if v["variant_id"] == variant_id), None)
        if variant_data:
            variant_data["pf"] = float(variant_data["pf"])
            variant_data["expectancy"] = float(variant_data["expectancy"])
            variant_data["sample_size"] = int(variant_data["sample_size"])
            variant_data["max_drawdown_R"] = float(variant_data["max_drawdown_R"])
            variant_data["year_positive_ratio"] = float(variant_data["year_positive_ratio"])
            variant_data["timeout_exit_rate"] = float(variant_data["timeout_exit_rate"])
    
    if not variant_data:
        scorecard["blockers"].append("VARIANT_NOT_FOUND_IN_RESEARCH")
        return scorecard

    # --- BLOQUE A: ROBUSTEZ HISTÓRICA ---
    n = variant_data.get("sample_size", 0)
    scorecard["gates"]["sample_size"] = {"value": n, "threshold": POLICY["min_sample_size"], "pass": n >= POLICY["min_sample_size"]}
    if not scorecard["gates"]["sample_size"]["pass"]: scorecard["blockers"].append("INSUFFICIENT_SAMPLE_SIZE")

    pf = variant_data.get("pf", 0)
    scorecard["gates"]["profit_factor"] = {"value": pf, "threshold": POLICY["min_pf"], "pass": pf >= POLICY["min_pf"]}
    if not scorecard["gates"]["profit_factor"]["pass"]: scorecard["blockers"].append("PF_BELOW_THRESHOLD")

    exp = variant_data.get("expectancy", 0)
    scorecard["gates"]["expectancy"] = {"value": exp, "threshold": POLICY["min_expectancy"], "pass": exp >= POLICY["min_expectancy"]}
    if not scorecard["gates"]["expectancy"]["pass"]: scorecard["blockers"].append("EXPECTANCY_TOO_LOW")

    dd = abs(variant_data.get("max_drawdown_R", 0))
    scorecard["gates"]["max_drawdown"] = {"value": dd, "threshold": POLICY["max_drawdown"], "pass": dd <= POLICY["max_drawdown"]}
    if not scorecard["gates"]["max_drawdown"]["pass"]: scorecard["blockers"].append("DRAWDOWN_TOO_HIGH")

    ypr = variant_data.get("year_positive_ratio", 0)
    scorecard["gates"]["year_positive_ratio"] = {"value": ypr, "threshold": POLICY["min_year_positive_ratio"], "pass": ypr >= POLICY["min_year_positive_ratio"]}
    if not scorecard["gates"]["year_positive_ratio"]["pass"]: scorecard["blockers"].append("YEARLY_INSTABILITY")

    try:
        result_by_year = variant_data.get("result_by_year_json", "{}")
        if isinstance(result_by_year, str): result_by_year = json.loads(result_by_year)
        total_profit = sum(y["total_R"] for y in result_by_year.values() if y["total_R"] > 0)
        max_year_profit = max(y["total_R"] for y in result_by_year.values()) if result_by_year else 0
        concentration = max_year_profit / total_profit if total_profit > 0 else 1.0
        scorecard["gates"]["year_concentration"] = {"value": round(concentration, 4), "threshold": POLICY["max_year_profit_concentration"], "pass": concentration <= POLICY["max_year_profit_concentration"]}
    except:
        scorecard["gates"]["year_concentration"] = {"pass": False, "error": "DATA_PARSE_ERROR"}
    if not scorecard["gates"]["year_concentration"].get("pass", False): scorecard["blockers"].append("EXCESSIVE_YEARLY_CONCENTRATION")

    # --- BLOQUE B: ROBUSTEZ OPERATIVA ---
    shadow_dir = os.path.join(RESULTS_DIR, "shadow")
    has_shadow_dir = os.path.exists(shadow_dir)
    scorecard["gates"]["operational_namespace"] = {"value": "EXISTS" if has_shadow_dir else "MISSING", "pass": has_shadow_dir}
    if not has_shadow_dir: scorecard["blockers"].append("MISSING_OPERATIONAL_NAMESPACE")

    # --- BLOQUE C: ROBUSTEZ DE RIESGO ---
    ter = variant_data.get("timeout_exit_rate", 1.0)
    scorecard["gates"]["timeout_dependency"] = {"value": ter, "threshold": POLICY["max_timeout_exit_rate"], "pass": ter <= POLICY["max_timeout_exit_rate"]}
    if not scorecard["gates"]["timeout_dependency"]["pass"]: scorecard["blockers"].append("HIGH_TIMEOUT_DEPENDENCY")

    # --- BLOQUE D: ROBUSTEZ FORWARD / SHADOW ---
    shadow_ledger_path = os.path.join(RESULTS_DIR, "shadow", f"SCBI_SHADOW_LEDGER_{variant_id}.csv")
    shadow_trades = load_csv(shadow_ledger_path)
    n_shadow = len(shadow_trades)
    scorecard["gates"]["shadow_execution"] = {"value": n_shadow, "threshold": POLICY["min_shadow_trades"], "pass": n_shadow >= POLICY["min_shadow_trades"]}
    if not scorecard["gates"]["shadow_execution"]["pass"]: scorecard["blockers"].append("INSUFFICIENT_SHADOW_SAMPLE")

    # --- VEREDICTO FINAL ---
    all_passed = all(g.get("pass", False) for g in scorecard["gates"].values())
    research_passed = all(scorecard["gates"][k]["pass"] for k in ["sample_size", "profit_factor", "expectancy", "max_drawdown", "year_positive_ratio", "year_concentration"])
    risk_passed = scorecard["gates"]["timeout_dependency"]["pass"]

    if all_passed:
        scorecard["verdict"] = "REAL_ELIGIBLE"
    elif research_passed and risk_passed:
        scorecard["verdict"] = "SHADOW_READY"
        scorecard["recommendations"].append("La línea es robusta en backtest. Proceder a crear namespace y ejecutar shadow.")
    else:
        scorecard["verdict"] = "NOT_READY"
        scorecard["recommendations"].append("Refinar parámetros en laboratorio. No cumple estándares de robustez base.")

    if scorecard["blockers"]:
        scorecard["primary_blocker"] = scorecard["blockers"][0]

    return scorecard

def generate_reports(scorecard):
    with open(os.path.join(GATE_DIR, "real_readiness_scorecard.json"), 'w', encoding='utf-8') as f:
        json.dump(scorecard, f, indent=2)

    md = f"""# Real Readiness Scorecard
## Reporte de Evaluación Institucional

**Línea Evaluada:** `{scorecard['variant_id']}`
**Fecha de Evaluación:** `{scorecard['timestamp']}`
**Veredicto:** `{scorecard['verdict']}`

---

## 1. RESUMEN DE GATES

| Gate | Valor | Umbral | Estado |
|------|-------|---------|--------|
"""
    for name, data in scorecard['gates'].items():
        status = "✅ PASS" if data.get('pass') else "❌ FAIL"
        val = data.get('value', 'N/A')
        thresh = data.get('threshold', 'N/A')
        md += f"| {name.replace('_', ' ').capitalize()} | {val} | {thresh} | {status} |\n"

    md += f"\n---\n\n## 2. BLOQUEOS IDENTIFICADOS\n\n"
    if scorecard['blockers']:
        for b in scorecard['blockers']:
            md += f"- `{b}`\n"
    else:
        md += "Ninguno. Todos los gates han sido superados.\n"

    md += f"\n---\n\n## 3. SIGUIENTE PASO ÚNICO\n\n"
    if scorecard['verdict'] == "NOT_READY":
        md += "**REFINAR ESTRATEGIA EN LABORATORIO.** La línea no cumple los estándares mínimos de robustez histórica o de riesgo."
    elif scorecard['verdict'] == "SHADOW_READY":
        md += "**ACTIVAR INFRAESTRUCTURA SHADOW.** La línea es robusta en backtest. Se requiere crear el namespace `results/shadow` e iniciar ejecución forward controlada (N=20)."
    elif scorecard['verdict'] == "REAL_ELIGIBLE":
        md += "**PROCEDER A AUDITORÍA PRE-PILOTO.** La línea ha superado todos los gates, incluyendo forward execution satisfactoria."

    with open(os.path.join(GATE_DIR, "real_readiness_scorecard.md"), 'w', encoding='utf-8') as f:
        f.write(md)

    summary = f"VEREDICTO: {scorecard['verdict']}\nLINEA: {scorecard['variant_id']}\nPRIMARY_BLOCKER: {scorecard.get('primary_blocker', 'NONE')}\nNEXT_STEP: {md.split('## 3. SIGUIENTE PASO ÚNICO')[-1].strip()}\n"
    with open(os.path.join(GATE_DIR, "real_readiness_summary.txt"), 'w', encoding='utf-8') as f:
        f.write(summary)

if __name__ == "__main__":
    target_variant = "tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_0_1_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m"
    if len(sys.argv) > 1: target_variant = sys.argv[1]
    res = evaluate_line(target_variant)
    generate_reports(res)
    print(f"Evaluación completada. Veredicto: {res['verdict']}")
