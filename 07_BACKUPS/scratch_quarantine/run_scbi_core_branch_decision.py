"""
SCBI Core Branch Decision Engine

Reads structural_edge_decomposition.json and evaluates ex-ante thresholds
to determine if a separate SCBI_CORE branch should be formalized.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
INPUT_FILE = ROOT / "results" / "SCBI_2020_2025_DURABILITY" / "structural_edge_decomposition.json"
OUTPUT_DIR = ROOT / "results" / "SCBI_CORE_BRANCH_DECISION"


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Falta {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Data extraction
    global_base = data["global_baseline"]
    total_trades = global_base["N"]
    total_r = global_base["total_r"]
    global_pf = global_base["pf"]

    # Source aggregation
    sources = data["by_liquidity_source"]
    london_n = sources["london_h"]["N"] + sources["london_l"]["N"]
    london_r = sources["london_h"]["total_r"] + sources["london_l"]["total_r"]
    london_pf = data["ablations"]["london_only"]["pf"]

    pdh_pdl_n = sources["pdh"]["N"] + sources["pdl"]["N"]
    pdh_pdl_r = sources["pdh"]["total_r"] + sources["pdl"]["total_r"]
    
    # Ablations
    without_pdh_pdl = data["ablations"]["without_pdh_pdl"]
    
    # Stress Ablations
    stress_global = data["stress_ablations"]["global_stress"]
    stress_without_pdh_pdl = data["stress_ablations"]["without_pdh_pdl_stress"]

    # Cross regime stability
    cross = data["cross_source_regime"]
    pdh_20_21_pf = cross["pdh_2020-2021"]["pf"]
    pdl_22_23_pf = cross["pdl_2022-2023"]["pf"]
    london_min_pf = min([
        cross["london_h_2020-2021"]["pf"], cross["london_h_2022-2023"]["pf"], cross["london_h_2024-2025"]["pf"],
        cross["london_l_2020-2021"]["pf"], cross["london_l_2022-2023"]["pf"], cross["london_l_2024-2025"]["pf"]
    ])

    print("==========================================================")
    print("SCBI CORE BRANCH DECISION - METRICS EVALUATION")
    print("==========================================================")

    # 1. Core Edge Test (London H/L)
    is_core_profit = (london_r / total_r) > 0.30
    is_core_pf = london_pf > 3.0
    is_core_stable = london_min_pf > 1.5
    core_passed = is_core_profit and is_core_pf and is_core_stable
    print(f"\n[CORE EDGE TEST: LONDON H/L] -> {'PASS' if core_passed else 'FAIL'}")
    print(f" - Profit share > 30%: {london_r/total_r*100:.1f}% ({is_core_profit})")
    print(f" - Global PF > 3.0: {london_pf:.3f} ({is_core_pf})")
    print(f" - Min PF any regime > 1.5: {london_min_pf:.3f} ({is_core_stable})")

    # 2. Structural Drag Test (PDH/PDL)
    is_drag_trades = (pdh_pdl_n / total_trades) > 0.30
    is_drag_profit = (pdh_pdl_r / total_r) < 0.10
    is_drag_negative_regime = (pdh_20_21_pf < 1.0) or (pdl_22_23_pf < 1.0)
    drag_passed = is_drag_trades and is_drag_profit and is_drag_negative_regime
    print(f"\n[STRUCTURAL DRAG TEST: PDH/PDL] -> {'PASS' if drag_passed else 'FAIL'}")
    print(f" - Trades share > 30%: {pdh_pdl_n/total_trades*100:.1f}% ({is_drag_trades})")
    print(f" - Profit share < 10%: {pdh_pdl_r/total_r*100:.1f}% ({is_drag_profit})")
    print(f" - Any negative regime (PF<1): PDH 20-21={pdh_20_21_pf:.3f}, PDL 22-23={pdl_22_23_pf:.3f} ({is_drag_negative_regime})")

    # 3. Material Dilution Test
    pf_improvement = without_pdh_pdl["pf"] - global_pf
    is_material_pf = pf_improvement > 1.5
    is_material_stress = stress_without_pdh_pdl["total_r"] > stress_global["total_r"]
    material_passed = is_material_pf and is_material_stress
    print(f"\n[MATERIAL DILUTION TEST] -> {'PASS' if material_passed else 'FAIL'}")
    print(f" - PF improvement > 1.5: {pf_improvement:.3f} (Global: {global_pf:.3f} -> W/O PDHL: {without_pdh_pdl['pf']:.3f}) ({is_material_pf})")
    print(f" - Net value destruction under stress: {stress_global['total_r']:.2f}R vs W/O PDHL {stress_without_pdh_pdl['total_r']:.2f}R ({is_material_stress})")

    # Final Decision
    print("\n==========================================================")
    if core_passed and drag_passed and material_passed:
        decision = "OPEN_SCBI_CORE_RESEARCH_BRANCH"
    else:
        decision = "KEEP_GLOBAL_LINE_UNCHANGED"
    
    print(f"FINAL DECISION: {decision}")
    print("==========================================================")

    # Save outputs for audit
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_dict = {
        "tests": {
            "core_edge_passed": core_passed,
            "structural_drag_passed": drag_passed,
            "material_dilution_passed": material_passed
        },
        "metrics": {
            "london_profit_share": round(london_r / total_r, 3),
            "london_pf": round(london_pf, 3),
            "london_min_regime_pf": round(london_min_pf, 3),
            "pdh_pdl_trade_share": round(pdh_pdl_n / total_trades, 3),
            "pdh_pdl_profit_share": round(pdh_pdl_r / total_r, 3),
            "pf_improvement_without_pdhl": round(pf_improvement, 3),
            "stress_r_global": round(stress_global['total_r'], 2),
            "stress_r_without_pdhl": round(stress_without_pdh_pdl['total_r'], 2)
        },
        "decision": decision
    }
    with open(OUTPUT_DIR / "core_branch_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)

if __name__ == "__main__":
    main()
