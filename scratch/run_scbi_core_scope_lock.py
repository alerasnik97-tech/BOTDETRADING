"""
SCBI Core Scope Lock Engine

Reads structural_edge_decomposition.json and evaluates ex-ante thresholds
to determine if SCBI_CORE should be LONDON_ONLY or LONDON_PLUS_ASIA.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
INPUT_FILE = ROOT / "results" / "SCBI_2020_2025_DURABILITY" / "structural_edge_decomposition.json"
OUTPUT_DIR = ROOT / "results" / "SCBI_CORE_SCOPE_DECISION"


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Falta {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. SCBI_M5_GLOBAL intacta
    global_base = data["global_baseline"]
    
    # 2. SCBI_CORE_LONDON_ONLY
    london_only = data["ablations"]["london_only"]

    # 3. SCBI_CORE_LONDON_PLUS_ASIA (which is without_pdh_pdl)
    london_plus_asia = data["ablations"]["without_pdh_pdl"]

    # Asia metrics extraction for the criteria
    cross = data["cross_source_regime"]
    asia_min_pf = min([
        cross["asia_h_2020-2021"]["pf"], cross["asia_h_2022-2023"]["pf"], cross["asia_h_2024-2025"]["pf"],
        cross["asia_l_2020-2021"]["pf"], cross["asia_l_2022-2023"]["pf"], cross["asia_l_2024-2025"]["pf"]
    ])
    
    asia_total_r = data["contribution_by_source"]["asia_h"]["total_r"] + data["contribution_by_source"]["asia_l"]["total_r"]
    london_plus_asia_r = london_plus_asia["total_r"]
    asia_share_of_combined = asia_total_r / london_plus_asia_r

    print("==========================================================")
    print("SCBI CORE SCOPE LOCK - METRICS EVALUATION")
    print("==========================================================")

    # Threshold evaluation
    is_asia_stable = asia_min_pf > 1.4
    is_asia_material = asia_share_of_combined > 0.10
    is_combined_pf_acceptable = london_plus_asia["pf"] > 3.5

    print(f"[ASIA STABILITY TEST] -> {'PASS' if is_asia_stable else 'FAIL'}")
    print(f" - Min PF any regime > 1.4: {asia_min_pf:.3f} ({is_asia_stable})")

    print(f"\n[ASIA MATERIALITY TEST] -> {'PASS' if is_asia_material else 'FAIL'}")
    print(f" - Share of combined R > 10%: {asia_share_of_combined*100:.1f}% ({is_asia_material})")

    print(f"\n[COMBINED DILUTION TEST] -> {'PASS' if is_combined_pf_acceptable else 'FAIL'}")
    print(f" - Combined PF > 3.5: {london_plus_asia['pf']:.3f} ({is_combined_pf_acceptable})")

    # Final Decision
    print("\n==========================================================")
    if is_asia_stable and is_asia_material and is_combined_pf_acceptable:
        decision = "LOCK_SCBI_CORE_AS_LONDON_PLUS_ASIA"
    else:
        decision = "LOCK_SCBI_CORE_AS_LONDON_ONLY"
    
    print(f"FINAL DECISION: {decision}")
    print("==========================================================")

    # Print explicit comparison map
    print("\n[COMPARISON MAP]")
    print(f"1. SCBI_M5_GLOBAL: N={global_base['N']} PF={global_base['pf']:.3f} Exp={global_base['expectancy']:.3f}R DD={global_base['max_drawdown']:.2f}R Total_R={global_base['total_r']:.2f}R")
    print(f"2. LONDON_ONLY:    N={london_only['N']} PF={london_only['pf']:.3f} Exp={london_only['expectancy']:.3f}R DD={london_only['max_drawdown']:.2f}R Total_R={london_only['total_r']:.2f}R")
    print(f"3. LONDON+ASIA:    N={london_plus_asia['N']} PF={london_plus_asia['pf']:.3f} Exp={london_plus_asia['expectancy']:.3f}R DD={london_plus_asia['max_drawdown']:.2f}R Total_R={london_plus_asia['total_r']:.2f}R")

    # Save outputs for audit
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_dict = {
        "tests": {
            "asia_stable": is_asia_stable,
            "asia_material": is_asia_material,
            "combined_dilution_acceptable": is_combined_pf_acceptable
        },
        "metrics": {
            "asia_min_regime_pf": round(asia_min_pf, 3),
            "asia_share_of_combined": round(asia_share_of_combined, 3),
            "london_plus_asia_pf": round(london_plus_asia["pf"], 3)
        },
        "decision": decision
    }
    with open(OUTPUT_DIR / "core_scope_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)

if __name__ == "__main__":
    main()
