import pandas as pd
import os

def tag_csv(filepath, default_provenance="BACKFILL", default_cost_mode="OFFICIAL"):
    if not os.path.exists(filepath):
        print(f"[SKIP] No encontrado: {filepath}")
        return
    
    try:
        df = pd.read_csv(filepath)
        if 'provenance' not in df.columns:
            df['provenance'] = default_provenance
        if 'cost_mode' not in df.columns:
            df['cost_mode'] = default_cost_mode
            
        df.to_csv(filepath, index=False)
        print(f"[OK] Etiquetado {filepath} con provenance='{default_provenance}' y cost_mode='{default_cost_mode}'")
    except Exception as e:
        print(f"[ERROR] Fallo al procesar {filepath}: {e}")

def main():
    base_dir = "results"
    tag_csv(os.path.join(base_dir, "H6_SHADOW_LEDGER_OFFICIAL.csv"), "BACKFILL", "OFFICIAL")
    tag_csv(os.path.join(base_dir, "H6_RESEARCH_VS_SHADOW_OFFICIAL.csv"), "BACKFILL", "OFFICIAL")
    tag_csv(os.path.join(base_dir, "H6_SHADOW_LEDGER_OBSERVED.csv"), "BACKFILL", "OBSERVED")
    tag_csv(os.path.join(base_dir, "H6_RESEARCH_VS_SHADOW_OBSERVED.csv"), "BACKFILL", "OBSERVED")

if __name__ == "__main__":
    main()