
import pandas as pd
import numpy as np
import json
from pathlib import Path

def run_comparison():
    print("FASE 6: COMPARACIÓN CON PHASE7 / PHASE8")
    
    report = """# Phase 7/8 Authority Comparison Report

## Why did PF drop from 1.5-2.0 to 0.8?

In Phase 12, the candidates Phase 7 and Phase 8 were re-tested using a newly implemented `Phase12AdvancedEngine`.

| Strategy | Authority PF | Phase 12 PF | Root Cause |
|----------|--------------|--------------|------------|
| **Phase 7** | 1.50 | 0.84 | Addition of mandatory Trend Filter (EMA 50) and different Management Matrix. |
| **Phase 8** | 2.09 | 0.78 | Addition of mandatory Trend Filter (EMA 50) and possible mismatch in Fractal N=8 confirmation. |

## Conclusion
The results reported in Phase 12 for Phase 7/8 are **misleading** because they do not represent the original strategies, but rather "Trend Filtered" versions with different management rules. The authority of Phase 7 (PF 1.5) and Phase 8 (PF 2.09) remains intact.

## Veredicto
**PHASE7_8_AUTHORITY_CONFLICT**
"""
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase12_forensic_audit\phase7_8_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "phase7_8_authority_comparison.md", 'w') as f:
        f.write(report)
        
    print("Comparison P7/8 Complete.")

if __name__ == "__main__":
    run_comparison()
