
import pandas as pd
import numpy as np
import json
from pathlib import Path

def run_comparison():
    print("FASE 5: COMPARACIÓN CON PHASE10 SELECTIVE FAKEOUT")
    
    report = """# Phase 10 vs Phase 12 Comparison Report

## Why did PF jump from 1.3 to 11.7?

The extraordinary results in Phase 12 are not due to better strategy performance, but due to **implementation degradation** and **logic bugs**.

| Feature | Phase 10 (Authority) | Phase 12 (Audit) | Impact |
|---------|-----------------------|-------------------|--------|
| **TP Calculation** | Correct (`entry + risk`) | **Inverted** (`entry - risk`) | **Massive** (Instant Wins) |
| **Bid/Ask** | Modeled in Engine | Entry at Bid (Longs) | Moderate (Unpaid Spread) |
| **OR Range** | 08:00 - 09:00 | 08:00 - 08:30 | Minor (Sample Increase) |
| **Engine** | Validated Phase 10 | New Phase 12 Engine | Source of the Bugs |

## Conclusion
The Phase 12 "Selective Fakeout V2" is a corrupted version of the original Phase 10 logic. The PF 11.71 is purely a mathematical artifact.

## Veredicto
**DIFFERENCE_INVALIDATES_PHASE12**
"""
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase12_forensic_audit\phase10_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "phase10_vs_phase12_comparison.md", 'w') as f:
        f.write(report)
        
    print("Comparison P10 Complete.")

if __name__ == "__main__":
    run_comparison()
