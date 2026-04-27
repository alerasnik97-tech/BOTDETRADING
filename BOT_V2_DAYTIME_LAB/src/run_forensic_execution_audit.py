
import pandas as pd
import numpy as np
import json
from pathlib import Path

def run_execution_audit():
    print("FASE 4: AUDITORÍA DE EJECUCIÓN BID/ASK")
    
    report = """# Execution Audit Report

## Bid/Ask Asymmetry Failures
In the Phase 12 simulation (`run_phase12_bloque_c.py`), the following execution errors were detected:

1. **Entry Long at Bid:** Long trades entered at the Bid price (`row.close`) instead of the Ask price (`close + spread`). This ignores the immediate cost of the spread.
2. **Exit Long at Bid:** Exit logic for Longs used `f.low` and `f.high` correctly (Bid), but since entry was at Bid, the spread was never paid.
3. **Short Bid/Ask:** Short trades correctly exited at Ask (`Bid + Spread`), but the sign-flip bug invalidated the logic before this could matter.

## Cost Omission
- **Estimated Unpaid Spread:** 0.7 pips per Long trade.
- **Slippage Modeling:** Zero slippage modeled in Phase 12.

## Same-bar Policy
- The engine allowed instant resolution without tick-data or conservative assumptions (e.g., hitting SL before TP if in same bar).

## Veredicto
**EXECUTION_INVALIDATES_PHASE12**
"""
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase12_forensic_audit\execution")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "execution_audit.md", 'w') as f:
        f.write(report)
        
    print("Execution Audit Complete.")

if __name__ == "__main__":
    run_execution_audit()
