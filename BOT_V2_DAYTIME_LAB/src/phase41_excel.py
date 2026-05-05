import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
LAB = ROOT / "BOT_V2_DAYTIME_LAB"
OUT = LAB / "outputs" / "phase41_manipulante_hybrid_replay_forward_audit"
MANIPULANTE = ROOT / "MANIPULANTE"
EXCEL_PATH = MANIPULANTE / "14_ANALISIS" / "MANIPULANTE_HYBRID_REPLAY_FORWARD_AUDIT.xlsx"

def generate_excel():
    os.makedirs(MANIPULANTE / "14_ANALISIS", exist_ok=True)
    with pd.ExcelWriter(EXCEL_PATH, engine='openpyxl') as writer:
        # 1. Executive Summary (as a small table)
        metrics = pd.read_csv(OUT / "phase41_replay_metrics_summary.csv")
        metrics.T.to_excel(writer, sheet_name="Executive Summary")
        
        # 2. Code Snapshot
        pd.read_csv(OUT / "code_snapshot_manifest.csv").to_excel(writer, sheet_name="Code Snapshot", index=False)
        
        # 3. Recent Replay
        pd.read_csv(OUT / "decisions_like_live" / "recent_decisions.csv").to_excel(writer, sheet_name="Recent Replay", index=False)
        
        # 4. Critical Windows
        if (OUT / "decisions_like_live" / "critical_2025_02_decisions.csv").exists():
            pd.read_csv(OUT / "decisions_like_live" / "critical_2025_02_decisions.csv").to_excel(writer, sheet_name="Critical Window 2025-02", index=False)
        if (OUT / "decisions_like_live" / "best_2024_06_decisions.csv").exists():
            pd.read_csv(OUT / "decisions_like_live" / "best_2024_06_decisions.csv").to_excel(writer, sheet_name="Best Window 2024-06", index=False)
            
        # 5. Comparisons
        pd.read_csv(OUT / "comparison" / "phase41_expected_vs_replay_trades.csv").to_excel(writer, sheet_name="Expected vs Replay", index=False)
        pd.read_csv(OUT / "comparison" / "phase41_missing_trades.csv").to_excel(writer, sheet_name="Missing Trades", index=False)
        pd.read_csv(OUT / "comparison" / "phase41_extra_trades.csv").to_excel(writer, sheet_name="Extra Trades", index=False)
        pd.read_csv(OUT / "comparison" / "phase41_outcome_mismatches.csv").to_excel(writer, sheet_name="Outcome Mismatches", index=False)

    print(f"[DONE] Excel created at {EXCEL_PATH}")

import os
if __name__ == "__main__":
    generate_excel()
