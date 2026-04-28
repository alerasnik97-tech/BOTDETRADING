
import pandas as pd
import numpy as np
from pathlib import Path
import json

def run_math_audit():
    print("Fase 3: TP/SL Math Audit...")
    trades_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase23_phase22_forensic_readiness\be_audit\phase22_be_05_audit_full.csv"
    t_df = pd.read_csv(trades_path)
    
    # R-multiple consistency
    t_df['calc_r'] = (t_df['exit_price'] - t_df['entry_price']) / t_df['risk']
    t_df.loc[t_df['direction'] == 'SHORT', 'calc_r'] = (t_df['entry_price'] - t_df['exit_price']) / t_df['risk']
    
    tp_outliers = t_df[t_df['res'] == 'TP'][abs(t_df['calc_r'] - 1.1) > 0.001]
    sl_outliers = t_df[t_df['res'] == 'SL'][abs(t_df['calc_r'] + 1.0) > 0.001]
    be_outliers = t_df[t_df['res'] == 'BE'][abs(t_df['calc_r']) > 0.001]
    
    summary = {
        "tp_math_errors": len(tp_outliers),
        "sl_math_errors": len(sl_outliers),
        "be_math_errors": len(be_outliers),
        "avg_tp_r": float(t_df[t_df['res'] == 'TP']['calc_r'].mean()) if len(t_df[t_df['res'] == 'TP']) > 0 else 0,
        "avg_sl_r": float(t_df[t_df['res'] == 'SL']['calc_r'].mean()) if len(t_df[t_df['res'] == 'SL']) > 0 else 0
    }
    
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase23_phase22_forensic_readiness\tp_sl_math")
    with open(out_dir / "phase22_tp_sl_math_audit_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Math Audit: TP Errors {len(tp_outliers)} | SL Errors {len(sl_outliers)} | BE Errors {len(be_outliers)}")
    if sum([len(tp_outliers), len(sl_outliers), len(be_outliers)]) == 0:
        print("VERDICT: PHASE22_TP_SL_MATH_CONFIRMED")
    else:
        print("VERDICT: PHASE22_TP_SL_BUG_FOUND")

if __name__ == "__main__":
    run_math_audit()
