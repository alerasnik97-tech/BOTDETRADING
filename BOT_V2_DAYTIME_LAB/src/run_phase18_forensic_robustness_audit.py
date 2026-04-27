
import pandas as pd
from pathlib import Path

def run_robustness_audit():
    print("Fase 8: Auditoría de Robusteza...")
    p_df = pd.read_csv(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase18_forensic_audit\reproduction\phase18_reproduced_trades.csv")
    p_df['time'] = pd.to_datetime(p_df['time'], utc=True)
    p_df['year'] = p_df['time'].dt.year
    
    years = sorted(p_df['year'].unique())
    results = []
    for year in years:
        y_df = p_df[p_df['year'] == year]
        tp_c = len(y_df[y_df['res'] == 'TP'])
        sl_c = len(y_df[y_df['res'] == 'SL'])
        pf = round((tp_c * 2.0) / sl_c, 2) if sl_c > 0 else 0
        results.append({"year": year, "sample": len(y_df), "pf": pf})
        
    res_df = pd.DataFrame(results)
    print(res_df)
    
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase18_forensic_audit\robustness")
    out_dir.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_dir / "phase18_forensic_robustness_by_year.csv", index=False)

if __name__ == "__main__":
    run_robustness_audit()
