
import pandas as pd
from pathlib import Path
import json

def run_manual_alignment():
    print("Running Phase 18 Manual Alignment...")
    # Load manual data
    m_df = pd.read_csv(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\DATA MANUAL\analytics (1).csv")
    m_df['time_naive'] = pd.to_datetime(m_df['dateStart']).dt.tz_localize(None)
    m_df['dir_bot'] = m_df['side'].apply(lambda x: 'LONG' if x.lower() == 'buy' else 'SHORT')
    m_df['res_audit'] = 'SL'
    m_df.loc[m_df['rPnL'] > 100, 'res_audit'] = 'TP'
    m_df.loc[(m_df['rPnL'] > -50) & (m_df['rPnL'] < 50), 'res_audit'] = 'BE'
    
    # Load Phase 18 signals
    p_df = pd.read_csv(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase18_h1_fractal_sweep\screening\phase18_screening_results.csv")
    p_df['time_naive'] = pd.to_datetime(p_df['time'], utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    
    matches = []
    
    for idx, m_trade in m_df.iterrows():
        m_time = m_trade['time_naive']
        m_dir = m_trade['dir_bot']
        
        # Look for signals in a 2-hour window around manual entry
        match = p_df[
            (p_df['time_naive'] >= m_time - pd.Timedelta(hours=1)) &
            (p_df['time_naive'] <= m_time + pd.Timedelta(hours=1)) &
            (p_df['direction'] == m_dir)
        ]
        
        if not match.empty:
            p_sig = match.iloc[0]
            matches.append({
                "trade_id": m_trade['id'],
                "manual_result": m_trade['res_audit'],
                "p18_match": True,
                "p18_result": p_sig['result'],
                "sweep_level": p_sig['sweep_level']
            })
        else:
            matches.append({
                "trade_id": m_trade['id'],
                "manual_result": m_trade['res_audit'],
                "p18_match": False
            })
            
    df_res = pd.DataFrame(matches)
    
    # Summary
    total_manual = len(m_df)
    matched_count = len(df_res[df_res['p18_match'] == True])
    capture_rate = round(matched_count / total_manual * 100, 2)
    
    # Manual winners (TP) capture rate
    m_tps = df_res[df_res['manual_result'] == 'TP']
    tp_capture_rate = round(len(m_tps[m_tps['p18_match'] == True]) / len(m_tps) * 100, 2)
    
    summary = {
        "total_manual_trades": total_manual,
        "matched_with_p18": matched_count,
        "capture_rate_pct": capture_rate,
        "manual_tp_capture_rate_pct": tp_capture_rate,
        "level_frequencies": df_res['sweep_level'].value_counts().to_dict()
    }
    
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase18_h1_fractal_sweep\manual_alignment")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_dir / "phase18_manual_alignment.csv", index=False)
    
    with open(out_dir / "phase18_manual_alignment_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"Alignment Complete. Capture Rate: {capture_rate}%")
    print(f"TP Capture Rate: {tp_capture_rate}%")

if __name__ == "__main__":
    run_manual_alignment()
