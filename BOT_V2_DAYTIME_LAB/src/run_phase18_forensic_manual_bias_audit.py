
import pandas as pd
import numpy as np
from pathlib import Path

def run_manual_bias_audit():
    print("Fase 7: Auditoría de Sesgo Manual...")
    # Load manual data
    m_df = pd.read_csv(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\DATA MANUAL\analytics (1).csv")
    m_df['time_naive'] = pd.to_datetime(m_df['dateStart']).dt.tz_localize(None)
    m_df['dir_bot'] = m_df['side'].apply(lambda x: 'LONG' if x.lower() == 'buy' else 'SHORT')
    m_df['res_audit'] = 'SL'
    m_df.loc[m_df['rPnL'] > 100, 'res_audit'] = 'TP'
    
    # Load Phase 18 reproduced trades
    p_df = pd.read_csv(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase18_forensic_audit\reproduction\phase18_reproduced_trades.csv")
    p_df['time_naive'] = pd.to_datetime(p_df['time'], utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    
    matches = []
    for _, m_trade in m_df.iterrows():
        m_time = m_trade['time_naive']
        m_dir = m_trade['dir_bot']
        
        match = p_df[
            (p_df['time_naive'] >= m_time - pd.Timedelta(hours=1)) &
            (p_df['time_naive'] <= m_time + pd.Timedelta(hours=1)) &
            (p_df['dir'] == m_dir)
        ]
        
        if not match.empty:
            matches.append({"manual_id": m_trade['id'], "manual_res": m_trade['res_audit'], "bot_res": match.iloc[0]['res']})
            
    m_results = pd.DataFrame(matches)
    if not m_results.empty:
        tp_captured = len(m_results[m_results['manual_res'] == 'TP'])
        sl_captured = len(m_results[m_results['manual_res'] == 'SL'])
        print(f"Manual TPs Captured: {tp_captured}")
        print(f"Manual SLs Captured: {sl_captured}")
        
    # Auto-only PF
    auto_only = p_df[~p_df['time_naive'].isin(m_df['time_naive'])] # Approximation
    tp_a = len(auto_only[auto_only['res'] == 'TP'])
    sl_a = len(auto_only[auto_only['res'] == 'SL'])
    pf_auto = round((tp_a * 2.0) / sl_a, 2) if sl_a > 0 else 0
    print(f"Auto-only PF: {pf_auto}")

if __name__ == "__main__":
    run_manual_bias_audit()
