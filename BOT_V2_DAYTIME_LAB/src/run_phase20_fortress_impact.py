
import pandas as pd
from pathlib import Path
from news_fortress.news_fortress_gate import NewsFortressGate
import json
import os

def run_impact():
    print("Fase 2: Measuring News Fortress Impact on Phase 18...")
    
    trades_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase20_news_fortress_frequency_recovery\baseline\phase18_baseline_trades.csv"
    news_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\research_lab\data\news\news_events.csv"
    
    if not os.path.exists(trades_path) or not os.path.exists(news_path):
        print("ERROR: Trades or news feed not found.")
        return
        
    df_trades = pd.read_csv(trades_path)
    df_news = pd.read_csv(news_path)
    
    # Ensure UTC awareness
    df_trades['time'] = pd.to_datetime(df_trades['time'], utc=True)
    df_news['timestamp_utc'] = pd.to_datetime(df_news['timestamp_utc'], utc=True)
    
    gate = NewsFortressGate(df_news)
    
    allowed_trades = []
    blocked_trades = []
    
    for idx, trade in df_trades.iterrows():
        trade_time = trade['time']
        allow, reason = gate.evaluate_trading_permission(trade_time)
        
        trade_data = trade.to_dict()
        trade_data['reason'] = reason
        
        if allow:
            allowed_trades.append(trade_data)
        else:
            blocked_trades.append(trade_data)
            
    df_allowed = pd.DataFrame(allowed_trades)
    df_blocked = pd.DataFrame(blocked_trades)
    
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase20_news_fortress_frequency_recovery\fortress_impact")
    df_allowed.to_csv(out_dir / "phase18_with_news_fortress_trades.csv", index=False)
    df_blocked.to_csv(out_dir / "phase18_blocked_by_news_fortress.csv", index=False)
    
    # Metrics
    def calc_pf(df):
        if df.empty: return 0.0
        tp_c = len(df[df['res'] == 'TP'])
        sl_c = len(df[df['res'] == 'SL'])
        return round((tp_c * 2.0) / sl_c, 2) if sl_c > 0 else 0
        
    pf_before = calc_pf(df_trades)
    pf_after = calc_pf(df_allowed)
    
    summary = {
        "trades_original": len(df_trades),
        "trades_allowed": len(df_allowed),
        "trades_blocked": len(df_blocked),
        "pf_before": pf_before,
        "pf_after": pf_after,
        "tp_blocked": len(df_blocked[df_blocked['res'] == 'TP']),
        "sl_blocked": len(df_blocked[df_blocked['res'] == 'SL']),
        "impact_verdict": "PHASE18_FORTRESS_IMPROVES_EDGE" if pf_after > pf_before else "PHASE18_FORTRESS_PROTECTS_WITH_ACCEPTABLE_COST"
    }
    
    with open(out_dir / "phase18_news_fortress_impact_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"Impact Analysis Complete. PF Before: {pf_before} | PF After: {pf_after}")
    print(f"Blocked {len(df_blocked)} trades ({round(len(df_blocked)/len(df_trades)*100, 2)}%).")

if __name__ == "__main__":
    run_impact()
