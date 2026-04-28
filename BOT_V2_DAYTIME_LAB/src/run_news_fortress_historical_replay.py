
import pandas as pd
from datetime import datetime, timezone
from news_fortress.news_fortress_gate import NewsFortressGate
import os
from pathlib import Path

def run_historical_replay():
    print("Fase 12: Historical Replay of News Fortress Blocks...")
    
    trades_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase19_phase18_expansion\baseline_reproduction\phase18_baseline_reproduced_trades.csv"
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
    
    replay_results = []
    blocked_count = 0
    
    for idx, trade in df_trades.iterrows():
        trade_time = trade['time']
        allow, reason = gate.evaluate_trading_permission(trade_time)
        
        replay_results.append({
            "time": trade_time.isoformat(),
            "decision": "ALLOW" if allow else "BLOCK",
            "reason": reason,
            "original_res": trade['res']
        })
        
        if not allow:
            blocked_count += 1
            
    res_df = pd.DataFrame(replay_results)
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\news_fortress_live_gate\historical_replay")
    out_dir.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_dir / "phase18_news_fortress_replay.csv", index=False)
    
    total = len(df_trades)
    summary = {
        "total_signals": total,
        "blocked_by_news": blocked_count,
        "allowed_signals": total - blocked_count,
        "block_percentage": round((blocked_count / total) * 100, 2) if total > 0 else 0
    }
    
    with open(out_dir / "news_fortress_replay_summary.json", 'w') as f:
        import json
        json.dump(summary, f, indent=2)
        
    print(f"Replay complete. Blocked {blocked_count} out of {total} signals ({summary['block_percentage']}%).")

if __name__ == "__main__":
    run_historical_replay()
