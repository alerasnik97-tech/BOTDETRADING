
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import timedelta

def run_news_audit():
    print("Phase 7: News Audit - STRONG_CANDIDATE_PHASE7_V1")
    
    trades_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase7_forensic_audit\reproduction\reproduced_trades.csv"
    trades = pd.read_csv(trades_path)
    trades['entry_time'] = pd.to_datetime(trades['entry_time'], utc=True)
    
    manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
    with open(manifest_path, 'r') as f: manifest = json.load(f)
    
    periods = ['period_2015_2019', 'period_2020_2026']
    news_list = []
    for p in periods:
        if 'news' in manifest[p]:
            df_n = pd.read_csv(manifest[p]['news'])
            if 'timestamp_utc' in df_n.columns:
                df_n['dt'] = pd.to_datetime(df_n['timestamp_utc'], utc=True)
            else:
                df_n['dt'] = pd.to_datetime(df_n['timestamp'], utc=True)
            news_list.append(df_n)
    news_df = pd.concat(news_list)
    
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase7_forensic_audit\news")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Check trades near news
    results = []
    for _, trade in trades.iterrows():
        t_time = trade['entry_time']
        # Find nearest news
        diffs = (news_df['dt'] - t_time).abs()
        min_idx = diffs.idxmin()
        nearest = news_df.loc[min_idx]
        dist_mins = diffs.min().total_seconds() / 60
        
        results.append({
            'entry_time': t_time,
            'result': trade['result'],
            'r_value': trade['r_value'],
            'nearest_news': nearest['event'] if 'event' in nearest else 'N/A',
            'dist_mins': round(dist_mins, 1)
        })
    
    audit_df = pd.DataFrame(results)
    audit_df.to_csv(out_dir / "trades_near_news.csv", index=False)
    
    # Check if any trade is < 30 mins (Should be 0 if News Guard works)
    violations = audit_df[audit_df['dist_mins'] < 30]
    
    summary = {
        "total_trades": len(trades),
        "violations_found": len(violations),
        "min_dist_found": audit_df['dist_mins'].min(),
        "avg_dist_to_news": audit_df['dist_mins'].mean()
    }
    
    with open(out_dir / "news_audit.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"News Audit Complete. Violations: {len(violations)}")

if __name__ == "__main__":
    run_news_audit()


