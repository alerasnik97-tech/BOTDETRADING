
import pandas as pd
import json
import os
from pathlib import Path

def audit_news_sources():
    sources = [
        {
            "id": "NEWS_PRIMARY",
            "path": r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\research_lab\data\news\news_events.csv",
            "type": "csv"
        },
        {
            "id": "NEWS_COVERAGE_PH17",
            "path": r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase17_news_feed_reliability\calendar_audit\news_calendar_coverage.csv",
            "type": "csv"
        }
    ]
    
    inventory = []
    
    for src in sources:
        path = src['path']
        if not os.path.exists(path):
            inventory.append({
                "id": src['id'],
                "path": path,
                "status": "FILE_NOT_FOUND"
            })
            continue
            
        try:
            if src['type'] == 'csv':
                df = pd.read_csv(path)
                rows = len(df)
                columns = [str(c) for c in df.columns]
                
                ts_col = 'timestamp_utc' if 'timestamp_utc' in df.columns else (df.columns[0] if len(df.columns)>0 else None)
                imp_col = 'impact_level' if 'impact_level' in df.columns else ('impact' if 'impact' in df.columns else None)
                curr_col = 'currency' if 'currency' in df.columns else None
                
                start_date = str(df[ts_col].min()) if ts_col else "N/A"
                end_date = str(df[ts_col].max()) if ts_col else "N/A"
                
                missing_ts = int(df[ts_col].isna().sum()) if ts_col else rows
                missing_curr = int(df[curr_col].isna().sum()) if curr_col else rows
                missing_imp = int(df[imp_col].isna().sum()) if imp_col else rows
                
                high_impact = int(len(df[df[imp_col].astype(str).str.upper() == 'HIGH'])) if imp_col else 0
                
                inventory.append({
                    "id": src['id'],
                    "path": path,
                    "rows": int(rows),
                    "columns": columns,
                    "start_date": start_date,
                    "end_date": end_date,
                    "missing_ts": missing_ts,
                    "missing_curr": missing_curr,
                    "missing_imp": missing_imp,
                    "high_impact_count": high_impact,
                    "status": "LOADED_OK"
                })
        except Exception as e:
            inventory.append({
                "id": src['id'],
                "path": path,
                "status": f"ERROR: {str(e)}"
            })
            
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\news_fortress_live_gate\inventory")
    pd.DataFrame(inventory).to_csv(out_dir / "news_feed_inventory.csv", index=False)
    
    with open(out_dir / "news_feed_inventory_summary.json", 'w') as f:
        json.dump(inventory, f, indent=2)
        
    md = "# NEWS FEED INVENTORY SUMMARY\n\n"
    md += "| Source ID | Path | Rows | Start | End | High Impact | Status |\n"
    md += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    for item in inventory:
        if item['status'].startswith("LOADED"):
            md += f"| {item['id']} | {item['path']} | {item.get('rows', 'N/A')} | {item.get('start_date', 'N/A')} | {item.get('end_date', 'N/A')} | {item.get('high_impact_count', 'N/A')} | {item['status']} |\n"
        else:
            md += f"| {item['id']} | {item['path']} | N/A | N/A | N/A | N/A | {item['status']} |\n"
            
    with open(out_dir / "news_feed_inventory_summary.md", 'w') as f:
        f.write(md)

if __name__ == "__main__":
    audit_news_sources()
