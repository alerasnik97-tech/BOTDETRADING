import pandas as pd
from pathlib import Path

def audit_sunday_data():
    project_root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
    price_dirs = [
        project_root / "data_free_2020" / "prepared",
        project_root / "data_candidates_2022_2025" / "prepared"
    ]
    
    results = []
    
    for d in price_dirs:
        h1_path = d / "EURUSD_H1.csv"
        if not h1_path.exists():
            continue
            
        print(f"Auditing {h1_path}...")
        df = pd.read_csv(h1_path)
        # Assume first column is timestamp
        df['timestamp'] = pd.to_datetime(df.iloc[:, 0], utc=True)
        df['dow'] = df['timestamp'].dt.dayofweek # 0=Monday, 6=Sunday
        df['day_name'] = df['timestamp'].dt.day_name()
        
        counts = df['day_name'].value_counts().to_dict()
        sunday_count = counts.get('Sunday', 0)
        total = len(df)
        
        print(f"  Total: {total}, Sunday: {sunday_count}")
        results.append({
            "path": str(h1_path),
            "total": total,
            "sunday": sunday_count,
            "monday": counts.get('Monday', 0),
            "friday": counts.get('Friday', 0)
        })
        
        if sunday_count > 0:
            sundays = df[df['day_name'] == 'Sunday'].head(5)
            print("  Ejemplos de domingos:")
            print(sundays.iloc[:, 0].to_list())
            
    # Check if they are being filtered in the loader
    # We saw fx_market_mask in research_lab/data_loader.py
    # Let's test it
    from research_lab.data_loader import fx_market_mask
    
    for d in price_dirs:
        h1_path = d / "EURUSD_H1.csv"
        if not h1_path.exists():
            continue
        df = pd.read_csv(h1_path)
        ts = pd.to_datetime(df.iloc[:, 0], utc=True).tz_convert("US/Eastern")
        mask = fx_market_mask(ts)
        masked_df = df[mask]
        
        print(f"After fx_market_mask on {h1_path.name}:")
        print(f"  Before: {len(df)}, After: {len(masked_df)}")
        
        # Check Sunday count after mask
        masked_ts = ts[mask]
        masked_sunday_count = (masked_ts.dt.dayofweek == 6).sum()
        print(f"  Sunday count after mask: {masked_sunday_count}")

if __name__ == "__main__":
    audit_sunday_data()
