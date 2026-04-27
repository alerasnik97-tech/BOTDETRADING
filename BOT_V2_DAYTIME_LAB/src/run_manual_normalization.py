
import pandas as pd
from pathlib import Path

def normalize_manual_data():
    print("Normalizing Manual Data...")
    src_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\DATA MANUAL\analytics (1).csv"
    df = pd.read_csv(src_path)
    
    # Map Columns
    # Result mapping: TP if rPnL > 0 and RR > 1? Actually, we use 'avgRiskReward'
    # If rPnL > 0 and status is closed.
    
    df['result'] = 'SL'
    df.loc[df['rPnL'] > 100, 'result'] = 'TP' # Profit > 100 as proxy for TP
    df.loc[(df['rPnL'] > -50) & (df['rPnL'] < 50), 'result'] = 'BE'
    df.loc[df['rPnL'] < -100, 'result'] = 'SL'
    
    norm_df = pd.DataFrame({
        "trade_id": df['id'],
        "date": pd.to_datetime(df['dateStart']).dt.date,
        "entry_time_ny": pd.to_datetime(df['dateStart']),
        "direction": df['side'].str.upper(),
        "result": df['result'],
        "entry_price": df['entryPrice'],
        "sl_price": df['initalSL'],
        "tp_price": df['avgClosePrice'],
        "rr": df['avgRiskReward'],
        "session_window": "NY", # Assuming based on times
        "notes": df['tags'].fillna('')
    })
    
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\manual_edge_alignment")
    out_dir.mkdir(parents=True, exist_ok=True)
    norm_df.to_csv(out_dir / "manual_trades_normalized.csv", index=False)
    print(f"Normalization Complete. {len(norm_df)} trades saved.")

if __name__ == "__main__":
    normalize_manual_data()
