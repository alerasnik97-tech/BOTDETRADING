import pandas as pd
import os

CSV_PATH = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase38_manipulante_deep_explainer\csv\phase38_raw_trades_enriched.csv"

def preflight_202502():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    if 'year_month' not in df.columns:
        df['year_month'] = pd.to_datetime(df['entry_time']).dt.strftime('%Y-%m')
    
    df_m = df[df['year_month'] == '2025-02']
    
    print(f"Sample: {len(df_m)}")
    if len(df_m) == 0:
        print("VERDICTO: NO_TRADES")
        return

    if 'r_result' in df_m.columns:
        total_r = df_m['r_result'].sum()
        wins = df_m[df_m['r_result'] > 0]['r_result'].sum()
        losses = abs(df_m[df_m['r_result'] < 0]['r_result'].sum())
        pf = wins / losses if losses > 0 else 99.0
        exp = df_m['r_result'].mean()
        
        print(f"Total R Bar: {total_r:.4f}")
        print(f"PF Bar: {pf:.4f}")
        print(f"Expectancy Bar: {exp:.4f}")
    
    if 'outcome' in df_m.columns:
        print("Outcomes:")
        print(df_m['outcome'].value_counts())
        print(f"TIME_EXIT count: {(df_m['outcome'] == 'TIME_EXIT').sum()}")

if __name__ == "__main__":
    preflight_202502()
