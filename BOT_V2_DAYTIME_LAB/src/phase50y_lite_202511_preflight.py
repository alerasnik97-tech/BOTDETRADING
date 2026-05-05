import pandas as pd
import os

CSV_PATH = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase38_manipulante_deep_explainer\csv\phase38_raw_trades_enriched.csv"

def preflight_202511():
    df = pd.read_csv(CSV_PATH)
    df['ym'] = pd.to_datetime(df['entry_time'], utc=True).dt.strftime('%Y-%m')
    dfm = df[df['ym'] == '2025-11']
    print(f"Sample: {len(dfm)}")
    if len(dfm) > 0:
        print(f"Total R Bar: {dfm['r_result'].sum():.4f}")
        print("Outcomes:")
        print(dfm['outcome'].value_counts())
        print(f"TIME_EXIT: {(dfm['outcome']=='TIME_EXIT').sum()}")

if __name__ == "__main__":
    preflight_202511()
