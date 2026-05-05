import pandas as pd
import os

CSV_PATH = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase38_manipulante_deep_explainer\csv\phase38_raw_trades_enriched.csv"

def preflight_201501():
    df = pd.read_csv(CSV_PATH)
    df['ym'] = pd.to_datetime(df['entry_time'], utc=True).dt.strftime('%Y-%m')
    dfm = df[df['ym'] == '2015-01']
    print(f"Sample: {len(dfm)}")
    if len(dfm) > 0:
        total_r = dfm['r_result'].sum()
        wins = dfm[dfm['r_result'] > 0]['r_result'].sum()
        losses = abs(dfm[dfm['r_result'] < 0]['r_result'].sum())
        pf = wins / losses if losses > 0 else 99.0
        exp = dfm['r_result'].mean()
        
        print(f"Total R Bar: {total_r:.4f}")
        print(f"PF Bar: {pf:.4f}")
        print(f"Expectancy Bar: {exp:.4f}")
        print("Outcomes:")
        print(dfm['outcome'].value_counts())
        print(f"TIME_EXIT: {(dfm['outcome']=='TIME_EXIT').sum()}")

if __name__ == "__main__":
    preflight_201501()
