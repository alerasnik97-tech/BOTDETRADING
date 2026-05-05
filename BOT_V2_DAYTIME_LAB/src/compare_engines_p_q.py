import pandas as pd
import os

BASE_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
REPORTS_DIR = os.path.join(BASE_DIR, "BOT_V2_DAYTIME_LAB", "reports", "manipulante_tick_historical")

FILE_P = os.path.join(REPORTS_DIR, "PHASE50P_FULL_BATCH_CERTIFIED_LAT_1S.csv")
FILE_Q = os.path.join(REPORTS_DIR, "PHASE50Q_INDEPENDENT_TRADE_LEVEL_LAT_0_COST_0.0.csv")

def compare():
    df_p = pd.read_csv(FILE_P)
    df_q = pd.read_csv(FILE_Q)
    
    # Force trade_id to string to avoid merge issues
    df_p['trade_id'] = df_p['trade_id'].astype(str)
    df_q['trade_id'] = df_q['trade_id'].astype(str)
    # We need to make sure trade_id is consistent. 
    # In P, it's the original trade_id. In Q, I used row['trade_id'] if present.
    # Let's check the column names.
    
    # Rename columns for clarity
    df_p = df_p.rename(columns={'outcome': 'P_outcome', 'R': 'P_R', 'entry_ts': 'P_entry_ts'})
    df_q = df_q.rename(columns={'outcome': 'Q_outcome', 'R': 'Q_R', 'entry_ts': 'Q_entry_ts'})
    
    merged = pd.merge(df_p[['trade_id', 'P_outcome', 'P_R', 'P_entry_ts']], 
                      df_q[['trade_id', 'Q_outcome', 'Q_R', 'Q_entry_ts']], 
                      on='trade_id', how='inner')
    
    merged['match'] = (merged['P_outcome'] == merged['Q_outcome']) | \
                      ((merged['P_outcome'].isin(['FORCED_CLOSE', 'TIME_EXIT'])) & (merged['Q_outcome'].isin(['FORCED_CLOSE', 'TIME_EXIT'])))
    
    merged['delta_R'] = merged['P_R'] - merged['Q_R']
    
    matches = merged['match'].sum()
    total = len(merged)
    match_rate = (matches / total) * 100
    
    print(f"Total trades: {total}")
    print(f"Matches: {matches} ({match_rate:.2f}%)")
    print(f"Total Delta R (P - Q): {merged['delta_R'].sum():.4f}")
    print(f"Max Abs Delta R: {merged['delta_R'].abs().max():.4f}")
    
    merged.to_csv(os.path.join(REPORTS_DIR, "PHASE50Q_ENGINE_COMPARISON_P_VS_Q.csv"), index=False)
    
    if match_rate >= 98 and abs(merged['delta_R'].sum()) < 2.0:
        print("VEREDICTO: ENGINES AGREE")
    else:
        print("VEREDICTO: ENGINE_DISAGREEMENT_REQUIRES_REVIEW")

if __name__ == "__main__":
    compare()
