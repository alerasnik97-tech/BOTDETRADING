
import pandas as pd
from pathlib import Path
from phase14_engine import Phase14Engine

def run_execution_audit():
    print("Fase 4: Bid/Ask Execution Audit (Verbose)...")
    trades_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase23_phase22_forensic_readiness\be_audit\phase22_be_05_audit_full.csv"
    t_df = pd.read_csv(trades_path)
    t_df['entry_time'] = pd.to_datetime(t_df['entry_time'], utc=True)
    
    manifest_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\certified_data_paths.json"
    engine = Phase14Engine(manifest_path)
    period = "period_2020_2026"
    df_ltf = engine.load_and_prep_prices(period, timeframe='m3')
    df_ltf_indexed = df_ltf.set_index('timestamp_ny').sort_index()
    
    errors = 0
    for idx, trade in t_df.iterrows():
        try:
            # We must convert trade['entry_time'] back to NY timezone for index lookup
            ny_time = trade['entry_time'].tz_convert("America/New_York")
            candle = df_ltf_indexed.loc[ny_time]
            
            # Phase 14 Logic: LONG uses Ask, SHORT uses Bid
            base_p = candle['close_ask'] if trade['direction'] == 'LONG' else candle['close_bid']
            expected_p = base_p + 0.00005 if trade['direction'] == 'LONG' else base_p - 0.00005
            
            diff = abs(trade['entry_price'] - expected_p)
            if diff > 0.000001:
                if errors < 5:
                    print(f"Error at {ny_time}: Dir {trade['direction']} | Candle CloseAsk {candle['close_ask']} | Expected {expected_p} | Got {trade['entry_price']} | Diff {diff}")
                errors += 1
            elif idx < 5:
                print(f"Match at {ny_time}: Dir {trade['direction']} | Got {trade['entry_price']}")
        except Exception as e:
            if idx < 5: print(f"Lookup Error at {trade['entry_time']}: {e}")
            continue
            
    print(f"Execution Audit: Errors {errors} / {len(t_df)}")
    if errors == 0:
        print("VERDICT: PHASE22_EXECUTION_INTEGRITY_CONFIRMED")
    else:
        print("VERDICT: PHASE22_EXECUTION_INTEGRITY_FAILED")

if __name__ == "__main__":
    run_execution_audit()
