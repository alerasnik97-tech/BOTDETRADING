
import pandas as pd
import os
from pathlib import Path

def normalize_manual_data():
    input_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\DATA MANUAL\analytics (1).csv"
    output_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\data\manual_normalized\manual_trades_normalized.csv"
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    # Create directory if not exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_csv(input_path)
    
    norm_df = pd.DataFrame()
    norm_df['trade_id'] = df['id']
    norm_df['date'] = pd.to_datetime(df['dateStart']).dt.date
    norm_df['timestamp_entry_ny'] = pd.to_datetime(df['dateStart'])
    norm_df['timestamp_exit_ny'] = pd.to_datetime(df['dateEnd'])
    norm_df['direction'] = df['side'].str.upper()
    
    norm_df['result_R'] = df['avgRiskReward']
    norm_df['result'] = 'BE'
    norm_df.loc[norm_df['result_R'] > 0.5, 'result'] = 'TP'
    norm_df.loc[norm_df['result_R'] < -0.5, 'result'] = 'SL'
    
    norm_df['pair'] = df['pair'].str.replace('OANDA:', '')
    norm_df['swept_level'] = ""
    norm_df['h1_sweep_time'] = ""
    norm_df['entry_timeframe'] = "3M"
    norm_df['entry_model_manual'] = ""
    norm_df['tp_R'] = df['maxRiskReward']
    norm_df['sl_type'] = "LTF Structure"
    norm_df['be_used'] = ((df['avgRiskReward'] > -0.5) & (df['avgRiskReward'] < 0.5))
    norm_df['notes'] = df['tags'].fillna("")
    norm_df['source_file'] = "analytics (1).csv"
    
    norm_df.to_csv(output_path, index=False)
    print(f"Normalized data saved to {output_path}")

if __name__ == "__main__":
    normalize_manual_data()


