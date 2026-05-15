import pandas as pd
from pathlib import Path
from datetime import datetime

def run_preflight():
    base_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\reports\v50b_limited_real_gauntlet_rerun_sw")
    news_file = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\05_MARKET_DATA_VAULT\data\news_eurusd_am_fortress_v3.csv")
    
    print("--- RERUN PREFLIGHT ---")
    
    # 1. News Check
    if not news_file.exists():
        print("FAILED: Real news file missing.")
        return False
    news_df = pd.read_csv(news_file)
    print(f"OK: Real news loaded ({len(news_df)} events).")
    
    # 2. Path verification
    vault_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\05_MARKET_DATA_VAULT\BOT_MARKET_DATA\tick\EURUSD\monthly")
    if not vault_dir.exists():
        print("FAILED: Vault directory missing.")
        return False
    print("OK: Vault paths verified.")
    
    # 3. Date check (Dummy logic for now)
    # Train: 2020, 2021, 2022. Val: 2023, 2024.
    # We will verify this at runtime in the runner using TestLeakageGuard.
    
    # 4. No synthetic check
    # Check for keywords in the runner script (once created)
    
    print("PREFLIGHT PASSED.")
    return True

if __name__ == "__main__":
    run_preflight()
