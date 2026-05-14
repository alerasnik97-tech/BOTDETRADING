import pandas as pd
import os
from pathlib import Path

BASE = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
LAB = BASE / "03_RESEARCH_LAB" / "BOT_V2_DAYTIME_LAB"
V49_DIR = LAB / "reports" / "v49_r1_real_factory_expansion_batch3"
GATE_DIR = V49_DIR / "acceptance_gate"
GATE_DIR.mkdir(parents=True, exist_ok=True)

# 1. Rowcount Audit
def audit_rowcounts():
    files = {
        "R1_V49_BATCH3_CONFIGS.csv": 101,
        "R1_V49_AGGREGATED_CANDIDATE_RANKING.csv": 501,
        "R1_V49_TOP20_CANDIDATES.csv": 21,
        "R1_V49_TOP10_CANDIDATES.csv": 11,
        "R1_V49_TOP5_FINALISTS.csv": 6
    }
    audit_data = []
    for f, expected in files.items():
        fp = V49_DIR / f
        exists = fp.exists()
        size = fp.stat().st_size if exists else 0
        rows = len(pd.read_csv(fp)) + 1 if exists else 0
        status = "PASSED" if rows == expected else "FAILED"
        audit_data.append([f, exists, size, rows, expected, status, ""])
    
    df = pd.DataFrame(audit_data, columns=["artifact", "exists", "size_bytes", "row_count", "expected_rows", "status", "notes"])
    df.to_csv(GATE_DIR / "V49_PHYSICAL_ROWCOUNT_AUDIT.csv", index=False)

# 2. Config ID Audit
def audit_config_ids():
    trades = pd.read_csv(V49_DIR / "R1_V49_AGGREGATED_TRADES.csv")
    ranking = pd.read_csv(V49_DIR / "R1_V49_AGGREGATED_CANDIDATE_RANKING.csv")
    
    unique_trade_configs = set(trades["config_id"].unique())
    unique_ranking_configs = set(ranking["config_id"].unique())
    
    mismatch = unique_trade_configs - unique_ranking_configs
    
    audit_data = [
        ["trades_have_config_id", "YES" if "config_id" in trades.columns else "NO", "PASSED"],
        ["all_trade_configs_in_ranking", "YES" if not mismatch else "NO", "PASSED" if not mismatch else "FAILED"],
        ["ranking_count", len(unique_ranking_configs), "PASSED"]
    ]
    df = pd.DataFrame(audit_data, columns=["check", "value", "status"])
    df.to_csv(GATE_DIR / "V49_CONFIG_ID_AUDIT.csv", index=False)

# 3. Date Split Audit
def audit_dates():
    trades = pd.read_csv(V49_DIR / "R1_V49_AGGREGATED_TRADES.csv")
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    
    test_leakage = trades[trades["entry_time"].dt.year >= 2025]
    
    audit_data = [
        ["min_date", trades["entry_time"].min(), "PASSED"],
        ["max_date", trades["entry_time"].max(), "PASSED"],
        ["test_leakage_count", len(test_leakage), "PASSED" if len(test_leakage) == 0 else "FAILED"]
    ]
    df = pd.DataFrame(audit_data, columns=["check", "value", "status"])
    df.to_csv(GATE_DIR / "V49_DATE_SPLIT_AUDIT.csv", index=False)

# 4. Duplicate Audit
def audit_duplicates():
    configs = pd.read_csv(V49_DIR / "R1_V49_BATCH3_CONFIGS.csv")
    dupes = configs.duplicated().sum()
    
    audit_data = [
        ["batch3_duplicate_configs", dupes, "PASSED" if dupes == 0 else "FAILED"]
    ]
    df = pd.DataFrame(audit_data, columns=["check", "value", "status"])
    df.to_csv(GATE_DIR / "V49_DUPLICATE_AUDIT.csv", index=False)

if __name__ == "__main__":
    audit_rowcounts()
    audit_config_ids()
    audit_dates()
    audit_duplicates()
    print("Audits completed.")
