import hashlib
import json
import os
from datetime import datetime

BASE_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
RAW_TRADES_PATH = os.path.join(BASE_DIR, "BOT_V2_DAYTIME_LAB", "outputs", "phase38_manipulante_deep_explainer", "csv", "phase38_raw_trades_enriched.csv")
TICK_DIR = r"C:\Users\alera\Desktop\Bot\BOT_MARKET_DATA\tick\EURUSD\monthly"
OFFICIAL_MONTHS = ["2024_05", "2024_06", "2024_07", "2024_08", "2024_10", "2024_11", "2025_01", "2025_03", "2025_07"]

def get_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

freeze_data = {
    "audit_timestamp": datetime.now().isoformat(),
    "raw_trades": {
        "path": RAW_TRADES_PATH,
        "sha256": get_sha256(RAW_TRADES_PATH) if os.path.exists(RAW_TRADES_PATH) else "MISSING"
    },
    "tick_parquets": {},
    "phase50p_reports": {
        "full_batch_lat1": os.path.join(BASE_DIR, "BOT_V2_DAYTIME_LAB", "reports", "manipulante_tick_historical", "PHASE50P_FULL_BATCH_CERTIFIED_LAT_1S.csv"),
        "full_batch_stress": os.path.join(BASE_DIR, "BOT_V2_DAYTIME_LAB", "reports", "manipulante_tick_historical", "PHASE50P_FULL_BATCH_CERTIFIED_LAT_1S_COST_0.2R.csv")
    },
    "official_months": OFFICIAL_MONTHS
}

for month in OFFICIAL_MONTHS:
    parquet_path = os.path.join(TICK_DIR, f"EURUSD_ticks_{month}.parquet")
    if os.path.exists(parquet_path):
        freeze_data["tick_parquets"][month] = {
            "path": parquet_path,
            "sha256": get_sha256(parquet_path)
        }
    else:
        freeze_data["tick_parquets"][month] = "MISSING"

output_path = os.path.join(BASE_DIR, "BOT_V2_DAYTIME_LAB", "reports", "manipulante_tick_historical", "PHASE50Q_INPUT_FREEZE.json")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(freeze_data, f, indent=4)

print(f"Input freeze created at: {output_path}")
