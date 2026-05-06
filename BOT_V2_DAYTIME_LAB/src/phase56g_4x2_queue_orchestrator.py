import os
import json
import subprocess
from datetime import datetime

# CONFIGURATION
RAW_TRADES_PATH = r"BOT_V2_DAYTIME_LAB\outputs\phase38_manipulante_deep_explainer\csv\phase38_raw_trades_enriched.csv"
CHECKPOINT_PATH = r"BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase56_batches\PHASE56_FULL_HISTORICAL_CHECKPOINT.json"
OUTPUT_BASE_DIR = r"BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\phase56_batches"

QUEUE = [
    {"id": "batch_09", "months": ["2016-07", "2016-08"]},
    {"id": "batch_10", "months": ["2016-09", "2016-10"]},
    {"id": "batch_11", "months": ["2016-11", "2016-12"]},
    {"id": "batch_12", "months": ["2017-01", "2017-02"]},
]

def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, 'r') as f:
            return json.load(f)
    return {"historical_progress": [], "summary": {}}

def save_checkpoint(cp):
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(cp, f, indent=4)

def run_command(cmd):
    print(f"[RUNNING] {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {result.stderr}")
    return result

def process_month(batch_id, m_str):
    print(f"\n{'='*60}\nPROCESSING {m_str} ({batch_id})\n{'='*60}")
    
    # 1. Extraction/Finalize (Parquet Rule)
    year, month = m_str.split('-')
    parquet_path = rf"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_MARKET_DATA\tick\EURUSD\monthly\EURUSD_ticks_{year}_{month}.parquet"
    
    if not os.path.exists(parquet_path):
        print(f"[EXTRACTING] {m_str}...")
        run_command(f"python src/phase50s_resumable_tick_extractor.py --year {year} --month {int(month)} --mode resume")
        run_command(f"python src/phase50s_resumable_tick_extractor.py --year {year} --month {int(month)} --mode finalize")
        run_command(f"python src/phase50s_resumable_tick_extractor.py --year {year} --month {int(month)} --mode validate")
    else:
        print(f"[PARQUET_FOUND] {parquet_path}")

    # 2. Replay
    # We use a batch-specific folder
    batch_folder = os.path.join(OUTPUT_BASE_DIR, f"{batch_id}_{m_str.replace('-','')}") # Simplified for 4x2
    # Actually, user wants phase56_batches\batch_YYYYMM_YYYYMM\
    # I'll use the specific batch ID to group them if possible, but let's follow the user's "phase56_batches\batch_YYYYMM_YYYYMM\" if they want.
    # The user said: phase56_batches\batch_YYYYMM_YYYYMM\
    # For Batch 09 (07 and 08), it would be batch_201607_201608.
    
    # Let's find the batch folder name
    for b in QUEUE:
        if b["id"] == batch_id:
            batch_folder_name = f"batch_{b['months'][0].replace('-','')}_{b['months'][1].replace('-','')}"
            break
    
    batch_dir = os.path.join(OUTPUT_BASE_DIR, batch_folder_name)
    os.makedirs(batch_dir, exist_ok=True)

    print(f"[REPLAYING] {m_str}...")
    # We use the existing replay logic (Phase 50) but targeted
    cmd_replay = f"python src/phase50_tick_replay_core.py --month {m_str} --output_dir {batch_dir} --commission_ftmo"
    run_command(cmd_replay)
    
    # 3. Update Checkpoint
    # Read the results from the generated JSON
    metrics_path = os.path.join(batch_dir, f"PHASE56_MONTH_{m_str.replace('-','')}_FTMO_NET.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            m_data = json.load(f)
        
        cp = load_checkpoint()
        # Remove existing if any (shouldn't be based on preflight)
        cp["historical_progress"] = [p for p in cp["historical_progress"] if p["month"] != m_str]
        
        entry = {
            "month": m_str,
            "sample": m_data.get("sample", 0),
            "PF_base": m_data.get("PF_base", 0),
            "expectancy_base": m_data.get("expectancy_base", 0),
            "total_R_base": m_data.get("total_R_base", 0),
            "PF_net_FTMO": m_data.get("PF_net_FTMO", 0),
            "expectancy_net_FTMO": m_data.get("expectancy_net_FTMO", 0),
            "total_R_net_FTMO": m_data.get("total_R_net_FTMO", 0),
            "avg_commission_R": m_data.get("avg_commission_R", 0),
            "verdict_net_FTMO": m_data.get("verdict_net_FTMO", "UNKNOWN"),
            "batch_id": batch_id,
            "completed_at": datetime.utcnow().isoformat()
        }
        cp["historical_progress"].append(entry)
        
        # Update summary
        total_R_net = sum(p.get("total_R_net_FTMO", 0) for p in cp["historical_progress"])
        total_sample = sum(p.get("sample", 0) for p in cp["historical_progress"])
        cp["summary"] = {
            "total_months": len(cp["historical_progress"]),
            "total_sample": total_sample,
            "total_R_net_FTMO": round(total_R_net, 2),
            "expectancy_net_FTMO": round(total_R_net / total_sample, 4) if total_sample > 0 else 0
        }
        
        # Batch status update
        if batch_id not in cp:
            cp[batch_id] = {"months": [], "status": "PARTIAL"}
        # Check if all months of this batch are done
        # We'll just mark it PARTIAL for now, and COMPLETED when the last month finishes
        
        save_checkpoint(cp)
        print(f"[CHECKPOINT_UPDATED] {m_str}")
    else:
        print(f"[WARNING] Metrics file not found: {metrics_path}")

def main():
    print(f"STARTING PHASE 56G — 4x2 QUEUE RUNNER")
    cp = load_checkpoint()
    
    for batch in QUEUE:
        batch_id = batch["id"]
        months = batch["months"]
        
        # Check if batch already completed
        if cp.get(batch_id, {}).get("status") == "COMPLETED":
            print(f"[SKIP] Batch {batch_id} already COMPLETED.")
            continue
            
        for m_str in months:
            # Check if month already in progress
            if any(p["month"] == m_str for p in cp["historical_progress"]):
                print(f"[SKIP] Month {m_str} already in checkpoint.")
                continue
            
            process_month(batch_id, m_str)
        
        # After processing both months in batch, mark batch as COMPLETED
        cp = load_checkpoint()
        batch_months_done = [m for m in months if any(p["month"] == m for p in cp["historical_progress"])]
        if len(batch_months_done) == len(months):
            cp[batch_id] = {
                "months": months,
                "status": "COMPLETED",
                "timestamp": datetime.utcnow().isoformat()
            }
            save_checkpoint(cp)
            print(f"[BATCH_COMPLETED] {batch_id}")

if __name__ == "__main__":
    main()
