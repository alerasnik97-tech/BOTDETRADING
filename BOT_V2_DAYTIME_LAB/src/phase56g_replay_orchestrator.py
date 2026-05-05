import os
import json
import subprocess
from datetime import datetime

# CONFIGURATION
CHECKPOINT_PATH = r"reports\manipulante_tick_historical\phase56_batches\PHASE56_FULL_HISTORICAL_CHECKPOINT.json"
OUTPUT_BASE_DIR = r"reports\manipulante_tick_historical\phase56_batches"
LIVE_STATUS_PATH = os.path.join(OUTPUT_BASE_DIR, "PHASE56_LIVE_STATUS.txt")

QUEUE = [
    {"batch_id": "batch_09", "months": ["2016-07", "2016-08"]},
    {"batch_id": "batch_10", "months": ["2016-09", "2016-10"]},
    {"batch_id": "batch_11", "months": ["2016-11", "2016-12"]},
    {"batch_id": "batch_12", "months": ["2017-01", "2017-02"]},
]

def update_live_status(month, phase, error=None):
    status = f"Timestamp: {datetime.now().isoformat()}\n"
    status += f"Current Month: {month}\n"
    status += f"Phase: {phase}\n"
    if error:
        status += f"Error: {error}\n"
    
    with open(LIVE_STATUS_PATH, 'w') as f:
        f.write(status)

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
    print(f"\n{'='*60}\nREPLAYING {m_str} ({batch_id})\n{'='*60}")
    update_live_status(m_str, "replay")
    
    year, month = m_str.split('-')
    
    # We use the specific batch folder naming requested: batch_YYYYMM_YYYYMM
    batch_info = next(b for b in QUEUE if b["batch_id"] == batch_id)
    batch_folder_name = f"batch_{batch_info['months'][0].replace('-','')}_{batch_info['months'][1].replace('-','')}"
    batch_dir = os.path.join(OUTPUT_BASE_DIR, batch_folder_name)
    os.makedirs(batch_dir, exist_ok=True)
    
    cmd = f"python src/phase56g_replay_standalone.py --year {year} --month {int(month)} --output_dir {batch_dir}"
    run_command(cmd)
    
    # Check results
    metrics_path = os.path.join(batch_dir, f"PHASE56_MONTH_{year}{month}_FTMO_NET.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            m_data = json.load(f)
        
        update_live_status(m_str, "checkpoint")
        cp = load_checkpoint()
        
        # Remove existing if any
        cp["historical_progress"] = [p for p in cp["historical_progress"] if p["month"] != m_str]
        
        entry = {
            "month": m_str,
            "sample": m_data["sample"],
            "PF_base": m_data["PF_base"],
            "expectancy_base": m_data["expectancy_base"],
            "total_R_base": m_data["total_R_base"],
            "PF_net_FTMO": m_data["PF_net_FTMO"],
            "expectancy_net_FTMO": m_data["expectancy_net_FTMO"],
            "total_R_net_FTMO": m_data["total_R_net_FTMO"],
            "avg_commission_R": m_data["avg_commission_R"],
            "verdict_net_FTMO": m_data["verdict_net_FTMO"],
            "batch_id": batch_id,
            "completed_at": datetime.now().isoformat()
        }
        cp["historical_progress"].append(entry)
        
        # Update summary
        hist = cp["historical_progress"]
        total_R_net = sum(p.get("total_R_net_FTMO", 0) for p in hist)
        total_sample = sum(p.get("sample", 0) for p in hist)
        cp["summary"] = {
            "total_months": len(hist),
            "total_sample": total_sample,
            "total_R_net_FTMO": round(total_R_net, 2),
            "expectancy_net_FTMO": round(total_R_net / total_sample, 4) if total_sample > 0 else 0
        }
        
        save_checkpoint(cp)
        print(f"[CHECKPOINT_UPDATED] {m_str}")
        update_live_status(m_str, "completed")
    else:
        print(f"[ERROR] Metrics file missing for {m_str}")
        update_live_status(m_str, "error", f"Metrics file missing at {metrics_path}")

def generate_reports():
    cp = load_checkpoint()
    relevant_months = ["2016-07", "2016-08", "2016-09", "2016-10", "2016-11", "2016-12", "2017-01", "2017-02"]
    subset = [p for p in cp["historical_progress"] if p["month"] in relevant_months]
    
    if not subset:
        print("No data for reports.")
        return

    # JSON Report
    report_json = {
        "title": "PHASE56G 8-MONTH REPLAY REPORT",
        "timestamp": datetime.now().isoformat(),
        "months": subset,
        "aggregate": {
            "total_months": len(subset),
            "total_sample": sum(p["sample"] for p in subset),
            "total_R_net_FTMO": round(sum(p["total_R_net_FTMO"] for p in subset), 2),
            "expectancy_net_FTMO": round(sum(p["total_R_net_FTMO"] for p in subset) / sum(p["sample"] for p in subset), 4) if sum(p["sample"] for p in subset) > 0 else 0
        }
    }
    
    with open(os.path.join(OUTPUT_BASE_DIR, "PHASE56G_4X2_QUEUE_REPORT.json"), 'w') as f:
        json.dump(report_json, f, indent=4)
        
    # MD Report
    md = f"# PHASE56G — 8-MONTH FORENSIC REPLAY REPORT\n\n"
    md += f"**Timestamp:** {datetime.now().isoformat()}\n\n"
    md += "| Month | Sample | PF Base | PF Net FTMO | Total R Net FTMO | Verdict |\n"
    md += "|-------|--------|---------|-------------|------------------|---------|\n"
    for p in subset:
        md += f"| {p['month']} | {p['sample']} | {p['PF_base']} | {p['PF_net_FTMO']} | {p['total_R_net_FTMO']} | {p['verdict_net_FTMO']} |\n"
    
    md += f"\n## Aggregate Results\n"
    md += f"- **Total Months:** {report_json['aggregate']['total_months']}\n"
    md += f"- **Total Sample:** {report_json['aggregate']['total_sample']}\n"
    md += f"- **Total R Net FTMO:** {report_json['aggregate']['total_R_net_FTMO']}\n"
    md += f"- **Expectancy Net FTMO:** {report_json['aggregate']['expectancy_net_FTMO']}\n"
    
    with open(os.path.join(OUTPUT_BASE_DIR, "PHASE56G_4X2_QUEUE_REPORT.md"), 'w') as f:
        f.write(md)

def main():
    print("STARTING PHASE 56G REPLAY ORCHESTRATOR")
    cp = load_checkpoint()
    
    for batch in QUEUE:
        batch_id = batch["batch_id"]
        for m_str in batch["months"]:
            # Check if already in checkpoint
            if any(p["month"] == m_str for p in cp["historical_progress"]):
                print(f"[SKIP] {m_str} already in checkpoint.")
                continue
            
            process_month(batch_id, m_str)
            cp = load_checkpoint() # Reload for next loop
            
    generate_reports()
    print("ORCHESTRATION FINISHED.")

if __name__ == "__main__":
    main()
