import csv
import os
import json
from datetime import datetime

def ensure_dirs(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def write_ledger_entry(entry, file_path):
    ensure_dirs(file_path)
    file_exists = os.path.exists(file_path)
    
    headers = [
        "date", "line_name", "classification", "signal_found", 
        "entry", "sl", "tp", "exit_reason", "pnl_r", 
        "timeout_flag", "news_blocked", "notes"
    ]
    
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(entry)

def save_json(data, file_path):
    ensure_dirs(file_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def load_json(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
