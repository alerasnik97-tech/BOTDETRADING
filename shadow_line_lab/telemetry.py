import os
import json
from datetime import datetime
from shadow_line_lab import config

def log_event(event_type, message):
    log_file = os.path.join(config.RESULTS_DIR, "shadow_telemetry.log")
    timestamp = datetime.utcnow().isoformat() + "Z"
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {event_type.upper()}: {message}\n")

def record_heartbeat():
    heartbeat_file = os.path.join(config.RESULTS_DIR, "shadow_heartbeat.json")
    status = {
        "last_heartbeat": datetime.utcnow().isoformat() + "Z",
        "status": "OPERATIONAL",
        "mode": "SHADOW_ONLY"
    }
    with open(heartbeat_file, 'w', encoding='utf-8') as f:
        json.dump(status, f, indent=2)
