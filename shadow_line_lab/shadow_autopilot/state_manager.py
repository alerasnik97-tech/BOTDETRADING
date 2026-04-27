import json
import os
import pandas as pd
from shadow_line_lab.shadow_autopilot import config

def save_overall_status(state):
    # JSON
    with open(config.AUTOPILOT_STATUS_JSON, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)

def update_autopilot_log(state):
    csv_cols = [
        "run_date", "target_date", "runner_status", "tribunal_status",
        "stack_status", "checkpoint_status", "overall_status",
        "trade_count", "cumulative_R", "alert_count"
    ]
    
    new_row = {k: state.get(k) for k in csv_cols}
    
    if os.path.exists(config.AUTOPILOT_LOG_CSV):
        history = pd.read_csv(config.AUTOPILOT_LOG_CSV)
        # Idempotencia: Actualizar si ya existe la fecha
        history = history[history["target_date"] != state["target_date"]]
        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
        history.to_csv(config.AUTOPILOT_LOG_CSV, index=False)
    else:
        pd.DataFrame([new_row]).to_csv(config.AUTOPILOT_LOG_CSV, index=False)
