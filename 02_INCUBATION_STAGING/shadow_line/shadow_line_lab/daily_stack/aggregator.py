import pandas as pd
import os
from datetime import datetime
from shadow_line_lab.daily_stack import config

def consolidate(orch_res, trib_scorecard):
    consolidated = {
        "run_date": datetime.utcnow().isoformat() + "Z",
        "target_date": orch_res.get("date"),
        "shadow_runner_status": orch_res.get("classification"),
        "shadow_signal_found": orch_res.get("signal_found", False),
        "shadow_trade_count": 1 if orch_res.get("classification") == "TRADE_EXECUTED" else 0,
        "tribunal_verdict": trib_scorecard.get("verdict"),
        "alert_count": len(trib_scorecard.get("alerts", [])),
        "cumulative_R": trib_scorecard.get("metrics", {}).get("cumulative_R", 0.0),
        "rolling_R_N5": trib_scorecard.get("metrics", {}).get("rolling_R_N5", 0.0),
        "rolling_R_N10": trib_scorecard.get("metrics", {}).get("rolling_R_N10", 0.0),
        "drawdown_R": trib_scorecard.get("metrics", {}).get("max_drawdown_R", 0.0),
        "blockers": ", ".join(orch_res.get("blockers", [])),
        "notes": orch_res.get("notes", ""),
        "alerts": trib_scorecard.get("alerts", []),
        "metrics": trib_scorecard.get("metrics", {})
    }
    return consolidated

def update_operational_log(data):
    # Columnas canónicas del CSV
    csv_cols = [
        "run_date", "target_date", "shadow_runner_status", "shadow_signal_found",
        "shadow_trade_count", "tribunal_verdict", "alert_count", "cumulative_R",
        "rolling_R_N5", "rolling_R_N10", "drawdown_R", "blockers", "notes"
    ]
    
    new_row = {k: data[k] for k in csv_cols}
    
    if os.path.exists(config.OPERATIONAL_LOG):
        history = pd.read_csv(config.OPERATIONAL_LOG)
        # Si ya existe el target_date, eliminamos el viejo y agregamos el nuevo
        history = history[history["target_date"] != data["target_date"]]
        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
        history.to_csv(config.OPERATIONAL_LOG, index=False)
    else:
        pd.DataFrame([new_row]).to_csv(config.OPERATIONAL_LOG, index=False)
