import pandas as pd
import os
from shadow_line_lab.checkpoint_review import config

def update_history(review):
    history_entry = {
        "evaluation_date": review["timestamp"],
        "current_N": review["current_n"],
        "checkpoint_target": review["checkpoint_target"],
        "decision": review["decision"],
        "pf": review["metrics"].get("pf", 0),
        "expectancy_r": review["metrics"].get("expectancy_r", 0),
        "max_dd_r": review["metrics"].get("max_dd_r", 0),
        "recommendation": review["decision"]
    }
    
    df = pd.DataFrame([history_entry])
    header = not os.path.exists(config.CHECKPOINT_HISTORY_CSV)
    
    # Evitar duplicados de la misma evaluación (basado en fecha/N si hace falta)
    df.to_csv(config.CHECKPOINT_HISTORY_CSV, mode='a', index=False, header=header)
