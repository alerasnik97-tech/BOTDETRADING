import pandas as pd
import os
from datetime import datetime
from shadow_line_lab.daily_stack import config

def build_incubation_summary(last_run_data):
    # Cargar histórico si existe
    if os.path.exists(config.OPERATIONAL_LOG):
        history = pd.read_csv(config.OPERATIONAL_LOG)
    else:
        history = pd.DataFrame([last_run_data])

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "total_shadow_runs": len(history),
        "total_shadow_trades": int(history['shadow_trade_count'].sum()),
        "current_tribunal_verdict": last_run_data["tribunal_verdict"],
        "cumulative_R": last_run_data["cumulative_R"],
        "max_drawdown_R": last_run_data["drawdown_R"],
        "escalating_or_not": last_run_data["tribunal_verdict"] == "SHADOW_ESCALATION_CANDIDATE",
        "recommendation": determine_recommendation(last_run_data),
        "metrics": last_run_data["metrics"],
        "alert_history_count": int(history['alert_count'].sum())
    }
    return summary

def determine_recommendation(data):
    v = data["tribunal_verdict"]
    if v == "SHADOW_ESCALATION_CANDIDATE":
        return "Proceder a revisión de gate institucional para posible Shadow Pilot."
    elif v == "SHADOW_HOLD":
        return "DETENER INCUBACIÓN. Revisar errores estructurales o deterioro masivo."
    elif v == "SHADOW_WARNING":
        return "Mantener vigilancia estrecha. Desviación detectada."
    else:
        return "Continuar incubación para recolectar muestra (N=20)."
