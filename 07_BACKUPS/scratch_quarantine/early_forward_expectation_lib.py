import json
import os
from pathlib import Path

def load_envelopes():
    PROJECT_ROOT = Path(__file__).parent.parent
    path = PROJECT_ROOT / "results" / "SCBI_EARLY_FORWARD_EXPECTATION_ENVELOPES.json"
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

def classify_metric(line_name, n, metric_name, value):
    envelopes = load_envelopes()
    if not envelopes or line_name not in envelopes:
        return "EXPECTATION_MODEL_NOT_RELIABLE"
    
    # Encontrar el checkpoint más cercano (hacia abajo)
    checkpoints = sorted([int(k) for k in envelopes[line_name].keys()])
    closest_n = None
    for cp in checkpoints:
        if n >= cp:
            closest_n = cp
        else:
            break
            
    if closest_n is None:
        return "NOT_ENOUGH_SAMPLE"
    
    env = envelopes[line_name][str(closest_n)]
    if metric_name not in env:
        return "EXPECTATION_MODEL_NOT_RELIABLE"
    
    p = env[metric_name]
    # Invertimos lógica para métricas donde "más es peor" (DD, Losing Streak)
    is_inverse = metric_name in ["max_dd", "max_losing_streak"]
    
    v = float(value)
    
    if is_inverse:
        # Para DD, -5 es menor que -2. P1 es el peor DD.
        if v <= p["0.01"]: return "OUTSIDE_EXPECTATION_ENVELOPE"
        if v <= p["0.05"]: return "EARLY_WARNING"
        if v <= p["0.25"]: return "STRETCHED_BUT_STILL_NORMAL"
        if v >= p["0.75"]: return "WITHIN_EXPECTATION_ENVELOPE" # Muy bueno
        return "WITHIN_EXPECTATION_ENVELOPE"
    else:
        # Para PF, Expectancy, WR. Mas es mejor.
        if v <= p["0.01"]: return "OUTSIDE_EXPECTATION_ENVELOPE"
        if v <= p["0.05"]: return "EARLY_WARNING"
        if v <= p["0.25"]: return "STRETCHED_BUT_STILL_NORMAL"
        if v >= p["0.75"]: return "WITHIN_EXPECTATION_ENVELOPE" # Muy bueno
        return "WITHIN_EXPECTATION_ENVELOPE"

def get_line_status(line_name, n, metrics_dict):
    """
    Recibe un dict con pf, expectancy, max_dd, win_rate, max_losing_streak
    Retorna la clasificacion mas conservadora (peor)
    """
    results = []
    for m_name, val in metrics_dict.items():
        status = classify_metric(line_name, n, m_name, val)
        results.append(status)
        
    order = [
        "OUTSIDE_EXPECTATION_ENVELOPE",
        "EARLY_WARNING",
        "STRETCHED_BUT_STILL_NORMAL",
        "EXPECTATION_MODEL_NOT_RELIABLE",
        "NOT_ENOUGH_SAMPLE",
        "WITHIN_EXPECTATION_ENVELOPE"
    ]
    
    for o in order:
        if o in results:
            return o
    return "NOT_ENOUGH_SAMPLE"
