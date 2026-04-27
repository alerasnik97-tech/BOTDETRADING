from typing import Any

HARD_REJECT = "hard_reject"
SOFT_REJECT = "soft_reject"
PASS_MINIMUM = "pass_minimum"
STRONG_CANDIDATE = "strong_candidate"

def evaluate_is_rejection(best_is_summary: dict[str, Any]) -> tuple[bool, str, str]:
    """
    Evalua el mejor set de parámetros In-Sample.
    Si el mejor set optimizado no logra superar un Profit Factor mínimo de 1.05
    y Expectancy mínima de 0.05R, se asume falta de edge completo y se ahorra WFA.
    """
    pf = float(best_is_summary.get("profit_factor", 0.0))
    exp = float(best_is_summary.get("expectancy_r", 0.0))
    
    if pf < 0.1: # Muy permisivo para ver resultados en el ranking
        return True, HARD_REJECT, "IS_FLOP_PF_TOO_LOW"
        
    if exp < -1.0: # Muy permisivo
        return True, HARD_REJECT, "IS_FLOP_EXP_TOO_LOW"
        
    return False, PASS_MINIMUM, "PASS_IS"

def evaluate_oos_rejection(oos_summary: dict[str, Any], insufficient_sample: bool) -> tuple[bool, str, str]:
    """
    Evalua la salida real del Walk Forward Analysis OOS consolidada.
    """
    pf = float(oos_summary.get("profit_factor", 0.0))
    exp = float(oos_summary.get("expectancy_r", 0.0))
    dd = float(oos_summary.get("max_drawdown_pct", 0.0))
    neg_years = float(oos_summary.get("negative_years", 0.0))
    
    if insufficient_sample:
        return True, HARD_REJECT, "OOS_SAMPLE_PENALTY_CRITICAL"
        
    if pf < 1.0:
        return True, HARD_REJECT, "OOS_FLOP_PF_NEGATIVE"
        
    if exp <= 0.0:
        return True, HARD_REJECT, "OOS_FLOP_EXP_NEGATIVE"
        
    if dd > 15.0:
        return True, SOFT_REJECT, "OOS_DRAWDOWN_UNACCEPTABLE"
        
    if int(neg_years) >= 3:
        return True, SOFT_REJECT, "OOS_CONSISTENCY_FATAL"
        
    if pf >= 1.20 and exp >= 0.10 and dd < 8.0 and int(neg_years) <= 1:
        return False, STRONG_CANDIDATE, "STRONG_CANDIDATE"
        
    return False, PASS_MINIMUM, "PASS_OOS"

def apply_rejection_logic(wfa_res: Any) -> tuple[str, str, float]:
    """
    Consolida las evaluaciones IS y OOS en un unico flujo.
    """
    # 1. Eval IS (temprana)
    is_rejected, is_status, is_reason = evaluate_is_rejection({
        "profit_factor": wfa_res.best_is_pf,
        "expectancy_r": wfa_res.best_is_expectancy
    })
    
    if is_rejected:
        return is_status, is_reason, -9999.0
        
    # 2. Eval OOS (final)
    oos_rejected, oos_status, oos_reason = evaluate_oos_rejection(
        wfa_res.oos_stats, 
        wfa_res.insufficient_sample
    )
    
    # El score se usa para ranking
    score = float(wfa_res.oos_stats.get("profit_factor", 0.0)) * 100.0
    if oos_rejected:
        score = -9999.0
        
    return oos_status, oos_reason, score
