import pytest
import pandas as pd
import numpy as np

class MetricsCalculator:
    """Implementación canónica de referencia para la certificación matemática de métricas E2E."""
    
    @staticmethod
    def compute_metrics(trades: list[dict]) -> dict:
        if not trades:
            return {"pf": 0.0, "winrate": 0.0, "expectancy": 0.0, "net_r": 0.0, "max_dd": 0.0, "n_trades": 0, "be_count": 0}
            
        pnl_arr = np.array([t["pnl"] for t in trades])
        r_arr = np.array([t.get("r_multi", t["pnl"]) for t in trades])
        reasons = [t.get("reason", "") for t in trades]
        
        wins = pnl_arr[pnl_arr > 0]
        losses = pnl_arr[pnl_arr < 0]
        
        gross_wins = wins.sum() if len(wins) > 0 else 0.0
        gross_losses = abs(losses.sum()) if len(losses) > 0 else 0.0
        
        pf = round(gross_wins / gross_losses, 4) if gross_losses > 0 else (999.0 if gross_wins > 0 else 0.0)
        winrate = round(len(wins) / len(pnl_arr), 4)
        expectancy = round(pnl_arr.mean(), 4)
        net_r = round(r_arr.sum(), 4)
        
        # Max Drawdown sobre PnL acumulado
        cum_pnl = np.cumsum(pnl_arr)
        running_max = np.maximum.accumulate(cum_pnl)
        drawdowns = running_max - cum_pnl
        max_dd = round(drawdowns.max(), 4) if len(drawdowns) > 0 else 0.0
        
        be_count = sum(1 for r in reasons if "BE" in r)
        
        return {
            "pf": pf,
            "gross_wins": round(gross_wins, 4),
            "gross_losses": round(gross_losses, 4),
            "winrate": winrate,
            "expectancy": expectancy,
            "net_r": net_r,
            "max_dd": max_dd,
            "n_trades": len(trades),
            "be_count": be_count
        }

    @staticmethod
    def compute_degradation(pf_val: float, pf_test: float) -> float:
        if pf_val <= 0:
            return 0.0
        return round(pf_test / pf_val, 4)

def test_metrics_known_answer_canonical():
    """Certificación de caja blanca sobre un dataset sintético cerrado con respuestas matemáticas exactas."""
    # Definimos un vector controlado de 10 operaciones
    # 5 Ganadoras de +2.0R (PnL = +200)
    # 3 Perdedoras de -1.0R (PnL = -100)
    # 2 Salidas en Break-Even (PnL = 0)
    
    known_trades = [
        {"pnl": 200.0, "r_multi": 2.0, "reason": "TP"},
        {"pnl": 200.0, "r_multi": 2.0, "reason": "TP"},
        {"pnl": -100.0, "r_multi": -1.0, "reason": "SL"},
        {"pnl": 0.0, "r_multi": 0.0, "reason": "BE-SL"},
        {"pnl": 200.0, "r_multi": 2.0, "reason": "TP"},
        {"pnl": -100.0, "r_multi": -1.0, "reason": "SL"},
        {"pnl": 200.0, "r_multi": 2.0, "reason": "TP"},
        {"pnl": -100.0, "r_multi": -1.0, "reason": "SL"},
        {"pnl": 200.0, "r_multi": 2.0, "reason": "TP"},
        {"pnl": 0.0, "r_multi": 0.0, "reason": "BE-SL"}
    ]
    
    # Cálculos teóricos esperados:
    # Gross Wins = 5 * 200 = 1000.0
    # Gross Losses = 3 * 100 = 300.0
    # Profit Factor = 1000 / 300 = 3.3333
    # Winrate = 5 / 10 = 0.50
    # Expectancy = (1000 - 300) / 10 = 70.0
    # Net R = (5 * 2) + (3 * -1) = 7.0R
    # Break-Even count = 2
    
    calc = MetricsCalculator()
    res = calc.compute_metrics(known_trades)
    
    assert res["n_trades"] == 10
    assert res["gross_wins"] == 1000.0
    assert res["gross_losses"] == 300.0
    assert res["pf"] == 3.3333
    assert res["winrate"] == 0.50
    assert res["expectancy"] == 70.0
    assert res["net_r"] == 7.0
    assert res["be_count"] == 2

def test_metrics_drawdown_calculation():
    """Verificación rigurosa del algoritmo de máxima caída acumulada (Max DD)."""
    # Vector de PnL: +100, -50, -60, +200
    # Cum PnL: [100, 50, -10, 190]
    # Running Max: [100, 100, 100, 190]
    # Drawdowns: [0, 50, 110, 0] => Max DD = 110.0
    trades = [
        {"pnl": 100.0, "r_multi": 1.0},
        {"pnl": -50.0, "r_multi": -0.5},
        {"pnl": -60.0, "r_multi": -0.6},
        {"pnl": 200.0, "r_multi": 2.0}
    ]
    res = MetricsCalculator.compute_metrics(trades)
    assert res["max_dd"] == 110.0

def test_metrics_degradation_ratio():
    """Certifica la aserción de degradación inter-fases OOS."""
    # Retención esperada: PF TEST / PF VALIDATION
    deg = MetricsCalculator.compute_degradation(pf_val=2.00, pf_test=1.30)
    assert deg == 0.6500
