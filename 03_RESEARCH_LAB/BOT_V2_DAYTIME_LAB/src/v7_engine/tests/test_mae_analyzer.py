import pytest
import pandas as pd
from src.v7_engine.mae_analyzer import MaeAnalyzer

@pytest.fixture
def mock_path():
    ts = pd.to_datetime(["2026-05-12 10:00:01", "2026-05-12 10:00:02", "2026-05-12 10:00:03"])
    return pd.DataFrame({
        "bid": [1.0500, 1.0450, 1.0600],
        "ask": [1.0501, 1.0451, 1.0601]
    }, index=ts)

def test_mae_zero_when_no_adverse_excursion():
    """1. MAE nulo ante ejecuciones directas a favor de la tendencia."""
    analyzer = MaeAnalyzer()
    ts = pd.to_datetime(["2026-05-12 10:00:01", "2026-05-12 10:00:02"])
    df = pd.DataFrame({"bid": [1.0500, 1.0550], "ask": [1.0501, 1.0551]}, index=ts)
    res = analyzer.analyze_trade_ticks(1, "long", ts[0], ts[1], 1.0500, 1.0400, 1.0600, 1.0, df)
    assert res["mae_r"] == 0.0

def test_mae_calculated_correctly_for_buy(mock_path):
    """2. Cálculo exacto del peor precio Bid en transacciones de compra."""
    analyzer = MaeAnalyzer()
    t_start = mock_path.index[0]
    t_end = mock_path.index[-1]
    # Entry = 1.0500, SL = 1.0400 => r_size = 0.0100
    # Peor Bid = 1.0450 => MAE = (1.0500 - 1.0450) / 0.01 = 0.50R
    res = analyzer.analyze_trade_ticks(2, "long", t_start, t_end, 1.0500, 1.0400, 1.0600, 1.0, mock_path)
    assert res["mae_r"] == 0.50
    assert res["max_adverse_price"] == 1.0450

def test_mae_calculated_correctly_for_sell(mock_path):
    """3. Cálculo exacto del peor precio Ask en transacciones de venta."""
    analyzer = MaeAnalyzer()
    t_start = mock_path.index[0]
    t_end = mock_path.index[-1]
    # Entry = 1.0500, SL = 1.0600 => r_size = 0.0100
    # Peor Ask = 1.0601 => MAE = (1.0601 - 1.0500) / 0.01 = 1.01R
    res = analyzer.analyze_trade_ticks(3, "short", t_start, t_end, 1.0500, 1.0600, 1.0400, -1.0, mock_path)
    assert res["mae_r"] == 1.01
    assert res["max_adverse_price"] == 1.0601

def test_mfe_calculated_correctly_for_buy(mock_path):
    """4. Medición pura de ganancia máxima teórica alcanzable en compras."""
    analyzer = MaeAnalyzer()
    t_start = mock_path.index[0]
    t_end = mock_path.index[-1]
    # Entry = 1.0500, r_size = 0.01
    # Mejor Bid = 1.0600 => MFE = (1.0600 - 1.0500) / 0.01 = 1.00R
    res = analyzer.analyze_trade_ticks(4, "long", t_start, t_end, 1.0500, 1.0400, 1.0600, 1.0, mock_path)
    assert res["mfe_r"] == 1.00
    assert res["max_favorable_price"] == 1.0600

def test_mfe_calculated_correctly_for_sell(mock_path):
    """5. Medición pura de ganancia máxima teórica alcanzable en ventas."""
    analyzer = MaeAnalyzer()
    t_start = mock_path.index[0]
    t_end = mock_path.index[-1]
    # Entry = 1.0500, r_size = 0.01
    # Mejor Ask = 1.0451 => MFE = (1.0500 - 1.0451) / 0.01 = 0.49R
    res = analyzer.analyze_trade_ticks(5, "short", t_start, t_end, 1.0500, 1.0600, 1.0400, 1.0, mock_path)
    assert res["mfe_r"] == 0.49
    assert res["max_favorable_price"] == 1.0451

def test_pct_mae_gt_0_9r():
    """6. Contabilización rigurosa de concentración de estrés al límite del Stop-Loss."""
    analyzer = MaeAnalyzer()
    trades = [
        {"mae_r": 0.95, "result_net_r": 2.0, "is_winner": True, "is_loser": False},
        {"mae_r": 0.92, "result_net_r": 2.0, "is_winner": True, "is_loser": False},
        {"mae_r": 0.10, "result_net_r": 2.0, "is_winner": True, "is_loser": False},
        {"mae_r": 0.20, "result_net_r": 2.0, "is_winner": True, "is_loser": False}
    ]
    summ = analyzer.generate_summary(trades)
    assert summ["pct_mae_gt_0_9r"] == 0.50

def test_pathological_status_when_many_trades_near_stop():
    """7. Veto institucional ante muestras con colas de excursión adversa inaceptables."""
    analyzer = MaeAnalyzer(pathological_threshold_gt_09r=0.25)
    trades = [
        {"mae_r": 0.95, "result_net_r": 2.0, "is_winner": True, "is_loser": False}, # 1/3 = 33%
        {"mae_r": 0.10, "result_net_r": 2.0, "is_winner": True, "is_loser": False},
        {"mae_r": 0.20, "result_net_r": 2.0, "is_winner": True, "is_loser": False}
    ]
    summ = analyzer.generate_summary(trades)
    assert summ["status"] == "PATHOLOGICAL"
    assert summ["pass_strong_vetoed"] is True

def test_watch_status_intermediate():
    """8. Asignación del régimen de observación ante distribuciones limítrofes."""
    analyzer = MaeAnalyzer(pathological_threshold_gt_09r=0.40, watch_threshold_gt_09r=0.10)
    trades = [
        {"mae_r": 0.91, "result_net_r": 2.0, "is_winner": True, "is_loser": False}, # 1/5 = 20%
        {"mae_r": 0.10, "result_net_r": 2.0, "is_winner": True, "is_loser": False},
        {"mae_r": 0.20, "result_net_r": 2.0, "is_winner": True, "is_loser": False},
        {"mae_r": 0.30, "result_net_r": 2.0, "is_winner": True, "is_loser": False},
        {"mae_r": 0.40, "result_net_r": 2.0, "is_winner": True, "is_loser": False}
    ]
    summ = analyzer.generate_summary(trades)
    assert summ["status"] == "WATCH"
    assert summ["pass_strong_vetoed"] is False

def test_healthy_status_low_mae():
    """9. Dictamen de salud óptima ante estrategias con captura limpia del edge."""
    analyzer = MaeAnalyzer(pathological_threshold_gt_09r=0.40, watch_threshold_gt_09r=0.20)
    trades = [
        {"mae_r": 0.10, "result_net_r": 2.0, "is_winner": True, "is_loser": False},
        {"mae_r": 0.15, "result_net_r": 2.0, "is_winner": True, "is_loser": False},
        {"mae_r": 0.05, "result_net_r": 2.0, "is_winner": True, "is_loser": False}
    ]
    summ = analyzer.generate_summary(trades)
    assert summ["status"] == "HEALTHY"
    assert summ["pass_strong_vetoed"] is False

def test_mae_uses_tick_path_not_bar_low_high_shortcut(mock_path):
    """10. Certifica que la lógica explora la serie subyacente física en lugar de resumir por High/Low de barra."""
    analyzer = MaeAnalyzer()
    res = analyzer.analyze_trade_ticks(10, "long", mock_path.index[0], mock_path.index[-1], 1.0500, 1.0400, 1.0600, 1.0, mock_path)
    assert res["max_adverse_price"] == float(mock_path["bid"].min())

def test_mae_summary_serializable():
    """11. Garantiza que las salidas agregadas del analizador consisten en tipos primitivos nativos."""
    analyzer = MaeAnalyzer()
    summ = analyzer.generate_summary([])
    assert isinstance(summ["status"], str)
    assert isinstance(summ["pass_strong_vetoed"], bool)

def test_pathological_blocks_pass_strong_flag():
    """12. Aserción final de enclavamiento impidiendo que el motor promueva sistemas sin robustez adversa."""
    analyzer = MaeAnalyzer(pathological_threshold_gt_09r=0.01)
    trades = [{"mae_r": 0.99, "result_net_r": 2.0, "is_winner": True, "is_loser": False}]
    summ = analyzer.generate_summary(trades)
    assert summ["pass_strong_vetoed"] is True
