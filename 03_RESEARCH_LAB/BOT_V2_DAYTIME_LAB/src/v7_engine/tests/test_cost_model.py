import pytest
from src.v7_engine.cost_model import CostModel, CostModelConfig, UnknownCostModeError

def test_commission_reduces_net_r():
    """1. Certifica que la comisión aplicada disminuye estrictamente el R neto transaccional."""
    model = CostModel(CostModelConfig(mode="conservative", commission_per_trade_r=0.10))
    res = model.apply_costs_to_trade(gross_r=2.00)
    assert res["gross_r"] == 2.00
    assert res["commission_r"] == 0.10
    assert res["net_r"] == 1.90

def test_pf_uses_net_r_not_gross_r():
    """2. Demuestra que el cálculo de Profit Factor futuro debe asimilar los retornos netos."""
    model = CostModel(CostModelConfig(mode="conservative", commission_per_trade_r=0.10))
    t1 = model.apply_costs_to_trade(gross_r=2.00)  # net_r = 1.90
    t2 = model.apply_costs_to_trade(gross_r=-1.00) # net_r = -1.10
    
    # PF bruto = 2.0 / 1.0 = 2.00
    # PF neto = 1.90 / 1.10 = 1.7272
    gross_pf = round(t1["gross_r"] / abs(t2["gross_r"]), 4)
    net_pf = round(t1["net_r"] / abs(t2["net_r"]), 4)
    
    assert gross_pf == 2.0000
    assert net_pf < gross_pf
    assert net_pf == round(1.90 / 1.10, 4)

def test_expectancy_uses_net_r():
    """3. Verifica que la esperanza matemática asimila la merma por fricciones fijas."""
    model = CostModel(CostModelConfig(mode="conservative", commission_per_trade_r=0.10))
    t1 = model.apply_costs_to_trade(gross_r=2.00)
    t2 = model.apply_costs_to_trade(gross_r=-1.00)
    
    mean_gross = round((t1["gross_r"] + t2["gross_r"]) / 2.0, 4) # 0.50
    mean_net = round((t1["net_r"] + t2["net_r"]) / 2.0, 4)       # (1.90 - 1.10)/2 = 0.40
    
    assert mean_net == 0.40
    assert mean_net < mean_gross

def test_zero_commission_preserves_gross_r():
    """4. Modo sin comisiones preserva milimétricamente el PnL y R brutos."""
    model = CostModel(CostModelConfig(mode="zero"))
    res = model.apply_costs_to_trade(gross_r=1.50)
    assert res["net_r"] == res["gross_r"] == 1.50
    assert res["commission_r"] == 0.0

def test_flat_commission_r_applied_once_per_closed_trade():
    """5. Deducción plana aplicada de forma determinista una sola vez por orden completada."""
    model = CostModel(CostModelConfig(mode="conservative", commission_per_trade_r=0.15))
    res = model.apply_costs_to_trade(gross_r=3.00)
    assert res["commission_r"] == 0.15
    assert res["net_r"] == 2.85

def test_cost_model_serializes_to_report():
    """6. Serialización estable del diccionario de configuración para auditoría OOS."""
    cfg = CostModelConfig(mode="conservative", commission_per_trade_r=0.08, slippage_pips=0.2)
    model = CostModel(cfg)
    data = model.serialize_config()
    assert data["mode"] == "conservative"
    assert data["commission_per_trade_r"] == 0.08
    assert data["slippage_pips"] == 0.2

def test_cost_model_blocks_unknown_mode():
    """7. Levanta excepción de forma nativa ante modos de fricción irreales o maliciosos."""
    with pytest.raises(UnknownCostModeError):
        CostModel(CostModelConfig(mode="fantasy_free_rebate"))

def test_net_metrics_differ_from_gross_when_commission_positive():
    """8. Diferenciación unívoca entre curvas de equidad brutas y netas."""
    model = CostModel(CostModelConfig(mode="conservative", commission_per_trade_r=0.10))
    res = model.apply_costs_to_trade(gross_r=1.00)
    assert res["net_r"] != res["gross_r"]

def test_be_trade_after_commission_is_not_false_profit():
    """9. Operaciones liquidadas en Break-Even arrojan una pérdida marginal neta real en lugar de beneficio neutro."""
    model = CostModel(CostModelConfig(mode="conservative", commission_per_trade_r=0.10))
    # Salida en BE aporta típicamente gross_r = 0.0
    res = model.apply_costs_to_trade(gross_r=0.00, reason="BE-SL")
    assert res["gross_r"] == 0.00
    assert res["net_r"] == -0.10

def test_loss_trade_commission_increases_loss():
    """10. Comisiones profundizan de forma estricta el impacto en el capital ante paradas de pérdidas."""
    model = CostModel(CostModelConfig(mode="conservative", commission_per_trade_r=0.10))
    res = model.apply_costs_to_trade(gross_r=-1.00, reason="SL")
    assert res["gross_r"] == -1.00
    assert res["net_r"] == -1.10
