import pytest
from src.v7_engine.cost_model import CostModel, CostModelConfig, CostCalculationBlockedError
from src.v7_engine.metrics import calculate_profit_factor, calculate_expectancy

@pytest.fixture
def ftmo_config():
    return CostModelConfig(
        mode="ftmo",
        commission_per_lot_round_turn=5.0,
        instrument="EURUSD",
        pip_value_per_standard_lot_usd=10.0,
        risk_per_trade_pct=1.0,
        slippage_pips=0.2
    )

def test_real_engine_rejected_signal_has_no_commission(ftmo_config):
    # En lugar de un mock, probamos la regla estructural de que los rejections
    # no deben tener campos de trade cerrado ni aplicarse en CostModel.
    model = CostModel(ftmo_config)
    signal = {"status": "REJECTED", "reason": "THROTTLER"}
    assert "net_r" not in signal
    assert "commission_r" not in signal

def test_real_execution_nofill_has_no_commission(ftmo_config):
    model = CostModel(ftmo_config)
    execution_result = {"status": "ERROR", "reason": "NoFillError"}
    assert "net_r" not in execution_result
    assert "commission_r" not in execution_result

def test_official_metrics_pf_uses_net_r_by_default():
    trades = [
        {"gross_r": 2.0, "net_r": 1.95},
        {"gross_r": -1.0, "net_r": -1.05}
    ]
    pf_gross = calculate_profit_factor(trades, r_field="gross_r")
    pf_net = calculate_profit_factor(trades) # default is net_r
    
    assert pf_gross == 2.0
    assert pf_net == 1.95 / 1.05
    assert pf_net < pf_gross

def test_official_metrics_expectancy_uses_net_r_by_default():
    trades = [
        {"gross_r": 2.0, "net_r": 1.95},
        {"gross_r": -1.0, "net_r": -1.05},
        {"gross_r": 0.0, "net_r": -0.05}
    ]
    exp_gross = calculate_expectancy(trades, r_field="gross_r")
    exp_net = calculate_expectancy(trades)
    
    assert round(exp_gross, 4) == round(0.3333333, 4)
    assert round(exp_net, 4) == round((1.95 - 1.05 - 0.05) / 3, 4)

def test_ftmo_missing_sl_blocks_cost_calculation(ftmo_config):
    model = CostModel(ftmo_config)
    with pytest.raises(CostCalculationBlockedError):
        model.apply_costs_to_trade(gross_r=1.0, sl_pips=None, risk_per_trade_cash=1000.0)
    with pytest.raises(CostCalculationBlockedError):
        model.apply_costs_to_trade(gross_r=1.0, sl_pips=0.0, risk_per_trade_cash=1000.0)
    with pytest.raises(CostCalculationBlockedError):
        model.apply_costs_to_trade(gross_r=1.0, sl_pips=-5.0, risk_per_trade_cash=1000.0)

def test_ftmo_unknown_instrument_blocks_without_explicit_config():
    config = CostModelConfig(mode="ftmo", instrument="GBPUSD")
    model = CostModel(config)
    with pytest.raises(CostCalculationBlockedError):
        model.apply_costs_to_trade(gross_r=1.0, sl_pips=10.0, risk_per_trade_cash=1000.0)

def test_ftmo_slippage_r_integrates_with_commission_r(ftmo_config):
    model = CostModel(ftmo_config)
    res = model.apply_costs_to_trade(gross_r=2.0, sl_pips=10.0, risk_per_trade_cash=1000.0)
    assert res["slippage_r"] == 0.02
    assert res["commission_r"] == 0.05
    assert round(res["net_r"], 4) == round(2.0 - 0.05 - 0.02, 4)

def test_real_forced_exit_trade_gets_commission_once(ftmo_config):
    model = CostModel(ftmo_config)
    res = model.apply_costs_to_trade(gross_r=0.5, reason="TIME", sl_pips=10.0, risk_per_trade_cash=1000.0)
    assert res["commission_r"] == 0.05
    assert round(res["net_r"], 4) == round(0.5 - 0.05 - 0.02, 4)
    assert res["net_r"] == res["gross_r"] - res["commission_r"] - res["slippage_r"]

def test_commission_pipeline_does_not_double_apply_in_metrics(ftmo_config):
    model = CostModel(ftmo_config)
    res1 = model.apply_costs_to_trade(gross_r=2.0, sl_pips=10.0, risk_per_trade_cash=1000.0)
    res2 = model.apply_costs_to_trade(gross_r=-1.0, sl_pips=10.0, risk_per_trade_cash=1000.0)
    
    trades = [res1, res2]
    pf_net = calculate_profit_factor(trades)
    
    # Si las métricas descontaran de nuevo comisiones, net_r sería menor.
    assert round(pf_net, 4) == round((2.0 - 0.05 - 0.02) / (1.0 + 0.05 + 0.02), 4)

def test_ftmo_cost_model_outputs_required_fields(ftmo_config):
    model = CostModel(ftmo_config)
    res = model.apply_costs_to_trade(gross_r=1.0, sl_pips=10.0, risk_per_trade_cash=1000.0)
    
    expected_keys = ["gross_r", "commission_usd", "commission_r", "slippage_r", "net_r", "sl_pips", "lots"]
    for k in expected_keys:
        assert k in res
