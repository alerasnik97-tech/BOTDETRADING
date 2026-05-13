import pytest
from src.v7_engine.cost_model import CostModel, CostModelConfig, CostCalculationBlockedError

@pytest.fixture
def ftmo_config():
    return CostModelConfig(
        mode="ftmo",
        commission_per_lot_round_turn=5.0,
        instrument="EURUSD",
        pip_value_per_standard_lot_usd=10.0,
        risk_per_trade_pct=1.0,
        slippage_pips=0.0
    )

def test_commission_r_10_pip_sl_is_005r(ftmo_config):
    model = CostModel(ftmo_config)
    res = model.apply_costs_to_trade(gross_r=1.0, sl_pips=10.0, risk_per_trade_cash=1000.0)
    assert res["commission_r"] == 0.05
    assert res["net_r"] == 0.95

def test_commission_r_5_pip_sl_is_010r(ftmo_config):
    model = CostModel(ftmo_config)
    res = model.apply_costs_to_trade(gross_r=1.0, sl_pips=5.0, risk_per_trade_cash=1000.0)
    assert res["commission_r"] == 0.10
    assert res["net_r"] == 0.90

def test_commission_r_20_pip_sl_is_0025r(ftmo_config):
    model = CostModel(ftmo_config)
    res = model.apply_costs_to_trade(gross_r=1.0, sl_pips=20.0, risk_per_trade_cash=1000.0)
    assert res["commission_r"] == 0.025
    assert res["net_r"] == 0.975

def test_commission_independent_of_account_size_when_risk_pct_fixed(ftmo_config):
    model = CostModel(ftmo_config)
    # Risk USD 1000
    res1 = model.apply_costs_to_trade(gross_r=1.0, sl_pips=10.0, risk_per_trade_cash=1000.0)
    # Risk USD 2000
    res2 = model.apply_costs_to_trade(gross_r=1.0, sl_pips=10.0, risk_per_trade_cash=2000.0)
    assert res1["commission_r"] == 0.05
    assert res2["commission_r"] == 0.05

def test_tp_trade_net_r_deducts_commission(ftmo_config):
    model = CostModel(ftmo_config)
    res = model.apply_costs_to_trade(gross_r=2.0, reason="TP", sl_pips=10.0, risk_per_trade_cash=1000.0)
    assert res["gross_r"] == 2.0
    assert res["net_r"] == 1.95

def test_sl_trade_net_r_deducts_commission(ftmo_config):
    model = CostModel(ftmo_config)
    res = model.apply_costs_to_trade(gross_r=-1.0, reason="SL", sl_pips=10.0, risk_per_trade_cash=1000.0)
    assert res["gross_r"] == -1.0
    assert res["net_r"] == -1.05

def test_be_trade_net_r_negative_after_commission(ftmo_config):
    model = CostModel(ftmo_config)
    res = model.apply_costs_to_trade(gross_r=0.0, reason="BE-SL", sl_pips=10.0, risk_per_trade_cash=1000.0)
    assert res["gross_r"] == 0.0
    assert res["net_r"] == -0.05

def test_forced_exit_deducts_commission(ftmo_config):
    model = CostModel(ftmo_config)
    res = model.apply_costs_to_trade(gross_r=0.5, reason="TIME", sl_pips=10.0, risk_per_trade_cash=1000.0)
    assert res["net_r"] == 0.45

def test_rejected_signal_has_no_commission(ftmo_config):
    # Simula el lifecycle del engine donde una señal es rechazada (ej. Throttler o News)
    model = CostModel(ftmo_config)
    
    class DummyEngine:
        def __init__(self, cost_model):
            self.cost_model = cost_model
            self.apply_costs_calls = 0
            
        def process_signal(self, signal_valid: bool):
            if not signal_valid:
                return {"status": "REJECTED", "reason": "THROTTLER"}
            
            self.apply_costs_calls += 1
            return self.cost_model.apply_costs_to_trade(gross_r=1.0, sl_pips=10.0, risk_per_trade_cash=1000.0)
            
    engine = DummyEngine(model)
    res = engine.process_signal(signal_valid=False)
    
    assert res["status"] == "REJECTED"
    assert "net_r" not in res
    assert engine.apply_costs_calls == 0

def test_nofill_has_no_commission(ftmo_config):
    # Simula NoFill (ej. slippage excede límite o gap de liquidez)
    model = CostModel(ftmo_config)
    
    class DummyExecution:
        def __init__(self, cost_model):
            self.cost_model = cost_model
            self.apply_costs_calls = 0
            
        def execute_trade(self, can_fill: bool):
            if not can_fill:
                raise ValueError("NoFillError: No valid ticks found.")
            
            self.apply_costs_calls += 1
            return self.cost_model.apply_costs_to_trade(gross_r=1.0, sl_pips=10.0, risk_per_trade_cash=1000.0)
            
    execution = DummyExecution(model)
    
    with pytest.raises(ValueError, match="NoFillError"):
        execution.execute_trade(can_fill=False)
        
    assert execution.apply_costs_calls == 0

def test_pf_net_uses_net_r_not_gross_r(ftmo_config):
    model = CostModel(ftmo_config)
    
    # gross_r: +2.0, -1.0
    t1 = model.apply_costs_to_trade(gross_r=2.0, sl_pips=10.0, risk_per_trade_cash=1000.0) # net_r = 1.95
    t2 = model.apply_costs_to_trade(gross_r=-1.0, sl_pips=10.0, risk_per_trade_cash=1000.0) # net_r = -1.05
    
    gross_pf = 2.0 / 1.0  # 2.0
    
    net_wins = sum([t["net_r"] for t in [t1, t2] if t["net_r"] > 0])
    net_losses = abs(sum([t["net_r"] for t in [t1, t2] if t["net_r"] < 0]))
    net_pf = net_wins / net_losses if net_losses > 0 else 0
    
    assert gross_pf == 2.0
    assert round(net_pf, 4) == round(1.95 / 1.05, 4)
    assert net_pf < gross_pf

def test_expectancy_net_uses_net_r(ftmo_config):
    model = CostModel(ftmo_config)
    
    # gross_r: +2.0, -1.0, 0.0
    t1 = model.apply_costs_to_trade(gross_r=2.0, sl_pips=10.0, risk_per_trade_cash=1000.0) # net_r = 1.95
    t2 = model.apply_costs_to_trade(gross_r=-1.0, sl_pips=10.0, risk_per_trade_cash=1000.0) # net_r = -1.05
    t3 = model.apply_costs_to_trade(gross_r=0.0, sl_pips=10.0, risk_per_trade_cash=1000.0) # net_r = -0.05
    
    gross_expectancy = (2.0 - 1.0 + 0.0) / 3 # 0.3333
    
    net_expectancy = sum([t["net_r"] for t in [t1, t2, t3]]) / 3
    
    assert round(gross_expectancy, 4) == 0.3333
    assert round(net_expectancy, 4) == round((1.95 - 1.05 - 0.05) / 3, 4)
    assert net_expectancy < gross_expectancy

def test_commission_not_applied_twice(ftmo_config):
    model = CostModel(ftmo_config)
    
    res = model.apply_costs_to_trade(gross_r=2.0, sl_pips=10.0, risk_per_trade_cash=1000.0)
    
    # Comisión FTMO esperada por $1000 arriesgados con sl 10 pips = 0.05R
    assert res["commission_r"] == 0.05
    
    # Verificamos net_r final. Si se aplicó doble sería 2.0 - 0.10 = 1.90. Debe ser 1.95.
    assert res["net_r"] == 1.95
    assert res["net_r"] == res["gross_r"] - res["commission_r"] - res["slippage_r"]


def test_ftmo_commission_basis_round_turn_vs_per_side():
    """Claude H6: Confirm USD 5/lot is round-turn, not per-side."""
    from src.v7_engine.cost_model import CostModel, CostModelConfig
    # Round-turn: 5 USD / (10 pips * 10 USD/pip) = 0.05R
    cfg_rt = CostModelConfig(mode="ftmo", commission_per_lot_round_turn=5.0, pip_value_per_standard_lot_usd=10.0)
    cm_rt = CostModel(cfg_rt)
    res_rt = cm_rt.apply_costs_to_trade(gross_r=2.0, sl_pips=10.0, risk_per_trade_cash=1000.0)
    assert abs(res_rt["commission_r"] - 0.05) < 1e-6, f"Round-turn should be 0.05R, got {res_rt['commission_r']}"
    assert abs(res_rt["net_r"] - 1.95) < 1e-6

    # Per-side equivalent (10 USD round-trip) = 0.10R
    cfg_ps = CostModelConfig(mode="ftmo", commission_per_lot_round_turn=10.0, pip_value_per_standard_lot_usd=10.0)
    cm_ps = CostModel(cfg_ps)
    res_ps = cm_ps.apply_costs_to_trade(gross_r=2.0, sl_pips=10.0, risk_per_trade_cash=1000.0)
    assert abs(res_ps["commission_r"] - 0.10) < 1e-6, f"Per-side equivalent should be 0.10R, got {res_ps['commission_r']}"
    assert abs(res_ps["net_r"] - 1.90) < 1e-6

    # Default documented = round_turn (field name confirms)
    assert "round_turn" in CostModelConfig.__dataclass_fields__["commission_per_lot_round_turn"].name
