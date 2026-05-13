import pytest
import pandas as pd
from datetime import datetime, timezone

from src.v7_engine.engine import UnifiedV7Engine
from src.v7_engine.cost_model import CostModel, CostModelConfig, CostCalculationBlockedError
from src.v7_engine.news_filter import NewsCalendar
from src.v6_utils.execution import FillResult, NoFillError
from src.v7_engine.metrics import calculate_profit_factor, calculate_expectancy

class DummyNewsCalendar(NewsCalendar):
    def __init__(self):
        super().__init__(events=[], covered_periods=[])
    def is_covered(self, ts):
        return True

@pytest.fixture
def empty_news():
    return DummyNewsCalendar()

@pytest.fixture
def ftmo_cost():
    return CostModel(CostModelConfig(mode="ftmo", pip_value_per_standard_lot_usd=10.0, instrument="EURUSD", commission_per_lot_round_turn=5.0))

@pytest.fixture
def engine(empty_news, ftmo_cost):
    return UnifiedV7Engine(empty_news, cost_model=ftmo_cost)

def _make_ticks(prices, start_time):
    idx = [start_time + pd.Timedelta(seconds=i) for i in range(len(prices))]
    return pd.DataFrame({"bid": prices, "ask": [p + 0.0001 for p in prices]}, index=idx)

def test_engine_closed_tp_trade_has_cost_fields_and_net_r(engine):
    start = pd.Timestamp("2024-01-02 14:00:00")
    ticks = _make_ticks([1.0000, 1.0010, 1.0020, 1.0030, 1.0040], start)
    
    fill, reason = engine.execute_signal("long", start, ticks)
    assert reason in ["FILLED", "SIGNAL"]
    
    record = engine.close_position_with_costs(fill, sl_price=0.9981, tp_price=1.0021, ticks_during=ticks)
    
    assert record.valid_closed_trade
    assert record.gross_r > 0
    assert record.commission_r > 0
    assert "slippage_r" in dir(record)
    assert abs(record.net_r - (record.gross_r - record.commission_r - record.slippage_r)) < 1e-4
    assert len(engine.get_trade_ledger()) == 1
    
    logs = engine.get_causal_log()
    assert any(l.get("event") == "COSTS_APPLIED" for l in logs)

def test_engine_closed_sl_trade_has_net_loss_below_minus_one(engine):
    start = pd.Timestamp("2024-01-02 14:00:00")
    ticks = _make_ticks([1.0000, 0.9990, 0.9980, 0.9970], start)
    
    fill, reason = engine.execute_signal("long", start, ticks)
    
    record = engine.close_position_with_costs(fill, sl_price=0.9981, tp_price=1.0021, ticks_during=ticks)
    
    assert round(record.gross_r, 4) == -1.0
    assert record.net_r < -1.0
    assert record.commission_r > 0
    assert engine.ftmo.current_balance < engine.ftmo.initial_balance

def test_engine_be_trade_is_negative_after_ftmo_commission(engine):
    start = pd.Timestamp("2024-01-02 14:00:00")
    ticks = _make_ticks([1.0000, 1.0010, 1.0040, 0.9990, 0.9980], start)
    
    fill, _ = engine.execute_signal("long", start, ticks)
    
    record = engine.close_position_with_costs(fill, sl_price=0.9991, tp_price=1.0041, ticks_during=ticks, be_trigger_r=1.0, be_move_to_offset=0.0)
    
    assert record.exit_reason == "BE"
    assert round(record.gross_r, 4) == 0.0
    assert record.net_r < record.gross_r
    assert record.net_r < 0

def test_engine_forced_exit_trade_has_commission_once(engine):
    from datetime import time
    engine.schedule_guard.end_time = time(16, 0)
    engine.schedule_guard.forced_exit_time = time(16, 0)
    start = pd.Timestamp("2024-01-02 20:59:00") # 15:59 NY time
    ticks = _make_ticks([1.0000, 1.0010, 1.0015], start)
    ticks.loc[start + pd.Timedelta(minutes=2)] = {"bid": 1.0015, "ask": 1.0016} # 16:01 NY time
    
    fill, _ = engine.execute_signal("long", start, ticks)
    record = engine.close_position_with_costs(fill, sl_price=0.9981, tp_price=1.0041, ticks_during=ticks)
    
    assert record.exit_reason == "TIME"
    assert record.commission_r > 0
    assert abs(record.net_r - (record.gross_r - record.commission_r - record.slippage_r)) < 1e-4

def test_engine_rejected_schedule_signal_creates_no_trade_record_and_no_cost(engine):
    start = pd.Timestamp("2024-01-02 04:00:00") # 23:00 NY time
    ticks = _make_ticks([1.0000, 1.0010], start)
    
    fill, reason = engine.execute_signal("long", start, ticks)
    assert reason == "BLOCKED_BY_SCHEDULE"
    assert fill is None
    assert len(engine.get_trade_ledger()) == 0

def test_engine_rejected_news_signal_creates_no_trade_record_and_no_cost():
    from src.v7_engine.news_filter import NewsEvent
    class BlockNews(NewsCalendar):
        def __init__(self):
            super().__init__(events=[NewsEvent("1", "News", pd.Timestamp("2024-01-02 14:00:00").to_pydatetime(), "EUR", "High")], covered_periods=[])
        def is_covered(self, ts):
            return True
            
    eng = UnifiedV7Engine(BlockNews(), cost_model=CostModel(CostModelConfig(mode="conservative")), news_mode="post5")
    start = pd.Timestamp("2024-01-02 14:00:00")
    ticks = _make_ticks([1.0000, 1.0010, 1.0020], start)
    
    fill, reason = eng.execute_signal("long", start, ticks)
    assert reason == "BLOCKED_BY_NEWS"
    assert len(eng.get_trade_ledger()) == 0

def test_engine_nofill_creates_no_trade_record_and_no_cost(engine):
    start = pd.Timestamp("2024-01-02 14:00:00")
    ticks = _make_ticks([1.0000], start) 
    
    with pytest.raises(NoFillError):
        engine.execute_signal("long", start, ticks) 
        
    assert len(engine.get_trade_ledger()) == 0
    
def test_engine_ftmo_blown_rejection_has_no_cost(empty_news, ftmo_cost):
    eng = UnifiedV7Engine(empty_news, cost_model=ftmo_cost)
    eng.ftmo.current_balance = 0.0 # Blown
    
    start = pd.Timestamp("2024-01-02 14:00:00")
    ticks = _make_ticks([1.0000, 1.0010], start)
    
    fill, reason = eng.execute_signal("long", start, ticks)
    assert reason == "BLOCKED_BY_BLOWN_STATE"
    assert len(eng.get_trade_ledger()) == 0

def test_engine_metrics_from_trade_ledger_use_net_r(engine):
    start = pd.Timestamp("2024-01-02 14:00:00")
    ticks1 = _make_ticks([1.0000, 1.0010, 1.0020, 1.0030], start)
    f1, _ = engine.execute_signal("long", start, ticks1)
    engine.close_position_with_costs(f1, sl_price=0.9981, tp_price=1.0021, ticks_during=ticks1)
    
    start2 = pd.Timestamp("2024-01-02 15:00:00")
    ticks2 = _make_ticks([1.0000, 0.9990, 0.9980, 0.9970], start2)
    f2, _ = engine.execute_signal("long", start2, ticks2)
    engine.close_position_with_costs(f2, sl_price=0.9971, tp_price=1.0021, ticks_during=ticks2)
    
    ledger = engine.get_trade_ledger()
    assert len(ledger) == 2
    
    pf_net = calculate_profit_factor(ledger)
    exp_net = calculate_expectancy(ledger)
    
    pf_gross = calculate_profit_factor(ledger, "gross_r")
    
    assert pf_net < pf_gross
    assert exp_net < calculate_expectancy(ledger, "gross_r")

def test_engine_costs_block_when_sl_missing(engine):
    start = pd.Timestamp("2024-01-02 14:00:00")
    ticks = _make_ticks([1.0000, 1.0010, 1.0020], start)
    
    fill, _ = engine.execute_signal("long", start, ticks)
    
    with pytest.raises(CostCalculationBlockedError):
        engine.close_position_with_costs(fill, sl_price=1.0011, tp_price=1.0020, ticks_during=ticks)


def test_engine_ftmo_state_updates_with_net_pnl_not_gross():
    """Claude C1: Confirm FTMO state uses net_pnl_usd, not gross."""
    import pandas as pd
    from src.v7_engine.engine import UnifiedV7Engine
    from src.v7_engine.news_filter import NewsCalendar
    from src.v7_engine.cost_model import CostModel, CostModelConfig
    from src.v6_utils.execution import FillResult

    cal = NewsCalendar([])
    cost_cfg = CostModelConfig(mode="ftmo", commission_per_lot_round_turn=5.0)
    cm = CostModel(cost_cfg)
    engine = UnifiedV7Engine(news_calendar=cal, initial_balance=100000.0, cost_model=cm,
        entry_start_hour=0, entry_end_hour=23, news_mode="none", active_phase="test", test_start_year=2099)

    balance_before = engine.ftmo.current_balance

    # Create fill
    idx = pd.date_range("2024-01-15 10:00", periods=1, freq="s", tz="UTC")
    fill = FillResult(
        fill_time=idx[0], fill_price=1.10000, side="long",
        signal_time=idx[0], signal_bar_close=idx[0], reason="FILLED"
    )

    # Create ticks: entry at 1.10000, exit at TP 1.10210 (SL at 1.09900 = 10 pips, TP = 2.1R)
    tick_times = pd.date_range("2024-01-15 10:00:01", periods=100, freq="s", tz="UTC")
    prices = [1.10000 + i * 0.00003 for i in range(100)]
    prices[-1] = 1.10210  # ensure TP hit
    ticks = pd.DataFrame({"bid": prices, "ask": [p + 0.00002 for p in prices]}, index=tick_times)

    trade = engine.close_position_with_costs(fill, sl_price=1.09900, tp_price=1.10210, ticks_during=ticks)

    # Verify: gross_r != net_r (commission was applied)
    assert trade.gross_r != trade.net_r, "gross_r should differ from net_r"
    assert trade.commission_r > 0, "commission_r should be positive"

    # Verify FTMO was updated with NET pnl (not gross)
    balance_after = engine.ftmo.current_balance
    balance_change = balance_after - balance_before
    expected_net_change = trade.net_pnl_usd

    assert abs(balance_change - expected_net_change) < 0.01, (
        f"FTMO balance changed by {balance_change} but net_pnl_usd is {expected_net_change}. "
        f"If FTMO used gross, change would be {trade.gross_pnl_usd}"
    )
    # Confirm it did NOT use gross
    assert abs(balance_change - trade.gross_pnl_usd) > 0.01, "FTMO should NOT use gross_pnl_usd"


def test_slippage_stress_reduces_net_r_monotonically():
    """Claude H5: Increasing slippage must monotonically reduce net_r."""
    from src.v7_engine.cost_model import CostModel, CostModelConfig
    results = []
    for slip in [0.0, 0.1, 0.2, 0.5]:
        cfg = CostModelConfig(mode="ftmo", commission_per_lot_round_turn=5.0, slippage_pips=slip)
        cm = CostModel(cfg)
        res = cm.apply_costs_to_trade(gross_r=2.0, sl_pips=10.0, risk_per_trade_cash=1000.0)
        results.append(res["net_r"])
    # Monotonically decreasing
    for i in range(len(results) - 1):
        assert results[i] >= results[i+1], (
            f"net_r should decrease: slippage {[0.0,0.1,0.2,0.5][i]} -> {results[i]}, "
            f"slippage {[0.0,0.1,0.2,0.5][i+1]} -> {results[i+1]}"
        )
    # Slippage 0.5 pip with 10 pip SL = 0.05R additional cost
    assert abs(results[3] - (2.0 - 0.05 - 0.05)) < 1e-4, f"Expected ~1.90, got {results[3]}"
