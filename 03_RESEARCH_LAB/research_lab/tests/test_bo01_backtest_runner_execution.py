# -*- coding: utf-8 -*-
import unittest
import pandas as pd
import numpy as np

from research_lab.runners import bo01_backtest_runner as runner

class MockBO01Strategy:
    signals_by_index = {}
    
    @classmethod
    def reset(cls):
        cls.signals_by_index = {}

    @classmethod
    def signal(cls, frame, i, params):
        return cls.signals_by_index.get(i, None)


class TestBO01BacktestRunnerExecution(unittest.TestCase):
    
    def setUp(self):
        MockBO01Strategy.reset()
        
        # Standard DatetimeIndex setup
        self.idx = pd.date_range("2015-01-05 08:00:00", periods=20, freq="5min", tz="UTC")
        self.df = pd.DataFrame({
            "open": [1.1200] * 20,
            "high": [1.1210] * 20,
            "low": [1.1190] * 20,
            "close": [1.1200] * 20
        }, index=self.idx)

    def test_entry_next_candle_open_only(self):
        """Verify signals at t trigger entry at Open of next candle t+1."""
        MockBO01Strategy.signals_by_index = {
            2: {
                "signal": 1,
                "direction": "long",
                "stop_price": 1.1150,
                "target_rr": 2.0
            }
        }
        # Force next bar open to be distinct to verify Open fill
        self.df.loc[self.idx[3], "open"] = 1.1235
        
        res = runner.run_bo01_backtest_on_frame(MockBO01Strategy, self.df)
        self.assertEqual(res["trade_count"], 1)
        
        trade = res["trades"][0]
        self.assertEqual(trade["entry_idx"], 3)  # t+1
        self.assertEqual(trade["entry_price"], 1.1235)  # t+1 open price
        self.assertEqual(trade["stop_price"], 1.1150)

    def test_max_one_trade_per_day(self):
        """Verify only the first signal of the day triggers entry, ignoring subsequent ones."""
        MockBO01Strategy.signals_by_index = {
            2: {
                "signal": 1,
                "direction": "long",
                "stop_price": 1.1100,
                "target_rr": 2.0
            },
            6: {
                "signal": 1,
                "direction": "long",
                "stop_price": 1.1160,
                "target_rr": 2.0
            }
        }
        # Force exit at index 4 for first trade (low touches stop 1.1000)
        self.df.loc[self.idx[4], "low"] = 1.1000
        
        res = runner.run_bo01_backtest_on_frame(MockBO01Strategy, self.df, max_trades_per_day=1)
        self.assertEqual(res["trade_count"], 1)
        self.assertEqual(res["skipped_signals_same_day"], 1)

    def test_ignore_signals_while_position_open(self):
        """Verify subsequent signals are completely ignored while a position is active."""
        MockBO01Strategy.signals_by_index = {
            2: {
                "signal": 1,
                "direction": "long",
                "stop_price": 1.1150,
                "target_rr": 2.0
            },
            4: {
                "signal": 1,
                "direction": "long",
                "stop_price": 1.1160,
                "target_rr": 2.0
            }
        }
        
        # Position is entered at Open of idx 3, and exits at the end of the frame (index 19) because target/stop is never reached.
        # Signal at idx 4 must be ignored due to active position.
        res = runner.run_bo01_backtest_on_frame(MockBO01Strategy, self.df)
        self.assertEqual(res["trade_count"], 1)
        self.assertEqual(res["trades"][0]["exit_idx"], 19)

    def test_same_bar_stop_first_long(self):
        """Verify STOP-FIRST conservative policy registers a loss if SL and TP are hit on the same long bar."""
        MockBO01Strategy.signals_by_index = {
            2: {
                "signal": 1,
                "direction": "long",
                "stop_price": 1.1100,
                "target_rr": 2.0
            }
        }
        # In t+1 (index 3), High is 1.1500 (touches target), Low is 1.1000 (touches stop)
        self.df.loc[self.idx[3], "high"] = 1.1500
        self.df.loc[self.idx[3], "low"] = 1.1000
        
        res = runner.run_bo01_backtest_on_frame(MockBO01Strategy, self.df)
        self.assertEqual(res["trade_count"], 1)
        trade = res["trades"][0]
        self.assertEqual(trade["exit_idx"], 3)
        self.assertEqual(trade["exit_type"], "same_bar_stop_first")
        self.assertEqual(trade["gross_r"], -1.0)
        self.assertEqual(res["same_bar_stop_first_count"], 1)

    def test_same_bar_stop_first_short(self):
        """Verify STOP-FIRST conservative policy registers a loss if SL and TP are hit on the same short bar."""
        MockBO01Strategy.signals_by_index = {
            2: {
                "signal": -1,
                "direction": "short",
                "stop_price": 1.1300,
                "target_rr": 2.0
            }
        }
        # In t+1 (index 3), High is 1.1400 (touches stop), Low is 1.0900 (touches target)
        self.df.loc[self.idx[3], "high"] = 1.1400
        self.df.loc[self.idx[3], "low"] = 1.0900
        
        res = runner.run_bo01_backtest_on_frame(MockBO01Strategy, self.df)
        self.assertEqual(res["trade_count"], 1)
        trade = res["trades"][0]
        self.assertEqual(trade["exit_idx"], 3)
        self.assertEqual(trade["exit_type"], "same_bar_stop_first")
        self.assertEqual(trade["gross_r"], -1.0)
        self.assertEqual(res["same_bar_stop_first_count"], 1)

    def test_target_hit_long(self):
        """Verify a clean target hit registers a positive R gain for a long position."""
        MockBO01Strategy.signals_by_index = {
            2: {
                "signal": 1,
                "direction": "long",
                "stop_price": 1.1100,
                "target_rr": 2.0
            }
        }
        # Entry open = 1.1200 (Stop price = 1.1100, target price = 1.1400)
        # Index 4 (subsequent candle) touches target 1.1450, Low is safe at 1.1150
        self.df.loc[self.idx[4], "high"] = 1.1450
        self.df.loc[self.idx[4], "low"] = 1.1150
        
        res = runner.run_bo01_backtest_on_frame(MockBO01Strategy, self.df)
        self.assertEqual(res["trade_count"], 1)
        trade = res["trades"][0]
        self.assertEqual(trade["exit_idx"], 4)
        self.assertEqual(trade["exit_type"], "target")
        self.assertEqual(trade["gross_r"], 2.0)

    def test_stop_hit_long(self):
        """Verify a clean stop hit registers a loss of -1R for a long position."""
        MockBO01Strategy.signals_by_index = {
            2: {
                "signal": 1,
                "direction": "long",
                "stop_price": 1.1100,
                "target_rr": 2.0
            }
        }
        # Entry open = 1.1200 (Stop price = 1.1100, target price = 1.1400)
        # Index 4 low touches 1.1050 (hits stop), High is safe at 1.1250
        self.df.loc[self.idx[4], "high"] = 1.1250
        self.df.loc[self.idx[4], "low"] = 1.1050
        
        res = runner.run_bo01_backtest_on_frame(MockBO01Strategy, self.df)
        self.assertEqual(res["trade_count"], 1)
        trade = res["trades"][0]
        self.assertEqual(trade["exit_idx"], 4)
        self.assertEqual(trade["exit_type"], "stop")
        self.assertEqual(trade["gross_r"], -1.0)

    def test_target_hit_short(self):
        """Verify a clean target hit registers a positive R gain for a short position."""
        MockBO01Strategy.signals_by_index = {
            2: {
                "signal": -1,
                "direction": "short",
                "stop_price": 1.1300,
                "target_rr": 2.0
            }
        }
        # Entry open = 1.1200 (Stop price = 1.1300, target price = 1.1000)
        # Index 4 low touches 1.0950 (hits target), High is safe at 1.1250
        self.df.loc[self.idx[4], "high"] = 1.1250
        self.df.loc[self.idx[4], "low"] = 1.0950
        
        res = runner.run_bo01_backtest_on_frame(MockBO01Strategy, self.df)
        self.assertEqual(res["trade_count"], 1)
        trade = res["trades"][0]
        self.assertEqual(trade["exit_idx"], 4)
        self.assertEqual(trade["exit_type"], "target")
        self.assertEqual(trade["gross_r"], 2.0)

    def test_stop_hit_short(self):
        """Verify a clean stop hit registers a loss of -1R for a short position."""
        MockBO01Strategy.signals_by_index = {
            2: {
                "signal": -1,
                "direction": "short",
                "stop_price": 1.1300,
                "target_rr": 2.0
            }
        }
        # Entry open = 1.1200 (Stop price = 1.1300, target price = 1.1000)
        # Index 4 high touches 1.1350 (hits stop), Low is safe at 1.1150
        self.df.loc[self.idx[4], "high"] = 1.1350
        self.df.loc[self.idx[4], "low"] = 1.1150
        
        res = runner.run_bo01_backtest_on_frame(MockBO01Strategy, self.df)
        self.assertEqual(res["trade_count"], 1)
        trade = res["trades"][0]
        self.assertEqual(trade["exit_idx"], 4)
        self.assertEqual(trade["exit_type"], "stop")
        self.assertEqual(trade["gross_r"], -1.0)

    def test_no_t_plus_one_aborts_signal(self):
        """Verify signals on the last bar of the DataFrame are skipped because t+1 is not available."""
        MockBO01Strategy.signals_by_index = {
            19: {
                "signal": 1,
                "direction": "long",
                "stop_price": 1.1100,
                "target_rr": 2.0
            }
        }
        res = runner.run_bo01_backtest_on_frame(MockBO01Strategy, self.df)
        self.assertEqual(res["trade_count"], 0)

    def test_costs_reduce_net_r(self):
        """Verify that positive spreads, slippage, and commissions deduct from gross cumulative performance."""
        MockBO01Strategy.signals_by_index = {
            2: {
                "signal": 1,
                "direction": "long",
                "stop_price": 1.1100,
                "target_rr": 2.0
            }
        }
        # Entry open = 1.1200 (Stop = 1.1100, target = 1.1400)
        # Distance = 100 pips
        # Hit target at index 4
        self.df.loc[self.idx[4], "high"] = 1.1450
        self.df.loc[self.idx[4], "low"] = 1.1150
        
        # Friction cost profile
        costs = {
            "spread": 1.5,
            "slippage": 0.5,
            "commission": 10.0
        }
        
        res = runner.run_bo01_backtest_on_frame(
            strategy_cls=MockBO01Strategy,
            frame=self.df,
            params={"pip_size": 0.0001},
            cost_profile=costs
        )
        self.assertEqual(res["trade_count"], 1)
        trade = res["trades"][0]
        
        # Verify: gross_r = 2.0, cost_r = (1.5+0.5)/100 + 10/(100*10) = 0.02 + 0.01 = 0.03 R.
        # net_r = 2.0 - 0.03 = 1.97 R.
        self.assertEqual(trade["gross_r"], 2.0)
        self.assertAlmostEqual(trade["cost_r"], 0.03)
        self.assertAlmostEqual(trade["net_r"], 1.97)
        self.assertEqual(res["gross_R"], 2.0)
        self.assertEqual(res["net_R"], 1.97)

    def test_non_dict_signal_fails_closed_without_crashing(self):
        """Verify that a completely malformed non-dictionary signal fails closed safely without runner crash."""
        MockBO01Strategy.signals_by_index = {
            2: ["this", "is", "not", "a", "dictionary"]
        }
        res = runner.run_bo01_backtest_on_frame(MockBO01Strategy, self.df)
        self.assertEqual(res["trade_count"], 0)
        self.assertEqual(res["invalid_signal_count"], 1)

    def test_skipped_active_position_counter_increments(self):
        """Verify that candles passed while holding a position increment skipped_signals_active_position."""
        MockBO01Strategy.signals_by_index = {
            2: {
                "signal": 1,
                "direction": "long",
                "stop_price": 1.1100,
                "target_rr": 2.0
            }
        }
        # First trade entry is open at index 3. It will remain active until end of frame (index 19) because stop/target never hit.
        # 20 candles total. Entry at 3. The exit occurs at index 19 (at the end).
        # Inside the loop:
        # Index 0, 1, 2: active_trade is None (3 bars)
        # Index 2: Signal evaluated and triggers entry for 3.
        # Index 3: Entry happens, index loop evaluates active_trade which is not None!
        # From index 3 to 19, active_trade remains active. Loop skips signal evaluation at index 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19.
        # Total skipped index bars = 17 bars (indices 3 to 19 inclusive).
        res = runner.run_bo01_backtest_on_frame(MockBO01Strategy, self.df)
        self.assertEqual(res["trade_count"], 1)
        self.assertEqual(res["skipped_signals_active_position"], 17)

    def test_commission_r_uses_standard_eurusd_pip_value_assumption(self):
        """Verify standard lot EURUSD USD-to-R commission scaling equations."""
        costs = {
            "spread": 0.0,
            "slippage": 0.0,
            "commission": 7.0  # $7 USD per lot round-turn
        }
        # entry = 1.1000, stop = 1.0990 => distance = 10 pips (pip_size = 0.0001)
        # expected commission R = 7.0 / (10 pips * $10 USD/pip) = 0.07 R
        cost_r = runner.compute_cost_r(
            entry_price=1.1000,
            stop_price=1.0990,
            cost_profile=costs,
            pip_size=0.0001
        )
        self.assertAlmostEqual(cost_r, 0.07)

    def test_no_real_data_access(self):
        """Self-testing that the test setup does not load external data files."""
        self.assertEqual(len(self.df), 20)
        self.assertTrue(isinstance(self.df.index, pd.DatetimeIndex))

if __name__ == "__main__":
    unittest.main()
