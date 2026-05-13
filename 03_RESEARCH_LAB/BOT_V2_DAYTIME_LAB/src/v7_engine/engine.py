from __future__ import annotations
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

from src.v6_utils.execution import next_bar_execute, next_bar_execute_stop, FillResult, NoFillError
from src.v6_utils.numeric import snap_to_tick
from src.v7_engine.news_filter import NewsCalendar, is_blocked_by_news
from src.v7_engine.ftmo_compliance import FtmoComplianceEngine
from src.v7_engine.position_throttler import PositionThrottler
from src.v7_engine.schedule_guard import ScheduleGuard
from src.v7_engine.security_gates import guard_ticks_readonly

from src.v7_engine.cost_model import CostModel, CostModelConfig, CostCalculationBlockedError
from src.v7_engine.trade_record import TradeRecord
import uuid

class TestLeakageViolation(Exception):
    """
    Excepción crítica levantada incondicionalmente cuando una fase de optimización,
    barrido de parámetros o entrenamiento intenta acceder, consultar o ejecutar
    sobre marcas de tiempo pertenecientes a la partición de prueba reservada (TEST).
    """
    pass

class TestLeakageGuard:
    """
    Guarda forense inmutable (D5C Anti-Leakage Guard) que aísla físicamente
    el flujo de trabajo de optimización y evita filtraciones de información futuras.
    """
    def __init__(self, active_phase: str = "train", test_start_year: int = 2023):
        self.active_phase = active_phase.lower().strip()
        self.test_start_year = test_start_year

    def verify_timestamp(self, ts: datetime | pd.Timestamp) -> bool:
        if self.active_phase in ["train", "validation", "val", "sweep"]:
            year = ts.year
            if year >= self.test_start_year:
                raise TestLeakageViolation(
                    f"[ANTI-LEAKAGE GUARD] Violación crítica de partición OOS: "
                    f"La fase activa '{self.active_phase}' intentó consumir o evaluar "
                    f"datos correspondientes al conjunto reservado TEST ({ts})."
                )
        return True

class UnifiedV7Engine:
    """
    Motor central unificado V7 (Sección 3.4). Intercepta y orquesta en estricta
    secuencia causal las 5 capas de validación y control sobre la base de ejecución V6.
    Incorpora trazabilidad nativa del CausalLog y blindaje OOS de particiones (D5C).
    """
    def __init__(
        self,
        news_calendar: NewsCalendar,
        initial_balance: float = 100000.0,
        max_trades_per_day: int = 3,
        entry_start_hour: int = 8,
        entry_end_hour: int = 11,
        forced_exit_mode: str = "16:00",
        news_mode: str = "post5",
        active_phase: str = "test",
        test_start_year: int = 2023,
        cost_model: CostModel | None = None,
        default_instrument: str = "EURUSD"
    ):
        self.news_calendar = news_calendar
        self.ftmo = FtmoComplianceEngine(initial_balance=initial_balance)
        self.throttler = PositionThrottler(max_trades_per_day=max_trades_per_day)
        self.schedule_guard = ScheduleGuard(
            entry_start_hour=entry_start_hour,
            entry_end_hour=entry_end_hour,
            forced_exit_mode=forced_exit_mode
        )
        self.news_mode = news_mode.lower().strip()
        self.cost_model = cost_model if cost_model is not None else CostModel(CostModelConfig(mode="conservative"))
        self.default_instrument = default_instrument
        
        self.causal_log: list[dict[str, any]] = []
        self.trade_ledger: list[TradeRecord] = []
        self.leak_guard = TestLeakageGuard(active_phase=active_phase, test_start_year=test_start_year)
        
        self.pre_minutes = 0
        self.post_minutes = 0
        if "post" in self.news_mode:
            val = "".join(filter(str.isdigit, self.news_mode))
            self.post_minutes = int(val) if val else 5
        elif "pre" in self.news_mode:
            val = "".join(filter(str.isdigit, self.news_mode))
            mins = int(val) if val else 15
            self.pre_minutes = mins
            self.post_minutes = mins

    def get_causal_log(self) -> list[dict[str, any]]:
        return self.causal_log
        
    def get_trade_ledger(self) -> list[TradeRecord]:
        return self.trade_ledger

    def execute_signal(
        self,
        side: str,
        signal_bar_close: pd.Timestamp,
        ticks_after: pd.DataFrame,
        instrument: str = "EURUSD",
        entry_mode: str = "market",
        stop_price: float | None = None
    ) -> tuple[FillResult | None, str]:
        self.leak_guard.verify_timestamp(signal_bar_close)
        guard_ticks_readonly("data/dukascopy/EURUSD_ticks.csv", mode="r")
        
        ts_utc = signal_bar_close.to_pydatetime()
        if ts_utc.tzinfo is not None:
            ts_utc = ts_utc.replace(tzinfo=None)
            
        if not self.schedule_guard.is_entry_permitted(ts_utc):
            self.causal_log.append({"event": "SIGNAL_REJECTED", "signal_ts": str(signal_bar_close), "reason": "BLOCKED_BY_SCHEDULE"})
            return None, "BLOCKED_BY_SCHEDULE"
            
        if self.pre_minutes > 0 or self.post_minutes > 0:
            blocked, _ = is_blocked_by_news(ts_utc, self.news_calendar, pre_minutes=self.pre_minutes, post_minutes=self.post_minutes)
            if blocked:
                self.causal_log.append({"event": "SIGNAL_REJECTED", "signal_ts": str(signal_bar_close), "reason": "BLOCKED_BY_NEWS"})
                return None, "BLOCKED_BY_NEWS"
                
        if not self.ftmo.update_state(ts_utc, closed_pnl=0.0, floating_pnl=0.0):
            self.causal_log.append({"event": "SIGNAL_REJECTED", "signal_ts": str(signal_bar_close), "reason": "BLOCKED_BY_BLOWN_STATE"})
            return None, "BLOCKED_BY_BLOWN_STATE"
            
        if not self.throttler.allow_trade(ts_utc):
            self.causal_log.append({"event": "SIGNAL_REJECTED", "signal_ts": str(signal_bar_close), "reason": "BLOCKED_BY_THROTTLER"})
            return None, "BLOCKED_BY_THROTTLER"
            
        if entry_mode == "stop":
            if stop_price is None:
                self.causal_log.append({"event": "SIGNAL_REJECTED", "signal_ts": str(signal_bar_close), "reason": "MISSING_STOP_PRICE"})
                return None, "MISSING_STOP_PRICE"
            fill = next_bar_execute_stop(side, signal_bar_close, stop_price, ticks_after, expiry_minutes=60, instrument=instrument)
            if fill is None:
                self.causal_log.append({"event": "SIGNAL_REJECTED", "signal_ts": str(signal_bar_close), "reason": "STOP_NOT_TOUCHED"})
                return None, "STOP_NOT_TOUCHED"
        else:
            fill = next_bar_execute(side, signal_bar_close, ticks_after, instrument)
            
        self.causal_log.append({"event": "EXECUTION_FILL", "fill_ts": str(fill.fill_time), "side": side, "fill_price": fill.fill_price, "reason": fill.reason})
        return fill, fill.reason


    def close_position_with_costs(
        self,
        fill: FillResult,
        sl_price: float,
        tp_price: float,
        ticks_during: pd.DataFrame,
        instrument: str = "EURUSD",
        be_trigger_r: float | None = None,
        be_move_to_offset: float = 0.00005
    ) -> TradeRecord:
        
        ticks_eval = ticks_during[ticks_during.index > fill.fill_time]
        if ticks_eval.empty:
            raise NoFillError(f"Ausencia de ticks posteriores a la ejecución en {fill.fill_time}")
            
        side = fill.side
        entry_price = fill.fill_price
        
        risk_amount = self.ftmo.get_position_risk_amount()
        sl_dist = abs(entry_price - sl_price)
        if sl_dist <= 0:
            raise CostCalculationBlockedError("SL distance <= 0")
            
        sl_pips = sl_dist / self.cost_model.config.pip_size
        if sl_pips <= 0:
            raise CostCalculationBlockedError("sl_pips <= 0")
            
        trade_units = risk_amount / sl_dist
        
        exit_result: FillResult | None = None
        be_activated = False
        current_sl = sl_price
        
        for row in ticks_eval.itertuples():
            ts = row.Index
            ts_utc = ts.to_pydatetime().replace(tzinfo=None)
            
            if self.schedule_guard.should_force_exit(ts_utc):
                curr_exit_price = row.bid if side == "long" else row.ask
                exit_result = FillResult(ts, snap_to_tick(curr_exit_price, instrument), side, fill.signal_time, fill.signal_bar_close, "TIME")
                break
                
            if be_trigger_r is not None and not be_activated:
                if side == "long":
                    r_fav = (row.bid - entry_price) / sl_dist
                    if r_fav >= be_trigger_r:
                        current_sl = entry_price + be_move_to_offset
                        be_activated = True
                        self.causal_log.append({"event": "BE_ACTIVATED", "ts": str(ts), "new_sl": current_sl})
                else:
                    r_fav = (entry_price - row.ask) / sl_dist
                    if r_fav >= be_trigger_r:
                        current_sl = entry_price - be_move_to_offset
                        be_activated = True
                        self.causal_log.append({"event": "BE_ACTIVATED", "ts": str(ts), "new_sl": current_sl})
                        
            if side == "long":
                curr_p = row.bid
                if curr_p <= current_sl:
                    exit_result = FillResult(ts, current_sl, side, fill.signal_time, fill.signal_bar_close, "BE" if be_activated else "SL")
                    break
                if curr_p >= tp_price:
                    exit_result = FillResult(ts, tp_price, side, fill.signal_time, fill.signal_bar_close, "TP")
                    break
            else:
                curr_p = row.ask
                if curr_p >= current_sl:
                    exit_result = FillResult(ts, current_sl, side, fill.signal_time, fill.signal_bar_close, "BE" if be_activated else "SL")
                    break
                if curr_p <= tp_price:
                    exit_result = FillResult(ts, tp_price, side, fill.signal_time, fill.signal_bar_close, "TP")
                    break
                    
        if exit_result is None:
            last_r = ticks_eval.iloc[-1]
            last_p = last_r["bid"] if side == "long" else last_r["ask"]
            exit_result = FillResult(ticks_eval.index[-1], snap_to_tick(last_p, instrument), side, fill.signal_time, fill.signal_bar_close, "EOM")
            
        price_diff = (exit_result.fill_price - entry_price) if side == "long" else (entry_price - exit_result.fill_price)
        gross_r = price_diff / sl_dist
        
        try:
            cost_res = self.cost_model.apply_costs_to_trade(gross_r=gross_r, reason=exit_result.reason, sl_pips=sl_pips, risk_per_trade_cash=risk_amount)
        except CostCalculationBlockedError as e:
            self.causal_log.append({"event": "COST_CALCULATION_BLOCKED", "reason": str(e)})
            raise
            
        net_r = cost_res["net_r"]
        commission_r = cost_res["commission_r"]
        commission_usd = cost_res.get("commission_usd", 0.0)
        slippage_r = cost_res["slippage_r"]
        lots = cost_res.get("lots", 0.0)
        
        gross_pnl_usd = gross_r * risk_amount
        net_pnl_usd = net_r * risk_amount
        
        intended_end = fill.fill_time + pd.Timedelta(hours=8)
        w_start = ticks_during.index[0] if not ticks_during.empty else fill.fill_time
        w_end = ticks_during.index[-1] if not ticks_during.empty else fill.fill_time
        w_sec = (w_end - w_start).total_seconds() if not ticks_during.empty else 0.0
        scanned_cnt = len(ticks_during)
        
        tick_window_complete = (w_end >= intended_end or exit_result.reason in ["TP", "SL", "BE", "TIME"])
        
        eom_type = "NO_EOM"
        if exit_result.reason == "EOM":
            if not tick_window_complete:
                eom_type = "ARTIFICIAL_TRUNCATION"
            else:
                eom_type = "REAL_DATA_END"
        elif exit_result.reason == "TIME":
            eom_type = "SESSION_FORCED_EXIT"
            
        trade = TradeRecord(
            trade_id=str(uuid.uuid4()),
            side=side,
            signal_time=fill.signal_time.to_pydatetime() if hasattr(fill.signal_time, 'to_pydatetime') else fill.signal_time,
            fill_time=fill.fill_time.to_pydatetime() if hasattr(fill.fill_time, 'to_pydatetime') else fill.fill_time,
            exit_time=exit_result.fill_time.to_pydatetime() if hasattr(exit_result.fill_time, 'to_pydatetime') else exit_result.fill_time,
            entry_price=entry_price,
            exit_price=exit_result.fill_price,
            sl_price=sl_price,
            tp_price=tp_price,
            exit_reason=exit_result.reason,
            gross_r=gross_r,
            sl_pips=sl_pips,
            lots=lots,
            commission_usd=commission_usd,
            commission_r=commission_r,
            slippage_r=slippage_r,
            net_r=net_r,
            gross_pnl_usd=gross_pnl_usd,
            net_pnl_usd=net_pnl_usd,
            risk_usd=risk_amount,
            cost_model_mode=self.cost_model.config.mode,
            ftmo_cost_applied=True,
            instrument=instrument,
            forced_exit=(exit_result.reason == "TIME" or exit_result.reason == "EOM"),
            be_activated=be_activated,
            valid_closed_trade=(eom_type != "ARTIFICIAL_TRUNCATION"),
            tick_window_start=str(w_start),
            tick_window_end=str(w_end),
            intended_position_end=str(intended_end),
            actual_tick_window_end=str(w_end),
            tick_window_seconds=w_sec,
            intended_window_seconds=8 * 3600.0,
            tick_window_complete=tick_window_complete,
            eom_type=eom_type,
            ticks_scanned_count=scanned_cnt
        )

        
        self.trade_ledger.append(trade)
        
        self.causal_log.append({"event": "POSITION_CLOSED", "exit_ts": str(exit_result.fill_time), "reason": exit_result.reason, "realized_pnl": gross_pnl_usd})
        self.causal_log.append({"event": "COSTS_APPLIED", "net_r": net_r, "commission_usd": commission_usd})
        self.causal_log.append({"event": "FTMO_UPDATED_NET", "net_pnl_usd": net_pnl_usd})
        
        self.ftmo.update_state(
            exit_result.fill_time.to_pydatetime().replace(tzinfo=None),
            closed_pnl=net_pnl_usd,
            floating_pnl=0.0
        )
        
        return trade
        
    def simulate_position(self, *args, **kwargs) -> FillResult:
        """Compatibilidad legacy"""
        record = self.close_position_with_costs(*args, **kwargs)
        # Recreamos FillResult para no romper código viejo
        from src.v6_utils.execution import FillResult as LegacyFillResult
        import pandas as pd
        return LegacyFillResult(pd.Timestamp(record.exit_time), record.exit_price, record.side, pd.Timestamp(record.signal_time), pd.Timestamp(record.signal_time), record.exit_reason)
