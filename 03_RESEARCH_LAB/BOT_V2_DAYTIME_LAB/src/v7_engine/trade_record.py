from dataclasses import dataclass
from datetime import datetime

@dataclass
class TradeRecord:
    trade_id: str
    side: str
    signal_time: datetime
    fill_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    sl_price: float
    tp_price: float
    exit_reason: str
    gross_r: float
    sl_pips: float
    lots: float
    commission_usd: float
    commission_r: float
    slippage_r: float
    net_r: float
    gross_pnl_usd: float
    net_pnl_usd: float
    risk_usd: float
    cost_model_mode: str
    ftmo_cost_applied: bool
    instrument: str
    forced_exit: bool
    be_activated: bool
    valid_closed_trade: bool = True
    
    # Campos de trazabilidad dimensional y truncamiento (Seccion 7 del prompt)
    tick_window_start: str = ""
    tick_window_end: str = ""
    intended_position_end: str = ""
    actual_tick_window_end: str = ""
    tick_window_seconds: float = 0.0
    intended_window_seconds: float = 0.0
    tick_window_complete: bool = True
    eom_type: str = "NO_EOM"
    ticks_scanned_count: int = 0

