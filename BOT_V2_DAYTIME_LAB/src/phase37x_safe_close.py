from __future__ import annotations
import MetaTrader5 as mt5 # type: ignore
from typing import Any
import time
from phase37_ftmo_trial_support import account_gate
from phase37x_position_state import get_position_state

from phase54_execution_logger import log_execution_event

def execute_safe_close(symbol: str = "EURUSD", max_attempts: int = 3, retry_delay: int = 5) -> dict[str, Any]:
    # 1. Account Security Gates
    acc = account_gate()
    if acc.get("state") != "FTMO_DEMO_TRIAL_CONFIRMED":
        return {"status": "SAFE_CLOSE_BLOCKED_ACCOUNT", "reason": acc.get("state")}
    
    attempts = 0
    last_res = None
    
    log_execution_event("EXIT", "ATTEMPT", {
        "symbol": symbol,
        "close_reason": "TIME_EXIT",
        "notes": "Safe close initiated",
        "source_file": "phase37x_safe_close.py"
    })

    while attempts < max_attempts:
        attempts += 1
        # 2. Position Check
        pos = get_position_state(symbol)
        if pos["state"] == "FLAT":
            log_execution_event("EXIT", "SAFE_CLOSE", {
                "symbol": symbol,
                "close_reason": "ALREADY_FLAT",
                "source_file": "phase37x_safe_close.py"
            })
            return {"status": "SAFE_CLOSE_SUCCESS", "attempts": attempts, "details": "Already flat"}
        
        if pos["state"] not in {"MANIPULANTE_POSITION_OPEN", "POSITION_WITHOUT_SLTP_BLOCKER"}:
            log_execution_event("EXIT", "ERROR", {
                "symbol": symbol,
                "close_reason": "UNKNOWN_POSITION_STATE",
                "notes": f"state: {pos['state']}",
                "source_file": "phase37x_safe_close.py"
            })
            return {"status": "SAFE_CLOSE_BLOCKED_UNKNOWN_POSITION", "reason": pos["state"]}
        
        # 3. Order Preparation
        ticket = pos["ticket"]
        volume = pos["volume"]
        side = pos["side"]
        order_type = mt5.ORDER_TYPE_SELL if side == "BUY" else mt5.ORDER_TYPE_BUY
        
        tick = mt5.symbol_info_tick(symbol)
        price = tick.bid if side == "BUY" else tick.ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 37000,
            "comment": f"PHASE37X_SAFE_CLOSE_A{attempts}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # 4. Order Check
        check = mt5.order_check(request)
        if check.retcode != mt5.TRADE_RETCODE_DONE:
            last_res = {"status": "SAFE_CLOSE_BLOCKED_ORDER_CHECK", "retcode": check.retcode}
            time.sleep(retry_delay)
            continue
        
        # 5. Order Send
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            last_res = {"status": "SAFE_CLOSE_FAILED", "retcode": result.retcode}
            time.sleep(retry_delay)
            continue
        
        # 6. Verify Immediate Flat
        verify = get_position_state(symbol)
        if verify["state"] == "FLAT":
            log_execution_event("EXIT", "FILLED", {
                "symbol": symbol,
                "close_reason": "SAFE_CLOSE",
                "order_ticket": ticket,
                "executed_price": result.price,
                "bid": tick.bid,
                "ask": tick.ask,
                "spread_pips": (tick.ask - tick.bid) * 10000,
                "source_file": "phase37x_safe_close.py"
            })
            return {"status": "SAFE_CLOSE_SUCCESS", "ticket": ticket, "attempts": attempts}
        
        time.sleep(retry_delay)

    log_execution_event("EXIT", "ERROR", {
        "symbol": symbol,
        "close_reason": "CRITICAL_STILL_OPEN",
        "source_file": "phase37x_safe_close.py"
    })
    return {"status": "CRITICAL_POSITION_STILL_OPEN", "attempts": attempts, "last_error": last_res}
