from __future__ import annotations
import MetaTrader5 as mt5 # type: ignore
from typing import Any
from phase37_ftmo_trial_support import account_gate
from phase37x_position_state import get_position_state

def execute_safe_close(symbol: str = "EURUSD") -> dict[str, Any]:
    # 1. Gates
    acc = account_gate()
    if acc.get("state") != "FTMO_DEMO_TRIAL_CONFIRMED":
        return {"status": "SAFE_CLOSE_BLOCKED_ACCOUNT", "reason": acc.get("state")}
    
    # 2. Position
    pos = get_position_state(symbol)
    if pos["state"] == "FLAT":
        return {"status": "SAFE_CLOSE_NOT_REQUIRED_FLAT"}
    
    if pos["state"] not in {"MANIPULANTE_POSITION_OPEN", "POSITION_WITHOUT_SLTP_BLOCKER"}:
        return {"status": "SAFE_CLOSE_BLOCKED_UNKNOWN_POSITION", "reason": pos["state"]}
    
    # 3. Order
    ticket = pos["ticket"]
    volume = pos["volume"]
    side = pos["side"]
    order_type = mt5.ORDER_TYPE_SELL if side == "BUY" else mt5.ORDER_TYPE_BUY
    
    price = mt5.symbol_info_tick(symbol).bid if side == "BUY" else mt5.symbol_info_tick(symbol).ask
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "position": ticket,
        "price": price,
        "deviation": 20,
        "magic": 37000,
        "comment": "PHASE37X_SAFE_CLOSE",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    # 4. Check
    check = mt5.order_check(request)
    if check.retcode != mt5.TRADE_RETCODE_DONE:
        return {"status": "SAFE_CLOSE_BLOCKED_ORDER_CHECK", "retcode": check.retcode}
    
    # 5. Send
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return {"status": "SAFE_CLOSE_FAILED", "retcode": result.retcode}
    
    # 6. Verify
    verify = get_position_state(symbol)
    if verify["state"] == "FLAT":
        return {"status": "SAFE_CLOSE_SUCCESS", "ticket": ticket}
    else:
        return {"status": "CRITICAL_POSITION_STILL_OPEN", "ticket": ticket}
