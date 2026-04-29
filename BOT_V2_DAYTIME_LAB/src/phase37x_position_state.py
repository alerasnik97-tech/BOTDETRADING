from __future__ import annotations
from typing import Any
import MetaTrader5 as mt5 # type: ignore

def get_position_state(symbol: str = "EURUSD") -> dict[str, Any]:
    positions = mt5.positions_get(symbol=symbol)
    
    if positions is None:
        return {"state": "MT5_CONNECTION_ERROR", "positions": []}
    
    if len(positions) == 0:
        return {"state": "FLAT", "positions": []}
    
    if len(positions) > 1:
        return {"state": "MULTIPLE_POSITIONS_BLOCKER", "count": len(positions)}
    
    pos = positions[0]
    pos_dict = pos._asdict() if hasattr(pos, "_asdict") else {}
    
    # Check for SL/TP
    has_sl = float(pos_dict.get("sl", 0)) > 0
    has_tp = float(pos_dict.get("tp", 0)) > 0
    
    state = "MANIPULANTE_POSITION_OPEN"
    # Basic heuristic for Manipulante: magic number or comment
    # Since we don't have a specific magic fixed yet in the prompts, we assume current position is it
    # but we flag if SL/TP is missing
    if not has_sl or not has_tp:
        state = "POSITION_WITHOUT_SLTP_BLOCKER"
        
    return {
        "state": state,
        "ticket": pos_dict.get("ticket"),
        "side": "BUY" if pos_dict.get("type") == 0 else "SELL",
        "volume": pos_dict.get("volume"),
        "open_price": pos_dict.get("price_open"),
        "sl": pos_dict.get("sl"),
        "tp": pos_dict.get("tp"),
        "profit": pos_dict.get("profit"),
        "has_sl": has_sl,
        "has_tp": has_tp
    }
