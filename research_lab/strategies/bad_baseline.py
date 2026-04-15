from __future__ import annotations

NAME = "bad_baseline"
WARMUP_BARS = 10

def parameter_grid(max_combinations: int = 1, seed: int = 42) -> list[dict]:
    return [{"session_name": "all_day", "target_rr": 1.0, "stop_atr": 1.0, "break_even_at_r": 0.0}]

def signal(frame, i: int, params: dict) -> dict | None:
    # Comprar ciegamente siempre a las 14:00 (hora NY)
    hour = frame.index[i].hour
    minute = frame.index[i].minute
    
    if hour == 14 and minute == 0:
        return {
            "direction": "long",
            "stop_mode": "atr",
            "stop_atr": params["stop_atr"],
            "target_rr": params["target_rr"],
            "break_even_at_r": 0.0,
            "session_name": params["session_name"],
            "max_hold_bars": 12, # 1 hr approx in M5
        }
    return None
