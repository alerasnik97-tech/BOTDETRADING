def _get_val(t, field, default=0.0):
    if isinstance(t, dict):
        return t.get(field, default)
    return getattr(t, field, default)

def calculate_profit_factor(trades: list, r_field: str = "net_r") -> float:
    if not trades:
        return 0.0
    wins = sum(_get_val(t, r_field, 0.0) for t in trades if _get_val(t, r_field, 0.0) > 0)
    losses = abs(sum(_get_val(t, r_field, 0.0) for t in trades if _get_val(t, r_field, 0.0) < 0))
    if losses == 0:
        return float('inf') if wins > 0 else 0.0
    return wins / losses

def calculate_expectancy(trades: list, r_field: str = "net_r") -> float:
    if not trades:
        return 0.0
    total_r = sum(_get_val(t, r_field, 0.0) for t in trades)
    return total_r / len(trades)

def calculate_equity_curve(trades: list, r_field: str = "net_r") -> list[float]:
    curve = [0.0]
    for t in trades:
        curve.append(curve[-1] + _get_val(t, r_field, 0.0))
    return curve

def calculate_max_drawdown(trades: list, r_field: str = "net_r") -> float:
    curve = calculate_equity_curve(trades, r_field)
    peak = curve[0]
    mdd = 0.0
    for val in curve:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > mdd:
            mdd = dd
    return mdd

def summarize_trades(trades: list, r_field: str = "net_r") -> dict:
    if not trades:
        return {}
    pf = calculate_profit_factor(trades, r_field)
    exp = calculate_expectancy(trades, r_field)
    mdd = calculate_max_drawdown(trades, r_field)
    total_r = sum(_get_val(t, r_field, 0.0) for t in trades)
    return {
        "total_trades": len(trades),
        "profit_factor": round(pf, 4),
        "expectancy": round(exp, 4),
        "max_drawdown": round(mdd, 4),
        "total_r": round(total_r, 4)
    }

