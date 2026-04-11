"""
NY MOMENTUM SYSTEM — Backtester
================================
Estrategia de trend following en sesión NY (11:00-19:00)
Indicadores: EMA 21/50/200 + RSI 14 + ATR 14 + ADX 14

Requisitos:
    pip install pandas numpy yfinance ta pytz

Uso:
    python ny_momentum_system.py

Para cambiar el par: modificar SYMBOL en la sección CONFIG.
Para correr todos los pares: descomentar el loop al final.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pytz
from datetime import datetime, time
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "symbol":          "EURUSD=X",       # Par a testear
    "start":           "2020-01-01",
    "end":             "2025-01-01",
    "timeframe":       "1h",             # Velas de 1 hora
    "session_start":   time(11, 0),      # 11:00 AM NY
    "session_end":     time(18, 30),     # Cierre forzado 18:30 NY
    "ema_fast":        21,
    "ema_slow":        50,
    "ema_trend":       200,
    "rsi_period":      14,
    "atr_period":      14,
    "adx_period":      14,
    "rsi_long_min":    52,               # RSI mínimo para long
    "rsi_short_max":   48,               # RSI máximo para short
    "adx_min":         20,               # ADX mínimo para operar
    "sl_atr_mult":     1.5,             # Stop: 1.5× ATR
    "tp_atr_mult":     3.0,             # Target: 3.0× ATR (1:2 R:R)
    "risk_pct":        0.01,            # 1% de capital por trade
    "initial_capital": 10_000,
    "daily_stop_pct":  0.025,           # Stop diario: -2.5%
    "weekly_stop_pct": 0.04,            # Stop semanal: -4%
    "max_open_trades": 3,
    "ny_tz":           "America/New_York",
}

PAIRS = [
    "EURUSD=X",
    "GBPUSD=X",
    "JPY=X",     # USD/JPY
    "AUDUSD=X",
    "GBPJPY=X",
    "CAD=X",     # USD/CAD
]

# ─────────────────────────────────────────────
# INDICADORES
# ─────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Calcula todos los indicadores sobre el dataframe de OHLCV."""
    d = df.copy()

    # EMAs
    d["ema_fast"]  = d["Close"].ewm(span=cfg["ema_fast"],  adjust=False).mean()
    d["ema_slow"]  = d["Close"].ewm(span=cfg["ema_slow"],  adjust=False).mean()
    d["ema_trend"] = d["Close"].ewm(span=cfg["ema_trend"], adjust=False).mean()

    # RSI
    delta = d["Close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=cfg["rsi_period"] - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=cfg["rsi_period"] - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["rsi"] = 100 - (100 / (1 + rs))

    # ATR
    h_l  = d["High"] - d["Low"]
    h_pc = (d["High"] - d["Close"].shift(1)).abs()
    l_pc = (d["Low"]  - d["Close"].shift(1)).abs()
    tr   = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    d["atr"] = tr.ewm(com=cfg["atr_period"] - 1, adjust=False).mean()

    # ADX
    up_move   = d["High"].diff()
    down_move = -d["Low"].diff()
    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_s    = tr.ewm(com=cfg["adx_period"] - 1, adjust=False).mean()
    plus_di  = 100 * pd.Series(plus_dm, index=d.index).ewm(com=cfg["adx_period"]-1, adjust=False).mean() / atr_s
    minus_di = 100 * pd.Series(minus_dm, index=d.index).ewm(com=cfg["adx_period"]-1, adjust=False).mean() / atr_s
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    d["adx"] = dx.ewm(com=cfg["adx_period"] - 1, adjust=False).mean()

    # Cruce de EMAs (señal fresca)
    d["ema_cross_long"]  = (d["ema_fast"] > d["ema_slow"]) & (d["ema_fast"].shift(1) <= d["ema_slow"].shift(1))
    d["ema_cross_short"] = (d["ema_fast"] < d["ema_slow"]) & (d["ema_fast"].shift(1) >= d["ema_slow"].shift(1))

    return d.dropna()


# ─────────────────────────────────────────────
# BACKTESTER
# ─────────────────────────────────────────────
class Trade:
    def __init__(self, direction, entry_price, sl, tp, entry_time, size):
        self.direction   = direction    # "long" | "short"
        self.entry_price = entry_price
        self.sl          = sl
        self.tp          = tp
        self.entry_time  = entry_time
        self.size        = size         # unidades de moneda base (aprox)
        self.exit_price  = None
        self.exit_time   = None
        self.pnl_pct     = None
        self.result      = None         # "win" | "loss" | "timeout"

    def close(self, price, ts, result):
        self.exit_price = price
        self.exit_time  = ts
        self.result     = result
        direction_mult  = 1 if self.direction == "long" else -1
        self.pnl_pct    = direction_mult * (price - self.entry_price) / self.entry_price


def run_backtest(symbol: str, cfg: dict) -> dict:
    print(f"\n{'='*50}")
    print(f"  Backtesting: {symbol}")
    print(f"  Período: {cfg['start']} → {cfg['end']}")
    print(f"{'='*50}")

    # ── Descargar datos ──
    raw = yf.download(symbol, start=cfg["start"], end=cfg["end"],
                      interval=cfg["timeframe"], auto_adjust=True, progress=False)

    if raw.empty:
        print(f"  ERROR: No se obtuvieron datos para {symbol}")
        return {}

    # Aplanar columnas MultiIndex si existen
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Convertir a timezone NY
    ny = pytz.timezone(cfg["ny_tz"])
    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("UTC")
    raw.index = raw.index.tz_convert(ny)

    # ── Calcular indicadores ──
    df = compute_indicators(raw, cfg)
    print(f"  Velas disponibles: {len(df):,}")

    # ── Variables de estado ──
    capital        = cfg["initial_capital"]
    open_trade     = None          # solo 1 trade a la vez por par
    trades         = []
    equity_curve   = []

    daily_start_cap  = capital
    weekly_start_cap = capital
    current_day      = None
    current_week     = None
    day_stopped      = False
    week_reduced     = False

    # ── Loop principal ──
    for i in range(1, len(df)):
        row      = df.iloc[i]
        prev     = df.iloc[i - 1]
        ts       = df.index[i]
        bar_time = ts.time()
        bar_day  = ts.date()
        bar_week = ts.isocalendar()[:2]

        # Reset diario / semanal
        if bar_day != current_day:
            daily_start_cap = capital
            current_day     = bar_day
            day_stopped     = False

        if bar_week != current_week:
            weekly_start_cap = capital
            current_week     = bar_week
            week_reduced     = False

        # Registrar equity
        equity_curve.append({"ts": ts, "equity": capital})

        # ── Chequear trade abierto ──
        if open_trade is not None:
            price = row["Close"]

            # Cierre forzado por horario
            if bar_time >= cfg["session_end"]:
                open_trade.close(price, ts, "timeout")
                capital += open_trade.pnl_pct * capital * cfg["risk_pct"] / (
                    cfg["sl_atr_mult"] * prev["atr"] / open_trade.entry_price
                ) if open_trade.entry_price > 0 else 0
                # Aproximación simplificada de P&L
                pnl = open_trade.pnl_pct * capital
                capital += pnl
                trades.append(open_trade)
                open_trade = None
                continue

            # SL hit
            sl_hit = (open_trade.direction == "long"  and row["Low"]  <= open_trade.sl) or \
                     (open_trade.direction == "short" and row["High"] >= open_trade.sl)
            # TP hit
            tp_hit = (open_trade.direction == "long"  and row["High"] >= open_trade.tp) or \
                     (open_trade.direction == "short" and row["Low"]  <= open_trade.tp)

            if tp_hit:
                exit_price = open_trade.tp
                open_trade.close(exit_price, ts, "win")
            elif sl_hit:
                exit_price = open_trade.sl
                open_trade.close(exit_price, ts, "loss")
            else:
                continue

            # P&L real en % sobre capital con 1% de riesgo
            direction_mult = 1 if open_trade.direction == "long" else -1
            raw_return = direction_mult * (open_trade.exit_price - open_trade.entry_price) / open_trade.entry_price
            sl_distance = abs(open_trade.entry_price - open_trade.sl) / open_trade.entry_price
            if sl_distance > 0:
                pnl_capital = (raw_return / sl_distance) * cfg["risk_pct"] * capital
            else:
                pnl_capital = 0
            capital += pnl_capital
            capital  = max(capital, 0)
            trades.append(open_trade)
            open_trade = None

        # ── Chequeos de filtro ──
        if day_stopped:
            continue
        if bar_time < cfg["session_start"] or bar_time >= cfg["session_end"]:
            continue

        # Stop diario
        if (capital - daily_start_cap) / daily_start_cap < -cfg["daily_stop_pct"]:
            day_stopped = True
            continue

        # Stop semanal → reducir size
        risk_pct = cfg["risk_pct"]
        if (capital - weekly_start_cap) / weekly_start_cap < -cfg["weekly_stop_pct"]:
            risk_pct = cfg["risk_pct"] * 0.5
            week_reduced = True

        price = row["Close"]
        atr   = row["atr"]

        # ── Condiciones LONG ──
        long_signal = (
            prev["ema_cross_long"]
            and price > row["ema_trend"]
            and row["rsi"] > cfg["rsi_long_min"]
            and row["adx"] > cfg["adx_min"]
        )

        # ── Condiciones SHORT ──
        short_signal = (
            prev["ema_cross_short"]
            and price < row["ema_trend"]
            and row["rsi"] < cfg["rsi_short_max"]
            and row["adx"] > cfg["adx_min"]
        )

        if long_signal and open_trade is None:
            sl = price - cfg["sl_atr_mult"] * atr
            tp = price + cfg["tp_atr_mult"] * atr
            open_trade = Trade("long", price, sl, tp, ts, size=1)

        elif short_signal and open_trade is None:
            sl = price + cfg["sl_atr_mult"] * atr
            tp = price - cfg["tp_atr_mult"] * atr
            open_trade = Trade("short", price, sl, tp, ts, size=1)

    # Cerrar trade pendiente al final
    if open_trade is not None:
        last_price = df.iloc[-1]["Close"]
        open_trade.close(last_price, df.index[-1], "timeout")
        trades.append(open_trade)

    # ── Estadísticas ──
    if not trades:
        print("  Sin trades ejecutados.")
        return {}

    equity_df = pd.DataFrame(equity_curve).set_index("ts")
    trade_df  = pd.DataFrame([{
        "entry_time":  t.entry_time,
        "exit_time":   t.exit_time,
        "direction":   t.direction,
        "entry_price": t.entry_price,
        "exit_price":  t.exit_price,
        "result":      t.result,
        "pnl_pct":     t.pnl_pct,
    } for t in trades])

    total_trades  = len(trade_df)
    wins          = (trade_df["result"] == "win").sum()
    losses        = (trade_df["result"] == "loss").sum()
    win_rate      = wins / total_trades * 100
    total_return  = (capital - cfg["initial_capital"]) / cfg["initial_capital"] * 100

    # Equity peak y max drawdown
    eq = equity_df["equity"]
    peak = eq.cummax()
    dd   = (eq - peak) / peak * 100
    max_dd = dd.min()

    # Profit factor (aproximado)
    win_trades  = trade_df[trade_df["result"] == "win"]["pnl_pct"]
    loss_trades = trade_df[trade_df["result"] == "loss"]["pnl_pct"]
    profit_factor = (win_trades.sum() / abs(loss_trades.sum())) if loss_trades.sum() != 0 else float("inf")

    # Resultados por año
    trade_df["year"] = trade_df["entry_time"].dt.year
    yearly = trade_df.groupby("year")["pnl_pct"].sum() * 100

    # Resultados por mes (análisis de meses negativos)
    trade_df["month"] = trade_df["entry_time"].dt.to_period("M")
    monthly = trade_df.groupby("month")["pnl_pct"].sum() * 100
    neg_months = (monthly < 0).sum()

    print(f"\n  {'RESULTADOS':─<38}")
    print(f"  Capital inicial:    ${cfg['initial_capital']:>10,.2f}")
    print(f"  Capital final:      ${capital:>10,.2f}")
    print(f"  Retorno total:      {total_return:>+9.2f}%")
    print(f"  Max Drawdown:       {max_dd:>+9.2f}%")
    print(f"  Total trades:       {total_trades:>10,}")
    print(f"  Win Rate:           {win_rate:>9.1f}%")
    print(f"  Profit Factor:      {profit_factor:>9.2f}")
    print(f"  Meses negativos:    {neg_months:>10} de {len(monthly)}")
    print(f"\n  Retorno anual:")
    for yr, ret in yearly.items():
        flag = " ✓" if ret > 0 else " ✗"
        print(f"    {yr}: {ret:>+7.2f}%{flag}")

    return {
        "symbol":         symbol,
        "total_return":   total_return,
        "max_dd":         max_dd,
        "win_rate":       win_rate,
        "profit_factor":  profit_factor,
        "total_trades":   total_trades,
        "neg_months":     neg_months,
        "total_months":   len(monthly),
        "yearly_returns": yearly.to_dict(),
        "equity_df":      equity_df,
        "trades_df":      trade_df,
    }


# ─────────────────────────────────────────────
# OPTIMIZADOR SIMPLE (grid search)
# ─────────────────────────────────────────────
def optimize(symbol: str, base_cfg: dict):
    """
    Mini grid search sobre los parámetros más impactantes.
    Retorna la mejor combinación por Sharpe aproximado.
    """
    print(f"\n{'='*50}")
    print(f"  OPTIMIZANDO: {symbol}")
    print(f"{'='*50}")

    param_grid = {
        "sl_atr_mult": [1.2, 1.5, 2.0],
        "tp_atr_mult": [2.4, 3.0, 4.0],
        "adx_min":     [18, 22, 25],
    }

    best_result = None
    best_score  = -np.inf

    for sl in param_grid["sl_atr_mult"]:
        for tp in param_grid["tp_atr_mult"]:
            for adx in param_grid["adx_min"]:
                if tp / sl < 1.8:     # R:R mínimo 1:1.8
                    continue
                cfg = {**base_cfg, "sl_atr_mult": sl, "tp_atr_mult": tp, "adx_min": adx}
                res = run_backtest(symbol, cfg)
                if not res:
                    continue
                # Score: retorno / |drawdown| (Calmar proxy)
                score = res["total_return"] / max(abs(res["max_dd"]), 1)
                if score > best_score:
                    best_score  = score
                    best_result = {**res, "params": {"sl": sl, "tp": tp, "adx": adx}}

    if best_result:
        print(f"\n  MEJORES PARÁMETROS para {symbol}:")
        print(f"    SL mult: {best_result['params']['sl']}")
        print(f"    TP mult: {best_result['params']['tp']}")
        print(f"    ADX min: {best_result['params']['adx']}")
        print(f"    Retorno: {best_result['total_return']:+.2f}%")
        print(f"    Max DD:  {best_result['max_dd']:+.2f}%")
    return best_result


# ─────────────────────────────────────────────
# REPORTE MULTI-PAR
# ─────────────────────────────────────────────
def run_all_pairs(pairs: list, cfg: dict):
    results = []
    for symbol in pairs:
        res = run_backtest(symbol, {**cfg, "symbol": symbol})
        if res:
            results.append(res)

    if not results:
        return

    print(f"\n{'='*60}")
    print("  RESUMEN MULTI-PAR")
    print(f"{'='*60}")
    print(f"  {'Par':<12} {'Retorno':>8} {'Max DD':>8} {'WR%':>6} {'PF':>6} {'Neg/Mes':>8}")
    print(f"  {'-'*54}")
    for r in results:
        sym = r["symbol"].replace("=X", "")
        print(f"  {sym:<12} {r['total_return']:>+7.1f}% {r['max_dd']:>+7.1f}% "
              f"{r['win_rate']:>5.1f}% {r['profit_factor']:>5.2f} "
              f"{r['neg_months']:>3}/{r['total_months']}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  NY MOMENTUM SYSTEM — Backtester")
    print("  Sesión: 11:00–19:00 NY  |  Timeframe: H1")
    print("  Período: 2020–2025\n")

    # ── Opción 1: Testear un solo par ──
    result = run_backtest(CONFIG["symbol"], CONFIG)

    # ── Opción 2: Testear todos los pares (descomentá estas líneas) ──
    # run_all_pairs(PAIRS, CONFIG)

    # ── Opción 3: Optimizar un par ──
    # optimize("EURUSD=X", CONFIG)
