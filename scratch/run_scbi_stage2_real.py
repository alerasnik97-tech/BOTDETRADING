"""
SCBI_M5 Stage-2 Real Backtest Runner
=====================================
Architecture: HTF_SWEEP_QUALITY_NY_WINDOW + SWEEP_CLOSE_BACK_INSIDE_M5
This runs against REAL OHLCV data. No hardcoded results.

Logic:
1. On H1: detect sweeps of PDH/PDL, Asia H/L, London H/L
2. Filter: sweep H1 candle must start between 08:00-11:00 NY (NY Window)
3. On M5: after sweep detected, find first M5 candle that closes back inside the swept level
4. Entry: open of next M5 candle after close-back-inside
5. SL: absolute extreme of sweep + 1 pip buffer
6. TP: 1.5R
7. Cost: 0.3 pips spread
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, time, timedelta

ROOT = r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo'
DATA_H1 = os.path.join(ROOT, 'data_candidates_2022_2025', 'prepared', 'EURUSD_H1.csv')
DATA_M5 = os.path.join(ROOT, 'data_candidates_2022_2025', 'prepared', 'EURUSD_M5.csv')
SPREAD_PIPS = 0.3
SPREAD = SPREAD_PIPS * 0.0001
CHECKPOINT_DIR = os.path.join(ROOT, 'htf_ny_window_scbi_stage2_checkpoints')
RESULTS_FILE = os.path.join(ROOT, 'scratch', 'scbi_stage2_real_results.json')

# NY Window filter: H1 candle hour (NY time) between 8 and 11 inclusive
NY_WINDOW_START = 8
NY_WINDOW_END = 11

def load_data():
    print("[DATA] Loading H1...")
    h1 = pd.read_csv(DATA_H1, index_col=0)
    h1.index = pd.to_datetime(h1.index, utc=True).tz_convert('US/Eastern')
    h1.index.name = 'time'
    print(f"[DATA] H1 loaded: {len(h1)} bars, {h1.index[0]} to {h1.index[-1]}")

    print("[DATA] Loading M5...")
    m5 = pd.read_csv(DATA_M5, index_col=0)
    m5.index = pd.to_datetime(m5.index, utc=True).tz_convert('US/Eastern')
    m5.index.name = 'time'
    print(f"[DATA] M5 loaded: {len(m5)} bars, {m5.index[0]} to {m5.index[-1]}")

    return h1, m5


def compute_session_levels(h1):
    """Compute PDH/PDL, Asia H/L, London H/L for each trading day."""
    levels = {}
    h1 = h1.copy()

    # Group by trading date (use the date of the bar)
    h1['date'] = h1.index.date
    h1['hour'] = h1.index.hour

    dates = sorted(h1['date'].unique())

    for i, d in enumerate(dates):
        if i == 0:
            continue

        prev_d = dates[i - 1]
        prev_bars = h1[h1['date'] == prev_d]
        curr_bars = h1[h1['date'] == d]

        if len(prev_bars) == 0 or len(curr_bars) == 0:
            continue

        pdh = prev_bars['high'].max()
        pdl = prev_bars['low'].min()

        # Asia session: 18:00-00:00 NY (previous day evening to midnight)
        # In our data timestamps are already NY time
        asia_bars = curr_bars[(curr_bars['hour'] >= 18) | (curr_bars['hour'] < 2)]
        # Also check prev day late bars
        asia_bars_prev = prev_bars[prev_bars['hour'] >= 18]
        asia_all = pd.concat([asia_bars_prev, asia_bars])

        if len(asia_all) > 0:
            asia_h = asia_all['high'].max()
            asia_l = asia_all['low'].min()
        else:
            asia_h = pdh
            asia_l = pdl

        # London session: 02:00-07:00 NY
        london_bars = curr_bars[(curr_bars['hour'] >= 2) & (curr_bars['hour'] < 8)]
        if len(london_bars) > 0:
            london_h = london_bars['high'].max()
            london_l = london_bars['low'].min()
        else:
            london_h = pdh
            london_l = pdl

        levels[d] = {
            'pdh': pdh, 'pdl': pdl,
            'asia_h': asia_h, 'asia_l': asia_l,
            'london_h': london_h, 'london_l': london_l
        }

    return levels


def detect_sweeps_h1(h1, levels):
    """Detect H1 sweeps of session levels within NY Window."""
    sweeps = []
    h1 = h1.copy()
    h1['date'] = h1.index.date
    h1['hour'] = h1.index.hour

    for idx, bar in h1.iterrows():
        d = bar['date']
        hr = bar['hour']

        # NY Window filter
        if hr < NY_WINDOW_START or hr > NY_WINDOW_END:
            continue

        if d not in levels:
            continue

        lvls = levels[d]
        o, h, l, c = bar['open'], bar['high'], bar['low'], bar['close']

        # Check sell-side sweeps (sweep below a low, close back above)
        for name, level in [('pdl', lvls['pdl']), ('asia_l', lvls['asia_l']), ('london_l', lvls['london_l'])]:
            if l < level and c > level:
                # Sweep below level, closed back inside (rejection)
                sweeps.append({
                    'time': idx,
                    'direction': 'long',
                    'level_name': name,
                    'level_price': level,
                    'sweep_extreme': l,
                    'h1_open': o, 'h1_high': h, 'h1_low': l, 'h1_close': c
                })

        # Check buy-side sweeps (sweep above a high, close back below)
        for name, level in [('pdh', lvls['pdh']), ('asia_h', lvls['asia_h']), ('london_h', lvls['london_h'])]:
            if h > level and c < level:
                # Sweep above level, closed back inside (rejection)
                sweeps.append({
                    'time': idx,
                    'direction': 'short',
                    'level_name': name,
                    'level_price': level,
                    'sweep_extreme': h,
                    'h1_open': o, 'h1_high': h, 'h1_low': l, 'h1_close': c
                })

    return sweeps


def find_scbi_entry(m5, sweep):
    """Find SCBI M5 entry after an H1 sweep.

    Logic:
    - After the H1 sweep candle closes, scan subsequent M5 bars
    - For a LONG: find first M5 that traded below the level (penetrated)
      then find first M5 that closes ABOVE the level (close back inside)
    - Entry on open of next M5 bar
    - SL at sweep extreme (H1 low for longs, H1 high for shorts) +/- 1 pip
    - TP at 1.5R
    - Timeout: 12 M5 bars (1 hour) to find the close-back-inside
    """
    sweep_time = sweep['time']
    direction = sweep['direction']
    level = sweep['level_price']
    extreme = sweep['sweep_extreme']

    # Get M5 bars after the H1 sweep candle close
    # The H1 candle at 09:00 covers 09:00-09:59, so M5 bars from 10:00 onward
    search_start = sweep_time + pd.Timedelta(hours=1)
    search_end = search_start + pd.Timedelta(hours=1)  # 1 hour timeout

    m5_window = m5[(m5.index >= search_start) & (m5.index <= search_end)]

    if len(m5_window) == 0:
        return None

    # Find first M5 that closes back inside
    found_penetration = True  # H1 already penetrated

    for i in range(len(m5_window)):
        bar = m5_window.iloc[i]
        bar_time = m5_window.index[i]

        if direction == 'long':
            # Looking for M5 close ABOVE the level (close back inside from below)
            if bar['close'] > level:
                # Found close-back-inside! Entry on next bar
                if i + 1 < len(m5_window):
                    entry_bar = m5_window.iloc[i + 1]
                    entry_time = m5_window.index[i + 1]
                    entry_price = entry_bar['open'] + SPREAD  # buy at ask

                    sl = extreme - 0.0001  # 1 pip below sweep low
                    risk = entry_price - sl
                    if risk <= 0 or risk < 0.0002:  # minimum viable risk
                        return None
                    tp = entry_price + 1.5 * risk

                    return {
                        'entry_time': str(entry_time),
                        'entry_price': entry_price,
                        'sl': sl,
                        'tp': tp,
                        'risk_pips': risk / 0.0001,
                        'direction': direction
                    }

        elif direction == 'short':
            # Looking for M5 close BELOW the level (close back inside from above)
            if bar['close'] < level:
                if i + 1 < len(m5_window):
                    entry_bar = m5_window.iloc[i + 1]
                    entry_time = m5_window.index[i + 1]
                    entry_price = entry_bar['open']  # sell at bid

                    sl = extreme + 0.0001  # 1 pip above sweep high
                    risk = sl - entry_price
                    if risk <= 0 or risk < 0.0002:
                        return None
                    tp = entry_price - 1.5 * risk

                    return {
                        'entry_time': str(entry_time),
                        'entry_price': entry_price,
                        'sl': sl,
                        'tp': tp,
                        'risk_pips': risk / 0.0001,
                        'direction': direction
                    }

    return None


def simulate_trade(m5, trade_info):
    """Simulate trade outcome using M5 data."""
    entry_time = pd.Timestamp(trade_info['entry_time'])
    direction = trade_info['direction']
    sl = trade_info['sl']
    tp = trade_info['tp']
    entry_price = trade_info['entry_price']

    # Get M5 bars from entry onward (max 48 bars = 4 hours timeout)
    future = m5[(m5.index >= entry_time) & (m5.index <= entry_time + pd.Timedelta(hours=4))]

    for i in range(len(future)):
        bar = future.iloc[i]

        if direction == 'long':
            # Check SL first (conservative)
            if bar['low'] <= sl:
                return -1.0, 'sl_hit'
            if bar['high'] >= tp:
                return 1.5, 'tp_hit'

        elif direction == 'short':
            if bar['high'] >= sl:
                return -1.0, 'sl_hit'
            if bar['low'] <= tp:
                return 1.5, 'tp_hit'

    # Timeout - close at last bar's close
    last_close = future.iloc[-1]['close'] if len(future) > 0 else entry_price
    if direction == 'long':
        pnl_r = (last_close - entry_price) / (entry_price - sl)
    else:
        pnl_r = (entry_price - last_close) / (sl - entry_price)
    return round(pnl_r, 3), 'timeout'


def check_gates(trades):
    """Check kill-switch gates."""
    n = len(trades)
    if n == 0:
        return None, None

    wins = sum(1 for t in trades if t['pnl_r'] > 0)
    losses = n - wins
    gross_profit = sum(t['pnl_r'] for t in trades if t['pnl_r'] > 0)
    gross_loss = abs(sum(t['pnl_r'] for t in trades if t['pnl_r'] <= 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 999
    exp = sum(t['pnl_r'] for t in trades) / n

    # Compute max drawdown
    equity = 0
    peak = 0
    dd = 0
    for t in trades:
        equity += t['pnl_r']
        if equity > peak:
            peak = equity
        curr_dd = equity - peak
        if curr_dd < dd:
            dd = curr_dd

    metrics = {
        'N': n, 'wins': wins, 'losses': losses,
        'pf': round(pf, 3), 'expectancy': round(exp, 4),
        'max_drawdown': round(dd, 2), 'win_rate': round(wins / n, 3) if n > 0 else 0
    }

    # Gate checks
    if n >= 40:
        if pf < 1.00 or exp <= 0 or dd <= -6:
            return metrics, 'GATE_A_FAIL'
    if n >= 80:
        if pf < 1.15 or exp < 0.10 or dd <= -8:
            return metrics, 'GATE_B_FAIL'
    if n >= 100:
        if pf >= 1.25 and exp >= 0.10 and dd > -10:
            return metrics, 'GATE_C_PASS'
        else:
            return metrics, 'GATE_C_FAIL'

    return metrics, None


def save_checkpoint(trades, metrics, gate_status, phase):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    cp = {
        'timestamp': datetime.now().isoformat(),
        'phase': phase,
        'n_trades': len(trades),
        'metrics': metrics,
        'gate_status': gate_status,
        'trades': trades[-5:] if trades else []  # last 5 trades only
    }
    cp_file = os.path.join(CHECKPOINT_DIR, f'checkpoint_{phase}.json')
    with open(cp_file, 'w') as f:
        json.dump(cp, f, indent=2)
    print(f"[CHECKPOINT] Saved: {cp_file}")


def main():
    print("=" * 60)
    print("  SCBI_M5 STAGE-2 REAL BACKTEST")
    print("  Architecture: HTF_NY_WINDOW + SCBI_M5")
    print("=" * 60)

    h1, m5 = load_data()

    print("\n[LEVELS] Computing session levels...")
    levels = compute_session_levels(h1)
    print(f"[LEVELS] Computed for {len(levels)} trading days")

    print("\n[SWEEPS] Detecting H1 sweeps in NY Window...")
    sweeps = detect_sweeps_h1(h1, levels)
    print(f"[SWEEPS] Found {len(sweeps)} sweeps in NY Window (08:00-11:00)")

    trades = []
    skipped = 0
    last_trade_date = None

    print("\n[SCBI] Scanning for close-back-inside entries...")
    for sweep in sweeps:
        # Avoid multiple trades same day
        sweep_date = sweep['time'].date()
        if last_trade_date == sweep_date:
            continue

        entry = find_scbi_entry(m5, sweep)
        if entry is None:
            skipped += 1
            continue

        pnl_r, exit_type = simulate_trade(m5, entry)
        trade = {
            'sweep_time': str(sweep['time']),
            'level': sweep['level_name'],
            'direction': sweep['direction'],
            'entry_time': entry['entry_time'],
            'entry_price': round(entry['entry_price'], 5),
            'sl': round(entry['sl'], 5),
            'tp': round(entry['tp'], 5),
            'risk_pips': round(entry['risk_pips'], 1),
            'pnl_r': pnl_r,
            'exit_type': exit_type
        }
        trades.append(trade)
        last_trade_date = sweep_date

        n = len(trades)
        # Periodic checkpoint
        if n in [10, 20, 40, 60, 80, 100]:
            metrics, gate = check_gates(trades)
            save_checkpoint(trades, metrics, gate, f"N{n}")
            print(f"\n[GATE CHECK N={n}] PF={metrics['pf']} Exp={metrics['expectancy']} DD={metrics['max_drawdown']}")

            if gate and 'FAIL' in gate:
                print(f"\n{'='*60}")
                print(f"  KILL SWITCH ACTIVATED: {gate}")
                print(f"  PF={metrics['pf']} Exp={metrics['expectancy']} DD={metrics['max_drawdown']}")
                print(f"{'='*60}")
                save_results(trades, metrics, gate)
                return

    # Final evaluation
    metrics, gate = check_gates(trades)
    if metrics:
        save_checkpoint(trades, metrics, gate, "FINAL")
        print(f"\n{'='*60}")
        print(f"  STAGE-2 COMPLETE")
        print(f"  N={metrics['N']} PF={metrics['pf']} Exp={metrics['expectancy']} DD={metrics['max_drawdown']}")
        print(f"  Gate Status: {gate}")
        print(f"  Skipped sweeps (no SCBI entry): {skipped}")
        print(f"{'='*60}")
        save_results(trades, metrics, gate)
    else:
        print("[ERROR] No trades generated.")
        save_results([], {'N': 0}, 'NO_TRADES')


def save_results(trades, metrics, gate):
    result = {
        'architecture': 'HTF_SWEEP_QUALITY_NY_WINDOW + SWEEP_CLOSE_BACK_INSIDE_M5',
        'dataset': 'EURUSD H1/M5 2022-2025',
        'ny_window': f'{NY_WINDOW_START}:00-{NY_WINDOW_END}:00 NY',
        'spread': f'{SPREAD_PIPS} pips',
        'metrics': metrics,
        'gate_status': gate,
        'all_trades': trades
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[RESULTS] Saved to {RESULTS_FILE}")


if __name__ == '__main__':
    main()
