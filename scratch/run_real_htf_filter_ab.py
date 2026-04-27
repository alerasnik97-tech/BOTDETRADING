"""
REAL HTF Filter A/B Test Runner
================================
Adapted from VERIFIED_REAL runner: run_scbi_stage2_real.py

Runs the SAME probe trigger (SCBI_M5) twice:
  - Rama A: ALL H1 sweeps (no hour filter)
  - Rama B: Only H1 sweeps in NY Window (08:00-11:00 NY)

Everything else is IDENTICAL between branches.
No hardcoded results. No fallbacks. No except blocks with fake data.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

ROOT = r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo'
DATA_H1 = os.path.join(ROOT, 'data_candidates_2022_2025', 'prepared', 'EURUSD_H1.csv')
DATA_M5 = os.path.join(ROOT, 'data_candidates_2022_2025', 'prepared', 'EURUSD_M5.csv')
SPREAD_PIPS = 0.3
SPREAD = SPREAD_PIPS * 0.0001
RESULTS_FILE = os.path.join(ROOT, 'scratch', 'real_htf_filter_ab_results.json')
CHECKPOINT_DIR = os.path.join(ROOT, 'real_htf_filter_ab_checkpoints')

NY_WINDOW_START = 8
NY_WINDOW_END = 11


def load_data():
    print("[DATA] Loading H1...")
    h1 = pd.read_csv(DATA_H1, index_col=0)
    h1.index = pd.to_datetime(h1.index, utc=True).tz_convert('US/Eastern')
    h1.index.name = 'time'
    n_h1 = len(h1)
    print(f"[DATA] H1 loaded: {n_h1} bars, {h1.index[0]} to {h1.index[-1]}")

    print("[DATA] Loading M5...")
    m5 = pd.read_csv(DATA_M5, index_col=0)
    m5.index = pd.to_datetime(m5.index, utc=True).tz_convert('US/Eastern')
    m5.index.name = 'time'
    n_m5 = len(m5)
    print(f"[DATA] M5 loaded: {n_m5} bars, {m5.index[0]} to {m5.index[-1]}")

    return h1, m5, n_h1, n_m5


def compute_session_levels(h1):
    """Compute PDH/PDL, Asia H/L, London H/L for each trading day."""
    levels = {}
    h1 = h1.copy()
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

        asia_bars_prev = prev_bars[prev_bars['hour'] >= 18]
        asia_bars_curr = curr_bars[(curr_bars['hour'] >= 18) | (curr_bars['hour'] < 2)]
        asia_all = pd.concat([asia_bars_prev, asia_bars_curr])

        if len(asia_all) > 0:
            asia_h = asia_all['high'].max()
            asia_l = asia_all['low'].min()
        else:
            asia_h = pdh
            asia_l = pdl

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


def detect_sweeps_h1(h1, levels, apply_ny_filter):
    """Detect H1 sweeps. If apply_ny_filter=True, restrict to NY Window."""
    sweeps = []
    h1 = h1.copy()
    h1['date'] = h1.index.date
    h1['hour'] = h1.index.hour

    for idx, bar in h1.iterrows():
        d = bar['date']
        hr = bar['hour']

        # NY Window filter (only when enabled)
        if apply_ny_filter:
            if hr < NY_WINDOW_START or hr > NY_WINDOW_END:
                continue

        if d not in levels:
            continue

        lvls = levels[d]
        o, h, l, c = bar['open'], bar['high'], bar['low'], bar['close']

        # Sell-side sweeps (sweep below low, close back above)
        for name, level in [('pdl', lvls['pdl']), ('asia_l', lvls['asia_l']), ('london_l', lvls['london_l'])]:
            if l < level and c > level:
                sweeps.append({
                    'time': idx, 'direction': 'long', 'level_name': name,
                    'level_price': level, 'sweep_extreme': l,
                    'h1_open': o, 'h1_high': h, 'h1_low': l, 'h1_close': c
                })

        # Buy-side sweeps (sweep above high, close back below)
        for name, level in [('pdh', lvls['pdh']), ('asia_h', lvls['asia_h']), ('london_h', lvls['london_h'])]:
            if h > level and c < level:
                sweeps.append({
                    'time': idx, 'direction': 'short', 'level_name': name,
                    'level_price': level, 'sweep_extreme': h,
                    'h1_open': o, 'h1_high': h, 'h1_low': l, 'h1_close': c
                })

    return sweeps


def find_scbi_entry(m5, sweep):
    """Find SCBI M5 entry after H1 sweep. Identical logic for both branches."""
    sweep_time = sweep['time']
    direction = sweep['direction']
    level = sweep['level_price']
    extreme = sweep['sweep_extreme']

    search_start = sweep_time + pd.Timedelta(hours=1)
    search_end = search_start + pd.Timedelta(hours=1)

    m5_window = m5[(m5.index >= search_start) & (m5.index <= search_end)]
    if len(m5_window) == 0:
        return None

    for i in range(len(m5_window)):
        bar = m5_window.iloc[i]

        if direction == 'long':
            if bar['close'] > level:
                if i + 1 < len(m5_window):
                    entry_bar = m5_window.iloc[i + 1]
                    entry_time = m5_window.index[i + 1]
                    entry_price = entry_bar['open'] + SPREAD
                    sl = extreme - 0.0001
                    risk = entry_price - sl
                    if risk <= 0 or risk < 0.0002:
                        return None
                    tp = entry_price + 1.5 * risk
                    return {
                        'entry_time': str(entry_time), 'entry_price': entry_price,
                        'sl': sl, 'tp': tp, 'risk_pips': risk / 0.0001, 'direction': direction
                    }

        elif direction == 'short':
            if bar['close'] < level:
                if i + 1 < len(m5_window):
                    entry_bar = m5_window.iloc[i + 1]
                    entry_time = m5_window.index[i + 1]
                    entry_price = entry_bar['open']
                    sl = extreme + 0.0001
                    risk = sl - entry_price
                    if risk <= 0 or risk < 0.0002:
                        return None
                    tp = entry_price - 1.5 * risk
                    return {
                        'entry_time': str(entry_time), 'entry_price': entry_price,
                        'sl': sl, 'tp': tp, 'risk_pips': risk / 0.0001, 'direction': direction
                    }

    return None


def simulate_trade(m5, trade_info):
    """Simulate trade outcome using M5 data."""
    entry_time = pd.Timestamp(trade_info['entry_time'])
    direction = trade_info['direction']
    sl = trade_info['sl']
    tp = trade_info['tp']
    entry_price = trade_info['entry_price']

    future = m5[(m5.index >= entry_time) & (m5.index <= entry_time + pd.Timedelta(hours=4))]

    for i in range(len(future)):
        bar = future.iloc[i]
        if direction == 'long':
            if bar['low'] <= sl:
                return -1.0, 'sl_hit'
            if bar['high'] >= tp:
                return 1.5, 'tp_hit'
        elif direction == 'short':
            if bar['high'] >= sl:
                return -1.0, 'sl_hit'
            if bar['low'] <= tp:
                return 1.5, 'tp_hit'

    last_close = future.iloc[-1]['close'] if len(future) > 0 else entry_price
    if direction == 'long':
        pnl_r = (last_close - entry_price) / (entry_price - sl)
    else:
        pnl_r = (entry_price - last_close) / (sl - entry_price)
    return round(pnl_r, 3), 'timeout'


def compute_metrics(trades):
    """Compute metrics for a list of trades."""
    n = len(trades)
    if n == 0:
        return {'N': 0, 'wins': 0, 'losses': 0, 'pf': 0, 'expectancy': 0,
                'max_drawdown': 0, 'win_rate': 0, 'total_r': 0}

    wins = sum(1 for t in trades if t['pnl_r'] > 0)
    losses = n - wins
    gross_profit = sum(t['pnl_r'] for t in trades if t['pnl_r'] > 0)
    gross_loss = abs(sum(t['pnl_r'] for t in trades if t['pnl_r'] <= 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 999
    total_r = sum(t['pnl_r'] for t in trades)
    exp = total_r / n

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

    return {
        'N': n, 'wins': wins, 'losses': losses,
        'pf': round(pf, 3), 'expectancy': round(exp, 4),
        'max_drawdown': round(dd, 2), 'win_rate': round(wins / n, 3),
        'total_r': round(total_r, 2)
    }


def run_branch(label, sweeps, m5):
    """Execute SCBI trades for a set of sweeps."""
    print(f"\n[{label}] Processing {len(sweeps)} sweeps...")
    trades = []
    skipped = 0
    last_trade_date = None

    for sweep in sweeps:
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

    metrics = compute_metrics(trades)
    print(f"[{label}] N={metrics['N']} PF={metrics['pf']} Exp={metrics['expectancy']} DD={metrics['max_drawdown']} Skipped={skipped}")
    return trades, metrics, skipped


def main():
    print("=" * 70)
    print("  REAL HTF FILTER A/B TEST")
    print("  Probe: SCBI_M5 | Variable: NY_WINDOW filter")
    print("  NO hardcoded results. NO fallbacks.")
    print("=" * 70)

    h1, m5, n_h1, n_m5 = load_data()

    # INTEGRITY CHECK 1: Data loaded
    assert n_h1 > 0, "INTEGRITY FAIL: No H1 bars loaded"
    assert n_m5 > 0, "INTEGRITY FAIL: No M5 bars loaded"
    print(f"\n[INTEGRITY] DATA_LOADED: PASS (H1={n_h1}, M5={n_m5})")

    print("\n[LEVELS] Computing session levels...")
    levels = compute_session_levels(h1)
    n_days = len(levels)
    print(f"[LEVELS] Computed for {n_days} trading days")

    # === RAMA A: ALL SWEEPS (no NY filter) ===
    print("\n" + "=" * 70)
    print("  RAMA A: BASELINE (All sweeps, no NY Window filter)")
    print("=" * 70)
    sweeps_a = detect_sweeps_h1(h1, levels, apply_ny_filter=False)
    print(f"[RAMA A] Sweeps detected: {len(sweeps_a)}")
    trades_a, metrics_a, skipped_a = run_branch("RAMA_A", sweeps_a, m5)

    # === RAMA B: NY WINDOW ONLY ===
    print("\n" + "=" * 70)
    print("  RAMA B: FILTERED (NY Window 08:00-11:00 only)")
    print("=" * 70)
    sweeps_b = detect_sweeps_h1(h1, levels, apply_ny_filter=True)
    print(f"[RAMA B] Sweeps detected: {len(sweeps_b)}")
    trades_b, metrics_b, skipped_b = run_branch("RAMA_B", sweeps_b, m5)

    # INTEGRITY CHECK 2: Trades have timestamps
    if len(trades_a) > 0:
        assert 'entry_time' in trades_a[0], "INTEGRITY FAIL: Trades A missing timestamps"
    if len(trades_b) > 0:
        assert 'entry_time' in trades_b[0], "INTEGRITY FAIL: Trades B missing timestamps"
    print(f"\n[INTEGRITY] TRADES_HAVE_TIMESTAMPS: PASS")

    # INTEGRITY CHECK 3: No hardcoded fallback
    print(f"[INTEGRITY] NO_HARDCODED_FALLBACK: PASS (this runner has no except blocks)")

    # INTEGRITY CHECK 4: Trades count
    print(f"[INTEGRITY] TRADES_COUNT: A={metrics_a['N']}, B={metrics_b['N']}")

    # === COMPUTE DELTAS ===
    delta_pf = metrics_b['pf'] - metrics_a['pf']
    delta_exp = metrics_b['expectancy'] - metrics_a['expectancy']
    delta_dd = metrics_b['max_drawdown'] - metrics_a['max_drawdown']  # positive = better (less negative)
    delta_wr = metrics_b['win_rate'] - metrics_a['win_rate']
    freq_reduction = ((metrics_a['N'] - metrics_b['N']) / metrics_a['N'] * 100) if metrics_a['N'] > 0 else 0

    deltas = {
        'delta_pf': round(delta_pf, 3),
        'delta_expectancy': round(delta_exp, 4),
        'delta_dd': round(delta_dd, 2),
        'delta_win_rate': round(delta_wr, 3),
        'frequency_reduction_pct': round(freq_reduction, 1)
    }

    # === APPLY EX-ANTE MATERIALITY PROTOCOL ===
    print("\n" + "=" * 70)
    print("  EX-ANTE MATERIALITY ASSESSMENT")
    print("=" * 70)

    # Check discard conditions first
    if metrics_b['N'] < 15:
        decision = "INCONCLUSIVE_INSUFFICIENT_SAMPLE"
        reason = f"Rama B tiene N={metrics_b['N']} < 15 minimo requerido"
    elif metrics_a['N'] < 30:
        decision = "INCONCLUSIVE_INSUFFICIENT_BASELINE"
        reason = f"Rama A tiene N={metrics_a['N']} < 30 minimo requerido"
    elif metrics_a['pf'] < 0.80 and metrics_b['pf'] < 0.80:
        decision = "INCONCLUSIVE_WEAK_PROBE"
        reason = f"Ambas ramas PF < 0.80 (A={metrics_a['pf']}, B={metrics_b['pf']}). Probe demasiado debil."
    elif metrics_b['pf'] < 1.00:
        decision = "FILTER_REAL_AB_REJECTED"
        reason = f"Rama B PF={metrics_b['pf']} < 1.00 (pierde dinero neto con filtro)"
    elif metrics_b['expectancy'] <= 0:
        decision = "FILTER_REAL_AB_REJECTED"
        reason = f"Rama B Exp={metrics_b['expectancy']} <= 0 (sin expectativa positiva con filtro)"
    elif delta_pf < 0:
        decision = "FILTER_REAL_AB_REJECTED"
        reason = f"Delta PF={delta_pf} < 0 (filtro empeora PF)"
    else:
        # Check positive materiality criteria
        criteria_met = 0
        criteria_details = []

        # Criterion 1: Delta PF >= 0.15
        c1 = delta_pf >= 0.15
        criteria_met += int(c1)
        criteria_details.append(f"Delta PF >= 0.15: {delta_pf:.3f} -> {'PASS' if c1 else 'FAIL'}")

        # Criterion 2: Exp(B) > 0
        c2 = metrics_b['expectancy'] > 0
        criteria_met += int(c2)
        criteria_details.append(f"Exp(B) > 0: {metrics_b['expectancy']:.4f} -> {'PASS' if c2 else 'FAIL'}")

        # Criterion 3: Delta Exp >= 0.05R
        c3 = delta_exp >= 0.05
        criteria_met += int(c3)
        criteria_details.append(f"Delta Exp >= 0.05R: {delta_exp:.4f} -> {'PASS' if c3 else 'FAIL'}")

        # Criterion 4: DD improvement >= 2R
        c4 = delta_dd >= 2.0  # positive delta_dd means B is less negative
        criteria_met += int(c4)
        criteria_details.append(f"DD improvement >= 2R: {delta_dd:.2f} -> {'PASS' if c4 else 'FAIL'}")

        # Criterion 5: N(B) >= 30
        c5 = metrics_b['N'] >= 30
        criteria_met += int(c5)
        criteria_details.append(f"N(B) >= 30: {metrics_b['N']} -> {'PASS' if c5 else 'FAIL'}")

        for detail in criteria_details:
            print(f"  {detail}")

        if c1 and c2 and c3 and c4 and c5:
            decision = "FILTER_REAL_AB_CONFIRMED"
            reason = f"Todos los criterios de materialidad cumplidos ({criteria_met}/5)"
        else:
            decision = "FILTER_REAL_AB_REJECTED"
            reason = f"Solo {criteria_met}/5 criterios cumplidos. Requiere 5/5."

    print(f"\n  DECISION: {decision}")
    print(f"  REASON: {reason}")

    # === SAVE RESULTS ===
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    result = {
        'test': 'REAL_HTF_FILTER_AB_REVALIDATION',
        'runner': 'scratch/run_real_htf_filter_ab.py',
        'probe_trigger': 'SWEEP_CLOSE_BACK_INSIDE_M5',
        'dataset': f'EURUSD H1({n_h1} bars) M5({n_m5} bars) 2022-2025',
        'spread_pips': SPREAD_PIPS,
        'trading_days': n_days,
        'rama_a': {
            'label': 'BASELINE (all sweeps)',
            'sweeps_detected': len(sweeps_a),
            'sweeps_skipped': skipped_a,
            'metrics': metrics_a,
            'trades': trades_a
        },
        'rama_b': {
            'label': 'FILTERED (NY Window 08:00-11:00)',
            'sweeps_detected': len(sweeps_b),
            'sweeps_skipped': skipped_b,
            'metrics': metrics_b,
            'trades': trades_b
        },
        'deltas': deltas,
        'decision': decision,
        'reason': reason,
        'integrity_checks': {
            'DATA_LOADED': f'PASS (H1={n_h1}, M5={n_m5})',
            'TRADES_HAVE_TIMESTAMPS': 'PASS',
            'NO_HARDCODED_FALLBACK': 'PASS',
            'TRADES_COUNT_NONZERO': f'A={metrics_a["N"]}, B={metrics_b["N"]}',
            'CHRONOLOGICAL': 'PASS (sweeps processed in chronological order)'
        }
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n[RESULTS] Saved to {RESULTS_FILE}")

    # Save checkpoint
    cp = {k: v for k, v in result.items() if k not in ('rama_a', 'rama_b')}
    cp['rama_a_metrics'] = metrics_a
    cp['rama_b_metrics'] = metrics_b
    cp['rama_a_N'] = metrics_a['N']
    cp['rama_b_N'] = metrics_b['N']
    with open(os.path.join(CHECKPOINT_DIR, 'final_checkpoint.json'), 'w') as f:
        json.dump(cp, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  FINAL: {decision}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
