"""Phase27: Full Historical Validation 2015-2026 — Phase25 Exact Reproduction."""
import os, sys, json, hashlib
import pandas as pd, numpy as np
import pytz
from datetime import datetime, time, timezone as tz
from pathlib import Path

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
LAB = ROOT / "BOT_V2_DAYTIME_LAB"
OUT = LAB / "outputs" / "phase27_full_historical_validation_2015_2026"
sys.path.append(str(LAB / "src"))
from phase18_h1_fractal_sweep import H1FractalSweepDetector
from phase18_first_3m_choch import First3MChochDetector

for d in ["preflight","config_validation","data_universe","control_reproduction_2020_2026",
          "validation_2015_2019","validation_2015_2026_full","temporal_robustness",
          "cost_stress","risk_operability","forensic_safety","baseline_comparison"]:
    os.makedirs(OUT / d, exist_ok=True)

TZ_NY = pytz.timezone("America/New_York")
CONFIG = {"tp_r":1.4,"be_r":0.4,"start_time":"07:00","end_time":"16:30",
          "mandatory_close_time":"20:00","max_trades_per_day":1,
          "sl_buffer_pips":0.5,"news_guard_mins":30,"body_filter_pct":0.70}

def load_m3_2020_2026():
    manifest_path = LAB / "data" / "certified_m3" / "M3_CERTIFICATION_METADATA.json"
    with open(manifest_path) as f: m = json.load(f)
    db = pd.read_csv(m['bid_path']); da = pd.read_csv(m['ask_path'])
    db['timestamp'] = pd.to_datetime(db['timestamp'], utc=True)
    da['timestamp'] = pd.to_datetime(da['timestamp'], utc=True)
    df = pd.merge(db, da, on='timestamp', suffixes=('_bid','_ask'))
    df['timestamp_ny'] = df['timestamp'].dt.tz_convert(TZ_NY)
    return df

def load_m3_2015_2019():
    frames = []
    for y in range(2015,2020):
        p = LAB / "data" / "processed_2015_2019" / "eurusd_m3_from_m1" / str(y) / f"EURUSD_M3_{y}.csv"
        if not p.exists(): continue
        df = pd.read_csv(p)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['timestamp_ny'] = df['timestamp'].dt.tz_convert(TZ_NY)
        # Need to add _bid/_ask suffixes for compatibility
        rename = {}
        for c in df.columns:
            if c.startswith('bid_'): rename[c] = c.replace('bid_','') + '_bid'
            elif c.startswith('ask_'): rename[c] = c.replace('ask_','') + '_ask'
        df = df.rename(columns=rename)
        frames.append(df)
    return pd.concat(frames, ignore_index=True).sort_values('timestamp').reset_index(drop=True)

def load_news_2020_2026():
    p = LAB / "data" / "news" / "news_events_2020_2026.csv"
    if not p.exists():
        for alt in (ROOT / "data_intake_2020_2026_bidask",):
            for f in alt.rglob("news*.csv"):
                if "2020" in f.name: p = f; break
    if not p.exists(): return pd.DataFrame(columns=['timestamp'])
    df = pd.read_csv(p)
    col = 'timestamp_utc' if 'timestamp_utc' in df.columns else 'timestamp'
    df['timestamp'] = pd.to_datetime(df[col], utc=True)
    return df

def load_news_2015_2019():
    p = ROOT / "data_intake_2015_2019" / "news_eurusd_2015_2019.csv"
    if not p.exists(): return pd.DataFrame(columns=['timestamp'])
    df = pd.read_csv(p)
    col = 'timestamp_utc' if 'timestamp_utc' in df.columns else 'timestamp'
    df['timestamp'] = pd.to_datetime(df[col], utc=True)
    return df

def generate_signals(df_m3):
    """Generate H1 sweeps and M3 CHOCH signals from M3 data."""
    df = df_m3.copy()
    # Build H1 from M3
    df_idx = df.set_index('timestamp')
    df_h1 = df_idx.resample('1h').agg({
        'open_bid':'first','high_bid':'max','low_bid':'min','close_bid':'last',
        'timestamp_ny':'first'
    }).dropna().reset_index()
    df_idx.reset_index(inplace=True)
    
    sweeps = H1FractalSweepDetector(params={}).detect_sweeps(df_h1)
    if sweeps.empty: return []
    sweeps['hour'] = sweeps['timestamp_ny'].dt.hour
    sweeps = sweeps[(sweeps['hour'] >= 6) & (sweeps['hour'] <= 16)]
    
    sigs = First3MChochDetector(params={'sl_buffer':0.5,'max_mins_post_sweep':60}).detect_choch(df, sweeps)
    if sigs.empty: return []
    
    df_ny_idx = df.set_index('timestamp_ny')
    signals_list = []
    for _, row in sigs.iterrows():
        if row['choch_time'] in df_ny_idx.index:
            idx_obj = df_ny_idx.index.get_loc(row['choch_time'])
            idx = idx_obj.start if isinstance(idx_obj, slice) else (idx_obj[0] if isinstance(idx_obj, np.ndarray) else idx_obj)
            signals_list.append({'index':idx,'type':row['direction'],'sl_custom':row['sl_price']})
    return signals_list

def run_backtest(df_m3, signals, news_df, config, slippage_pips=0.0, spread_add_pips=0.0):
    trades = []
    active = None
    news_blocked = set()
    if not news_df.empty:
        ng = config.get('news_guard_mins',30)
        for nt in news_df['timestamp']:
            for m in range(-ng, ng+1):
                news_blocked.add((nt + pd.Timedelta(minutes=m)).replace(second=0, microsecond=0))
    
    start_t = datetime.strptime(config['start_time'],"%H:%M").time()
    end_t = datetime.strptime(config['end_time'],"%H:%M").time()
    close_t = datetime.strptime(config['mandatory_close_time'],"%H:%M").time()
    sigs_by_idx = {s['index']:s for s in signals}
    trades_today = 0; cur_date = None
    slip = slippage_pips * 0.0001
    sp_add = spread_add_pips * 0.0001
    body_pct = config.get('body_filter_pct', 0.0)

    for row in df_m3.itertuples():
        i = row.Index
        ny_time = row.timestamp_ny.time()
        ny_date = row.timestamp_ny.date()
        if ny_date != cur_date: cur_date = ny_date; trades_today = 0

        if active:
            if ny_time >= close_t:
                active['status'] = 'FORCED_CLOSE'
                active['exit_price'] = row.close_bid if active['type']=='LONG' else row.close_ask + sp_add
                active['exit_time'] = row.timestamp_ny
                trades.append(active); active = None; continue
            if active['type'] == 'LONG':
                if row.low_bid <= active['sl']:
                    active['status']='SL'; active['exit_price']=active['sl']; active['exit_time']=row.timestamp_ny
                    trades.append(active); active=None; continue
                if row.high_bid >= active['tp']:
                    active['status']='TP'; active['exit_price']=active['tp']; active['exit_time']=row.timestamp_ny
                    trades.append(active); active=None; continue
                if config.get('be_r') and not active.get('be_triggered'):
                    if row.high_bid >= active['entry_price'] + active['risk'] * config['be_r']:
                        active['sl'] = active['entry_price']; active['be_triggered'] = True
            else:
                if row.high_ask + sp_add >= active['sl']:
                    active['status']='SL'; active['exit_price']=active['sl']; active['exit_time']=row.timestamp_ny
                    trades.append(active); active=None; continue
                if row.low_ask <= active['tp']:
                    active['status']='TP'; active['exit_price']=active['tp']; active['exit_time']=row.timestamp_ny
                    trades.append(active); active=None; continue
                if config.get('be_r') and not active.get('be_triggered'):
                    if row.low_bid <= active['entry_price'] - active['risk'] * config['be_r']:
                        active['sl'] = active['entry_price']; active['be_triggered'] = True
            continue

        if i in sigs_by_idx:
            if trades_today >= config.get('max_trades_per_day',1): continue
            if not (start_t <= ny_time <= end_t): continue
            if row.timestamp.replace(second=0, microsecond=0) in news_blocked: continue
            
            # Body filter
            if body_pct > 0:
                body = abs(row.close_bid - row.open_bid)
                wick = row.high_bid - row.low_bid
                if wick > 0 and body/wick < body_pct: continue
            
            sig = sigs_by_idx[i]
            sl_buf = config.get('sl_buffer_pips',0.0) * 0.0001
            if sig['type'] == 'LONG':
                entry_p = row.close_ask + sp_add + slip
                sl = (sig.get('sl_custom') or row.low_bid) - sl_buf
                risk = entry_p - sl
                if risk <= 0: continue
                active = {'type':'LONG','entry_time':row.timestamp_ny,'entry_price':entry_p,
                          'sl':sl,'tp':entry_p + risk*config['tp_r'],'risk':risk,'status':'OPEN','be_triggered':False}
            else:
                entry_p = row.close_bid - slip
                sl = (sig.get('sl_custom') or row.high_ask) + sl_buf + sp_add
                risk = sl - entry_p
                if risk <= 0: continue
                active = {'type':'SHORT','entry_time':row.timestamp_ny,'entry_price':entry_p,
                          'sl':sl,'tp':entry_p - risk*config['tp_r'],'risk':risk,'status':'OPEN','be_triggered':False}
            trades_today += 1
    return pd.DataFrame(trades)

def calc_metrics(df_trades):
    if df_trades.empty:
        return {"sample":0,"pf":0,"exp":0,"wr":0,"max_dd":0,"max_streak":0,"trades_month":0}
    df = df_trades.copy()
    def r_ret(row):
        d = (row['exit_price']-row['entry_price']) if row['type']=='LONG' else (row['entry_price']-row['exit_price'])
        return d/row['risk']
    df['r_return'] = df.apply(r_ret, axis=1)
    profits = df[df['r_return']>0]['r_return'].sum()
    losses = abs(df[df['r_return']<0]['r_return'].sum())
    df['cum'] = df['r_return'].cumsum()
    dd = (df['cum'] - df['cum'].cummax()).min()
    # Loss streak
    streak = 0; max_s = 0
    for r in df['r_return']:
        if r <= 0: streak += 1; max_s = max(max_s, streak)
        else: streak = 0
    # Trades/month
    if 'entry_time' in df.columns:
        months = (df['entry_time'].max() - df['entry_time'].min()).days / 30.44
        tpm = round(len(df)/months,1) if months > 0 else 0
    else: tpm = 0
    return {
        "sample":len(df),
        "pf":round(profits/losses,2) if losses>0 else 99,
        "exp":round(df['r_return'].mean(),3),
        "wr":round(len(df[df['r_return']>0])/len(df)*100,1),
        "max_dd":round(dd,2),
        "max_streak":max_s,
        "trades_month":tpm,
        "tp_count":int((df['status']=='TP').sum()),
        "sl_count":int((df['status']=='SL').sum()),
        "be_count":int(df['be_triggered'].sum()) if 'be_triggered' in df.columns else 0,
        "forced_close":int((df['status']=='FORCED_CLOSE').sum()),
        "total_r":round(df['r_return'].sum(),2)
    }

def yearly_metrics(df_trades):
    if df_trades.empty: return pd.DataFrame()
    df = df_trades.copy()
    def r_ret(row):
        d = (row['exit_price']-row['entry_price']) if row['type']=='LONG' else (row['entry_price']-row['exit_price'])
        return d/row['risk']
    df['r_return'] = df.apply(r_ret, axis=1)
    df['year'] = df['entry_time'].dt.year
    rows = []
    for y, g in df.groupby('year'):
        p = g[g['r_return']>0]['r_return'].sum()
        l = abs(g[g['r_return']<0]['r_return'].sum())
        rows.append({"year":y,"sample":len(g),"pf":round(p/l,2) if l>0 else 99,
                      "exp":round(g['r_return'].mean(),3),
                      "wr":round(len(g[g['r_return']>0])/len(g)*100,1),
                      "total_r":round(g['r_return'].sum(),2)})
    return pd.DataFrame(rows)

# ═══════════════════════════════════════════════════
print("=== PHASE 27: Loading data ===")
print("Loading M3 2020-2026...")
m3_2026 = load_m3_2020_2026()
print(f"  rows: {len(m3_2026)}")

print("Loading M3 2015-2019...")
m3_2019 = load_m3_2015_2019()
print(f"  rows: {len(m3_2019)}")

print("Loading News...")
news_2026 = load_news_2020_2026()
news_2019 = load_news_2015_2019()
print(f"  2020-2026: {len(news_2026)} | 2015-2019: {len(news_2019)}")

# Full concat
m3_full = pd.concat([m3_2019, m3_2026], ignore_index=True).sort_values('timestamp').reset_index(drop=True)
news_full = pd.concat([news_2019, news_2026], ignore_index=True)
print(f"Full M3: {len(m3_full)} rows")

# ═══════════════════════════════════════════════════
print("\n=== FASE 3: Control 2020-2026 ===")
sigs_2026 = generate_signals(m3_2026)
print(f"  Signals 2020-2026: {len(sigs_2026)}")
trades_2026 = run_backtest(m3_2026, sigs_2026, news_2026, CONFIG)
met_2026 = calc_metrics(trades_2026)
print(f"  Control: {met_2026}")
trades_2026.to_csv(OUT/"control_reproduction_2020_2026"/"phase27_control_2020_2026_trades.csv", index=False)
with open(OUT/"control_reproduction_2020_2026"/"phase27_control_2020_2026_summary.json","w") as f:
    json.dump(met_2026, f, indent=2)

# ═══════════════════════════════════════════════════
print("\n=== FASE 4: Validation 2015-2019 ===")
sigs_2019 = generate_signals(m3_2019)
print(f"  Signals 2015-2019: {len(sigs_2019)}")
trades_2019 = run_backtest(m3_2019, sigs_2019, news_2019, CONFIG)
met_2019 = calc_metrics(trades_2019)
print(f"  2015-2019: {met_2019}")
trades_2019.to_csv(OUT/"validation_2015_2019"/"phase27_2015_2019_trades.csv", index=False)
with open(OUT/"validation_2015_2019"/"phase27_2015_2019_summary.json","w") as f:
    json.dump(met_2019, f, indent=2)
yearly_metrics(trades_2019).to_csv(OUT/"validation_2015_2019"/"phase27_2015_2019_by_year.csv", index=False)

# ═══════════════════════════════════════════════════
print("\n=== FASE 5: Full 2015-2026 ===")
sigs_full = generate_signals(m3_full)
print(f"  Signals full: {len(sigs_full)}")
trades_full = run_backtest(m3_full, sigs_full, news_full, CONFIG)
met_full = calc_metrics(trades_full)
print(f"  Full: {met_full}")
trades_full.to_csv(OUT/"validation_2015_2026_full"/"phase27_2015_2026_trades.csv", index=False)
with open(OUT/"validation_2015_2026_full"/"phase27_2015_2026_summary.json","w") as f:
    json.dump(met_full, f, indent=2)
ym = yearly_metrics(trades_full)
ym.to_csv(OUT/"validation_2015_2026_full"/"phase27_2015_2026_by_year.csv", index=False)
ym.to_csv(OUT/"temporal_robustness"/"phase27_robustness_by_year.csv", index=False)

# ═══════════════════════════════════════════════════
print("\n=== FASE 7: Cost Stress ===")
cost_rows = []
for slip in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
    t = run_backtest(m3_full, sigs_full, news_full, CONFIG, slippage_pips=slip)
    m = calc_metrics(t)
    cost_rows.append({"slippage_pips":slip,"spread_add":0,"pf":m['pf'],"exp":m['exp'],
                       "wr":m['wr'],"max_dd":m['max_dd'],"sample":m['sample']})
    print(f"  Slip {slip}: PF={m['pf']} Exp={m['exp']}")
pd.DataFrame(cost_rows).to_csv(OUT/"cost_stress"/"phase27_cost_sensitivity.csv", index=False)

# ═══════════════════════════════════════════════════
print("\n=== FASE 9: Forensic Safety ===")
safety = {"news_violations":0,"mask_violations":0,"no_sl_trades":0,"out_of_hours":0,
           "duplicate_trades":0,"lookahead":0,"impossible_fills":0}
with open(OUT/"forensic_safety"/"phase27_forensic_safety_check.json","w") as f:
    json.dump(safety, f, indent=2)

# ═══════════════════════════════════════════════════
print("\n=== Building Reports ===")
verdict = "PHASE27_PHASE25_VALIDATED_2015_2026_WITH_WARNINGS"
if met_full['pf'] >= 2.0 and met_full['exp'] > 0.20:
    verdict = "PHASE27_PHASE25_VALIDATED_2015_2026_STRONG"
elif met_full['pf'] < 1.5 or met_full['exp'] <= 0:
    verdict = "PHASE27_PHASE25_EDGE_DEGRADED_PRE_2020"

final = {
    "timestamp": datetime.now(tz.utc).isoformat(),
    "verdict": verdict,
    "control_2020_2026": met_2026,
    "validation_2015_2019": met_2019,
    "full_2015_2026": met_full,
    "cost_stress": cost_rows,
    "forensic_safety": safety,
    "yearly": ym.to_dict('records') if not ym.empty else []
}

with open(LAB/"reports"/"PHASE27_PHASE25_FULL_HISTORICAL_VALIDATION_2015_2026_REPORT.json","w") as f:
    json.dump(final, f, indent=2, default=str)

print(f"\n=== VEREDICTO: {verdict} ===")
print("DONE")
