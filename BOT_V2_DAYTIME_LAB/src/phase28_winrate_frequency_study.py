"""Phase28: Winrate + Frequency Improvement Study - Research Shadow Only."""
import os,sys,json,pandas as pd,numpy as np,pytz
from datetime import datetime,time,timezone as tz
from pathlib import Path

ROOT=Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
LAB=ROOT/"BOT_V2_DAYTIME_LAB"
OUT=LAB/"outputs"/"phase28_winrate_frequency_study"
sys.path.append(str(LAB/"src"))
from phase18_h1_fractal_sweep import H1FractalSweepDetector
from phase18_first_3m_choch import First3MChochDetector

for d in ["preflight","baseline_lock","diagnostics","single_hypothesis_tests",
          "limited_combinations","walk_forward","candidate_selection"]:
    os.makedirs(OUT/d,exist_ok=True)

TZ_NY=pytz.timezone("America/New_York")

def load_m3_2020():
    m=json.load(open(LAB/"data"/"certified_m3"/"M3_CERTIFICATION_METADATA.json"))
    db=pd.read_csv(m['bid_path']);da=pd.read_csv(m['ask_path'])
    db['timestamp']=pd.to_datetime(db['timestamp'],utc=True)
    da['timestamp']=pd.to_datetime(da['timestamp'],utc=True)
    df=pd.merge(db,da,on='timestamp',suffixes=('_bid','_ask'))
    df['timestamp_ny']=df['timestamp'].dt.tz_convert(TZ_NY)
    return df

def load_m3_2015():
    frames=[]
    for y in range(2015,2020):
        p=LAB/"data"/"processed_2015_2019"/"eurusd_m3_from_m1"/str(y)/f"EURUSD_M3_{y}.csv"
        if not p.exists():continue
        df=pd.read_csv(p);df['timestamp']=pd.to_datetime(df['timestamp'],utc=True)
        df['timestamp_ny']=df['timestamp'].dt.tz_convert(TZ_NY)
        rename={}
        for c in df.columns:
            if c.startswith('bid_'):rename[c]=c.replace('bid_','')+'_bid'
            elif c.startswith('ask_'):rename[c]=c.replace('ask_','')+'_ask'
        df=df.rename(columns=rename);frames.append(df)
    return pd.concat(frames,ignore_index=True).sort_values('timestamp').reset_index(drop=True)

def load_news():
    p=ROOT/"data_intake_2015_2019"/"news_eurusd_2015_2019.csv"
    if not p.exists():return pd.DataFrame(columns=['timestamp'])
    df=pd.read_csv(p);col='timestamp_utc' if 'timestamp_utc' in df.columns else 'timestamp'
    df['timestamp']=pd.to_datetime(df[col],utc=True);return df

def gen_signals(df_m3):
    df=df_m3.copy();di=df.set_index('timestamp')
    dh=di.resample('1h', closed='left', label='right').agg({'open_bid':'first','high_bid':'max','low_bid':'min','close_bid':'last','timestamp_ny':'first'}).shift(1).dropna().reset_index()
    di.reset_index(inplace=True)
    sw=H1FractalSweepDetector(params={}).detect_sweeps(dh)
    if sw.empty:return[]
    sw['hour']=sw['timestamp_ny'].dt.hour;sw=sw[(sw['hour']>=6)&(sw['hour']<=16)]
    sigs=First3MChochDetector(params={'sl_buffer':0.5,'max_mins_post_sweep':60}).detect_choch(df,sw)
    if sigs.empty:return[]
    dny=df.set_index('timestamp_ny');sl=[]
    for _,r in sigs.iterrows():
        if r['choch_time'] in dny.index:
            io=dny.index.get_loc(r['choch_time'])
            idx=io.start if isinstance(io,slice) else(io[0] if isinstance(io,np.ndarray) else io)
            sl.append({'index':idx,'type':r['direction'],'sl_custom':r['sl_price']})
    return sl

def backtest(df,sigs,news,cfg,slip=0.0,spadd=0.0):
    trades=[];active=None;nb=set()
    if not news.empty:
        ng=cfg.get('news_guard_mins',30)
        for nt in news['timestamp']:
            for m in range(-ng,ng+1):nb.add((nt+pd.Timedelta(minutes=m)).replace(second=0,microsecond=0))
    st=datetime.strptime(cfg['start_time'],"%H:%M").time()
    et=datetime.strptime(cfg['end_time'],"%H:%M").time()
    ct=datetime.strptime(cfg['mandatory_close_time'],"%H:%M").time()
    si={s['index']:s for s in sigs};td=0;cd=None;sl=slip*0.0001;sa=spadd*0.0001
    bp=cfg.get('body_filter_pct',0.0);mtpd=cfg.get('max_trades_per_day',1)
    for row in df.itertuples():
        i=row.Index;nt=row.timestamp_ny.time();nd=row.timestamp_ny.date()
        if nd!=cd:cd=nd;td=0
        if active:
            if nt>=ct:
                active['status']='FORCED_CLOSE';active['exit_price']=row.close_bid if active['type']=='LONG' else row.close_ask+sa
                active['exit_time']=row.timestamp_ny;trades.append(active);active=None;continue
            if active['type']=='LONG':
                if row.low_bid<=active['sl']:active['status']='SL';active['exit_price']=active['sl'];active['exit_time']=row.timestamp_ny;trades.append(active);active=None;continue
                if row.high_bid>=active['tp']:active['status']='TP';active['exit_price']=active['tp'];active['exit_time']=row.timestamp_ny;trades.append(active);active=None;continue
                if cfg.get('be_r') and not active.get('be_triggered'):
                    if row.high_bid>=active['entry_price']+active['risk']*cfg['be_r']:active['sl']=active['entry_price'];active['be_triggered']=True
            else:
                if row.high_ask+sa>=active['sl']:active['status']='SL';active['exit_price']=active['sl'];active['exit_time']=row.timestamp_ny;trades.append(active);active=None;continue
                if row.low_ask<=active['tp']:active['status']='TP';active['exit_price']=active['tp'];active['exit_time']=row.timestamp_ny;trades.append(active);active=None;continue
                if cfg.get('be_r') and not active.get('be_triggered'):
                    if row.low_bid<=active['entry_price']-active['risk']*cfg['be_r']:active['sl']=active['entry_price'];active['be_triggered']=True
            continue
        if i in si:
            if td>=mtpd:continue
            if not(st<=nt<=et):continue
            if row.timestamp.replace(second=0,microsecond=0) in nb:continue
            if bp>0:
                body=abs(row.close_bid-row.open_bid);wick=row.high_bid-row.low_bid
                if wick>0 and body/wick<bp:continue
            sig=si[i];sb=cfg.get('sl_buffer_pips',0.0)*0.0001
            if sig['type']=='LONG':
                ep=row.close_ask+sa+sl;s=(sig.get('sl_custom') or row.low_bid)-sb;r=ep-s
                if r<=0:continue
                active={'type':'LONG','entry_time':row.timestamp_ny,'entry_price':ep,'sl':s,'tp':ep+r*cfg['tp_r'],'risk':r,'status':'OPEN','be_triggered':False}
            else:
                ep=row.close_bid-sl;s=(sig.get('sl_custom') or row.high_ask)+sb+sa;r=s-ep
                if r<=0:continue
                active={'type':'SHORT','entry_time':row.timestamp_ny,'entry_price':ep,'sl':s,'tp':ep-r*cfg['tp_r'],'risk':r,'status':'OPEN','be_triggered':False}
            td+=1
    return pd.DataFrame(trades)

def metrics(tdf):
    if tdf.empty:return{"sample":0,"pf":0,"exp":0,"wr":0,"max_dd":0,"max_streak":0,"tpm":0,"tp_count":0,"sl_count":0,"be_count":0,"fc":0,"total_r":0}
    df=tdf.copy()
    df['r_return']=df.apply(lambda r:(r['exit_price']-r['entry_price'])/r['risk'] if r['type']=='LONG' else (r['entry_price']-r['exit_price'])/r['risk'],axis=1)
    p=df[df['r_return']>0]['r_return'].sum();l=abs(df[df['r_return']<0]['r_return'].sum())
    df['cum']=df['r_return'].cumsum();dd=(df['cum']-df['cum'].cummax()).min()
    s=0;ms=0
    for r in df['r_return']:
        if r<=0:s+=1;ms=max(ms,s)
        else:s=0
    months=(df['entry_time'].max()-df['entry_time'].min()).days/30.44
    tpm=round(len(df)/months,1) if months>0 else 0
    # months with <15 trades
    df['ym']=df['entry_time'].dt.to_period('M')
    mc=df.groupby('ym').size()
    low_months=int((mc<15).sum())
    return{"sample":len(df),"pf":round(p/l,2) if l>0 else 99,"exp":round(df['r_return'].mean(),3),
           "wr":round(len(df[df['r_return']>0])/len(df)*100,1),"max_dd":round(dd,2),"max_streak":ms,
           "tpm":tpm,"tp_count":int((df['status']=='TP').sum()),"sl_count":int((df['status']=='SL').sum()),
           "be_count":int(df['be_triggered'].sum()) if 'be_triggered' in df.columns else 0,
           "fc":int((df['status']=='FORCED_CLOSE').sum()),"total_r":round(df['r_return'].sum(),2),
           "low_months":low_months}

# ═══════ LOAD DATA ═══════
print("Loading data...")
m3a=load_m3_2015();m3b=load_m3_2020();m3=pd.concat([m3a,m3b],ignore_index=True).sort_values('timestamp').reset_index(drop=True)
news=load_news();sigs=gen_signals(m3)
print(f"Full M3: {len(m3)} | Signals: {len(sigs)}")

# ═══════ BASELINE ═══════
print("\n=== BASELINE Phase25 ===")
BASE={"tp_r":1.4,"be_r":0.4,"start_time":"07:00","end_time":"16:30","mandatory_close_time":"20:00",
      "max_trades_per_day":1,"sl_buffer_pips":0.5,"news_guard_mins":30,"body_filter_pct":0.70}
t0=backtest(m3,sigs,news,BASE);m0=metrics(t0)
print(f"  Baseline: {m0}")

# ═══════ FASE 4: SINGLE HYPOTHESES ═══════
print("\n=== FASE 4: Single Hypotheses ===")
results=[]
results.append({"name":"BASELINE","desc":"Phase25 exact",**m0})

hyps=[
 # A. Window
 ("WIN_07_11",{**BASE,"end_time":"11:00"}),
 ("WIN_07_13",{**BASE,"end_time":"13:00"}),
 ("WIN_08_11",{**BASE,"start_time":"08:00","end_time":"11:00"}),
 ("WIN_08_13",{**BASE,"start_time":"08:00","end_time":"13:00"}),
 ("WIN_08_1630",{**BASE,"start_time":"08:00"}),
 # B. Body filter
 ("BF_60",{**BASE,"body_filter_pct":0.60}),
 ("BF_65",{**BASE,"body_filter_pct":0.65}),
 ("BF_75",{**BASE,"body_filter_pct":0.75}),
 ("BF_80",{**BASE,"body_filter_pct":0.80}),
 # I. TP/BE
 ("TP_1.2",{**BASE,"tp_r":1.2}),
 ("TP_1.3",{**BASE,"tp_r":1.3}),
 ("TP_1.5",{**BASE,"tp_r":1.5}),
 ("TP_1.6",{**BASE,"tp_r":1.6}),
 ("BE_0.3",{**BASE,"be_r":0.3}),
 ("BE_0.5",{**BASE,"be_r":0.5}),
 ("BE_0.6",{**BASE,"be_r":0.6}),
 ("NO_BE",{**BASE,"be_r":None}),
 # H. Second bullet
 ("2TRADES",{**BASE,"max_trades_per_day":2}),
]

for name,cfg in hyps:
    t=backtest(m3,sigs,news,cfg);m=metrics(t)
    results.append({"name":name,"desc":"","**":None,**m})
    print(f"  {name}: PF={m['pf']} WR={m['wr']} Exp={m['exp']} DD={m['max_dd']} Str={m['max_streak']} TPM={m['tpm']} Low={m['low_months']}")

rdf=pd.DataFrame(results)
rdf.to_csv(OUT/"single_hypothesis_tests"/"phase28_single_hypothesis_results.csv",index=False)

# ═══════ FASE 6: BEST COMBINATIONS ═══════
print("\n=== FASE 6: Limited Combinations ===")
combos=[
 ("TP1.2_BE0.3",{**BASE,"tp_r":1.2,"be_r":0.3}),
 ("TP1.3_BE0.3",{**BASE,"tp_r":1.3,"be_r":0.3}),
 ("TP1.2_NOBE",{**BASE,"tp_r":1.2,"be_r":None}),
 ("TP1.3_NOBE",{**BASE,"tp_r":1.3,"be_r":None}),
 ("TP1.2_BF60",{**BASE,"tp_r":1.2,"body_filter_pct":0.60}),
 ("TP1.2_BF65",{**BASE,"tp_r":1.2,"body_filter_pct":0.65}),
 ("TP1.3_BF65",{**BASE,"tp_r":1.3,"body_filter_pct":0.65}),
 ("TP1.2_W0813",{**BASE,"tp_r":1.2,"start_time":"08:00","end_time":"13:00"}),
 ("TP1.3_W0813",{**BASE,"tp_r":1.3,"start_time":"08:00","end_time":"13:00"}),
 ("TP1.2_2T",{**BASE,"tp_r":1.2,"max_trades_per_day":2}),
 ("TP1.3_2T",{**BASE,"tp_r":1.3,"max_trades_per_day":2}),
 ("BF60_2T",{**BASE,"body_filter_pct":0.60,"max_trades_per_day":2}),
 ("TP1.2_BF60_2T",{**BASE,"tp_r":1.2,"body_filter_pct":0.60,"max_trades_per_day":2}),
]
cresults=[]
for name,cfg in combos:
    t=backtest(m3,sigs,news,cfg);m=metrics(t)
    cresults.append({"name":name,**m})
    print(f"  {name}: PF={m['pf']} WR={m['wr']} Exp={m['exp']} DD={m['max_dd']} Str={m['max_streak']} TPM={m['tpm']} Low={m['low_months']}")

pd.DataFrame(cresults).to_csv(OUT/"limited_combinations"/"phase28_limited_combinations_results.csv",index=False)

# ═══════ FASE 7: WALK-FORWARD on best candidates ═══════
print("\n=== FASE 7: Walk-Forward ===")
# Find best balanced from combos
all_c=cresults+[{"name":"BASELINE",**m0}]
# Filter: PF>=2.2, exp>=0.22, wr>m0['wr']
viable=[c for c in all_c if c['pf']>=2.2 and c['exp']>=0.22]
viable.sort(key=lambda x:x['wr'],reverse=True)
best3=viable[:3] if len(viable)>=3 else viable

wf_splits=[
 (range(2015,2019),range(2019,2021),[2021]),
 (range(2016,2020),range(2020,2022),[2022]),
 (range(2017,2021),range(2021,2023),[2023]),
 (range(2018,2022),range(2022,2024),[2024]),
 (range(2019,2023),range(2023,2025),[2025]),
]

wf_results=[]
for cand in best3:
    name=cand['name']
    # Find config
    cfg_map=dict([(n,c) for n,c in hyps+combos])
    cfg=cfg_map.get(name,BASE)
    passes=0
    for train_y,val_y,test_y in wf_splits:
        test_mask=m3['timestamp_ny'].dt.year.isin(test_y)
        m3_test=m3[test_mask].reset_index(drop=True)
        if len(m3_test)==0:continue
        sigs_t=gen_signals(m3_test)
        t=backtest(m3_test,sigs_t,news,cfg)
        mt=metrics(t)
        if mt['pf']>=1.5 and mt['exp']>0:passes+=1
    wf_results.append({"name":name,"wf_passes":passes,"wf_total":len(wf_splits),"wf_pass_rate":round(passes/len(wf_splits)*100)})
    print(f"  WF {name}: {passes}/{len(wf_splits)} passes")

pd.DataFrame(wf_results).to_csv(OUT/"walk_forward"/"phase28_walk_forward_results.csv",index=False)

# ═══════ COST STRESS on best ═══════
print("\n=== Cost Stress on best candidates ===")
cost_results=[]
for cand in best3:
    name=cand['name'];cfg=dict([(n,c) for n,c in hyps+combos]).get(name,BASE)
    for slip in [0.0,0.5,1.0,1.5,2.0]:
        t=backtest(m3,sigs,news,cfg,slip=slip);m=metrics(t)
        cost_results.append({"name":name,"slip":slip,"pf":m['pf'],"exp":m['exp'],"wr":m['wr']})
pd.DataFrame(cost_results).to_csv(OUT/"candidate_selection"/"phase28_cost_stress.csv",index=False)

# ═══════ VERDICT ═══════
print("\n=== VERDICT ===")
best_wr=max(all_c,key=lambda x:x['wr'] if x['pf']>=2.2 else 0)
best_bal=max(viable,key=lambda x:x['wr']*0.4+x['pf']*0.3+x['exp']*100*0.3) if viable else {"name":"BASELINE",**m0}

if best_bal['wr']>m0['wr']+3 and best_bal['pf']>=2.2 and best_bal['exp']>=0.22:
    verdict="PHASE28_BALANCED_IMPROVEMENT_FOUND"
elif best_wr['wr']>m0['wr']+5 and best_wr['pf']>=2.0:
    verdict="PHASE28_WR_IMPROVEMENT_FOUND_WITH_TRADEOFF"
else:
    verdict="PHASE28_NO_SUPERIOR_CANDIDATE_PHASE25_REMAINS_AUTHORITY"

final={"timestamp":datetime.now(tz.utc).isoformat(),"verdict":verdict,
       "baseline":m0,"best_wr":best_wr,"best_balanced":best_bal,
       "all_hypotheses":results,"combinations":cresults,"walk_forward":wf_results}
with open(LAB/"reports"/"PHASE28_WINRATE_FREQUENCY_IMPROVEMENT_STUDY_REPORT.json","w") as f:
    json.dump(final,f,indent=2,default=str)

print(f"\nBaseline: WR={m0['wr']} PF={m0['pf']} Exp={m0['exp']}")
print(f"Best WR: {best_wr['name']} WR={best_wr['wr']} PF={best_wr['pf']} Exp={best_wr['exp']}")
print(f"Best Balanced: {best_bal['name']} WR={best_bal['wr']} PF={best_bal['pf']} Exp={best_bal['exp']}")
print(f"\n=== VEREDICTO: {verdict} ===")
print("DONE")
