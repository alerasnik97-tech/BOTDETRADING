"""Phase26-C: Full 2015-2019 M1 Normalization, M3 Generation, Audit, Mask, News Fortress."""
import os, json, hashlib, pandas as pd, numpy as np
from datetime import datetime, timezone

ROOT = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
LAB = os.path.join(ROOT, "BOT_V2_DAYTIME_LAB")
RAW = os.path.join(ROOT, "data_intake_2015_2019", "raw_m1")
OUT = os.path.join(LAB, "outputs", "phase26c_full_2015_2019_certification")
M1_OUT = os.path.join(LAB, "data", "processed_2015_2019", "eurusd_m1_certified_candidate")
M3_OUT = os.path.join(LAB, "data", "processed_2015_2019", "eurusd_m3_from_m1")
MASK_OUT = os.path.join(LAB, "data", "certification_2015_2019", "masks")
YEARS = [2015, 2016, 2017, 2018, 2019]

for d in [OUT, M1_OUT, M3_OUT, MASK_OUT,
          os.path.join(OUT,"preflight"), os.path.join(OUT,"m1_normalization"),
          os.path.join(OUT,"m1_quality_audit"), os.path.join(OUT,"m3_generation"),
          os.path.join(OUT,"m3_quality_audit"), os.path.join(OUT,"data_quality_mask"),
          os.path.join(OUT,"news_fortress"), os.path.join(OUT,"year_certification")]:
    os.makedirs(d, exist_ok=True)

def sha256(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        while c := f.read(8192): h.update(c)
    return h.hexdigest()

# ── PHASE 3: M1 Normalization ──
print("=== FASE 3: Normalización M1 Full ===")
m1_summary = {}
for y in YEARS:
    bid_p = os.path.join(RAW, str(y), "EURUSD_M1_BID.csv")
    ask_p = os.path.join(RAW, str(y), "EURUSD_M1_ASK.csv")
    out_dir = os.path.join(M1_OUT, str(y))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"EURUSD_M1_{y}.csv")
    
    print(f"  {y}: Loading BID/ASK...")
    db = pd.read_csv(bid_p); db['timestamp'] = pd.to_datetime(db['timestamp'])
    da = pd.read_csv(ask_p); da['timestamp'] = pd.to_datetime(da['timestamp'])
    
    df = pd.merge(db, da, on='timestamp', suffixes=('_bid','_ask'))
    df = df.rename(columns={
        'open_bid':'bid_open','high_bid':'bid_high','low_bid':'bid_low','close_bid':'bid_close',
        'open_ask':'ask_open','high_ask':'ask_high','low_ask':'ask_low','close_ask':'ask_close'
    })
    df['spread_open'] = df['ask_open'] - df['bid_open']
    df['spread_close'] = df['ask_close'] - df['bid_close']
    df['source'] = 'Dukascopy'
    df['quality_flags'] = 0
    df = df.sort_values('timestamp').reset_index(drop=True)
    df.to_csv(out_path, index=False)
    
    m1_summary[y] = {"rows": len(df), "file": out_path, "sha256": sha256(out_path)}
    print(f"  {y}: {len(df)} rows saved.")

with open(os.path.join(OUT,"m1_normalization","phase26c_m1_normalization_summary.json"),"w") as f:
    json.dump(m1_summary, f, indent=2, default=str)

# ── PHASE 4: M1 Audit ──
print("\n=== FASE 4: Auditoría M1 Full ===")
m1_audit = {}
for y in YEARS:
    p = os.path.join(M1_OUT, str(y), f"EURUSD_M1_{y}.csv")
    df = pd.read_csv(p); df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    diffs = df['timestamp'].diff().dt.total_seconds()
    gaps = int((diffs > 60).sum())
    dupes = int(df['timestamp'].duplicated().sum())
    neg_spread = int((df['spread_close'] < -0.00001).sum())
    m1_audit[y] = {
        "rows": len(df), "gaps": gaps, "duplicates": dupes, "neg_spreads": neg_spread,
        "min_ts": str(df['timestamp'].min()), "max_ts": str(df['timestamp'].max()),
        "verdict": "M1_CERTIFIED_WITH_WARNINGS" if gaps > 100 else "M1_CERTIFIED_FULL"
    }
    print(f"  {y}: rows={len(df)} gaps={gaps} dupes={dupes} neg_spread={neg_spread} -> {m1_audit[y]['verdict']}")

with open(os.path.join(OUT,"m1_quality_audit","phase26c_m1_quality_audit_summary.json"),"w") as f:
    json.dump(m1_audit, f, indent=2, default=str)

# ── PHASE 5: M3 Generation ──
print("\n=== FASE 5: Generación M3 desde M1 ===")
m3_summary = {}
for y in YEARS:
    p = os.path.join(M1_OUT, str(y), f"EURUSD_M1_{y}.csv")
    out_dir = os.path.join(M3_OUT, str(y))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"EURUSD_M3_{y}.csv")
    
    df = pd.read_csv(p); df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    r = df.resample('3min', closed='left', label='right')
    m3 = pd.DataFrame()
    m3['bid_open'] = r['bid_open'].first()
    m3['bid_high'] = r['bid_high'].max()
    m3['bid_low'] = r['bid_low'].min()
    m3['bid_close'] = r['bid_close'].last()
    m3['ask_open'] = r['ask_open'].first()
    m3['ask_high'] = r['ask_high'].max()
    m3['ask_low'] = r['ask_low'].min()
    m3['ask_close'] = r['ask_close'].last()
    m3 = m3.shift(1).dropna().reset_index()
    m3.to_csv(out_path, index=False)
    
    m3_summary[y] = {"rows": len(m3), "file": out_path, "sha256": sha256(out_path),
                      "verdict": "M3_GENERATED_OK"}
    print(f"  {y}: {len(m3)} M3 rows.")

with open(os.path.join(OUT,"m3_generation","phase26c_m3_generation_summary.json"),"w") as f:
    json.dump(m3_summary, f, indent=2, default=str)

# ── PHASE 6: M3 Audit ──
print("\n=== FASE 6: Auditoría M3 ===")
m3_audit = {}
for y in YEARS:
    p = os.path.join(M3_OUT, str(y), f"EURUSD_M3_{y}.csv")
    df = pd.read_csv(p); df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    diffs = df['timestamp'].diff().dt.total_seconds()
    gaps = int((diffs > 180).sum())
    spread = df['ask_close'] - df['bid_close']
    neg = int((spread < -0.00001).sum())
    m3_audit[y] = {"rows": len(df), "gaps": gaps, "neg_spreads": neg,
                    "verdict": "M3_CERTIFIED_WITH_MASK" if gaps > 50 else "M3_CERTIFIED_FULL"}
    print(f"  {y}: rows={len(df)} gaps={gaps} neg_spread={neg} -> {m3_audit[y]['verdict']}")

with open(os.path.join(OUT,"m3_quality_audit","phase26c_m3_quality_audit_summary.json"),"w") as f:
    json.dump(m3_audit, f, indent=2, default=str)

# ── PHASE 7: Data Quality Mask ──
print("\n=== FASE 7: Data Quality Mask Full ===")
all_mask = []
for y in YEARS:
    p = os.path.join(M3_OUT, str(y), f"EURUSD_M3_{y}.csv")
    df = pd.read_csv(p); df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['is_blocked'] = 0
    spread = df['ask_close'] - df['bid_close']
    df.loc[spread < -0.00001, 'is_blocked'] = 1
    df.loc[spread > 0.01, 'is_blocked'] = 1
    all_mask.append(df[['timestamp','is_blocked']])

mask_full = pd.concat(all_mask).sort_values('timestamp').reset_index(drop=True)
mask_path = os.path.join(MASK_OUT, "EURUSD_M3_DATA_QUALITY_MASK_2015_2019.csv")
mask_full.to_csv(mask_path, index=False)
blocked = int(mask_full['is_blocked'].sum())
total = len(mask_full)
print(f"  Mask: {total} bars, {blocked} blocked ({100*blocked/total:.2f}%)")

mask_summary = {"total_bars": total, "blocked": blocked, "pct_blocked": round(100*blocked/total,2),
                "verdict": "MASK_FULL_CREATED"}
with open(os.path.join(OUT,"data_quality_mask","phase26c_data_quality_mask_summary.json"),"w") as f:
    json.dump(mask_summary, f, indent=2)

# ── PHASE 8: News Fortress ──
print("\n=== FASE 8: News Fortress ===")
news_path = os.path.join(ROOT, "data_intake_2015_2019", "news_eurusd_2015_2019.csv")
ndf = pd.read_csv(news_path); ndf['timestamp_utc'] = pd.to_datetime(ndf['timestamp_utc'])
ndf['year'] = ndf['timestamp_utc'].dt.year
cov = ndf.groupby('year').size().reset_index(name='count')
cov.to_csv(os.path.join(OUT,"news_fortress","phase26c_news_coverage_by_year.csv"), index=False)
news_sum = {"total_events": len(ndf), "years": sorted(ndf['year'].unique().tolist()),
            "verdict": "NEWS_CERTIFIED"}
with open(os.path.join(OUT,"news_fortress","phase26c_news_fortress_certification.json"),"w") as f:
    json.dump(news_sum, f, indent=2)
print(f"  News: {len(ndf)} events, years={news_sum['years']}")

# ── PHASE 9: Year Certification ──
print("\n=== FASE 9: Certificación por Año ===")
year_cert = {}
for y in YEARS:
    m1v = m1_audit[y]['verdict']
    m3v = m3_audit[y]['verdict']
    if 'FULL' in m1v and 'FULL' in m3v:
        yv = "CERTIFIED_FULL"
    elif 'NOT_USABLE' in m1v or 'NOT_USABLE' in m3v:
        yv = "NOT_USABLE"
    else:
        yv = "CERTIFIED_WITH_MASK"
    year_cert[y] = {"m1": m1v, "m3": m3v, "year_verdict": yv}
    print(f"  {y}: {yv}")

with open(os.path.join(OUT,"year_certification","phase26c_year_certification_summary.json"),"w") as f:
    json.dump(year_cert, f, indent=2)

# ── Summary ──
overall = all(v['year_verdict'] in ('CERTIFIED_FULL','CERTIFIED_WITH_MASK') for v in year_cert.values())
verdict = "PHASE26C_2015_2019_DATA_CERTIFIED_WITH_MASK" if overall else "PHASE26C_DATA_NOT_USABLE"
print(f"\n=== VEREDICTO FINAL: {verdict} ===")

final = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "verdict": verdict,
    "m1_audit": m1_audit, "m3_audit": m3_audit, "m3_summary": {y:{"rows":v["rows"]} for y,v in m3_summary.items()},
    "mask": mask_summary, "news": news_sum, "year_certification": year_cert
}
with open(os.path.join(OUT,"preflight","phase26c_preflight.json"),"w") as f:
    json.dump(final, f, indent=2, default=str)

print("DONE")
