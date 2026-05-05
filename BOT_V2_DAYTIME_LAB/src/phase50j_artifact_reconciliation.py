import pandas as pd
import json
import os
from datetime import datetime

# Paths
BASE_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
REPORTS_DIR = os.path.join(BASE_DIR, "BOT_V2_DAYTIME_LAB", "reports", "manipulante_tick_historical")
SRC_CSV = os.path.join(REPORTS_DIR, "PHASE50H_MULTI_MONTH_TRADE_LEVEL_RESULTS.csv")
CANONICAL_MONTH_SET_FILE = os.path.join(REPORTS_DIR, "PHASE50J_CANONICAL_MONTH_SET.json")

# Output Paths
OUTPUT_CSV = os.path.join(REPORTS_DIR, "PHASE50J_CANONICAL_TRADE_LEVEL_RESULTS.csv")
OUTPUT_METRICS_CSV = os.path.join(REPORTS_DIR, "PHASE50J_CANONICAL_MONTHLY_METRICS.csv")
OUTPUT_AGGREGATE_JSON = os.path.join(REPORTS_DIR, "PHASE50J_CANONICAL_AGGREGATE_METRICS.json")

def reconcile():
    print("Starting PHASE50J Forensic Reconciliation...")
    
    # Load canonical months
    with open(CANONICAL_MONTH_SET_FILE, 'r') as f:
        month_set = json.load(f)
    official_months = month_set['months_official']
    
    # Load source data
    df = pd.read_csv(SRC_CSV)
    
    # 1. Filter official months
    df_clean = df[df['month'].isin(official_months)].copy()
    
    # 2. Exclude non-official (e.g., 2025-08)
    excluded_trades = df[~df['month'].isin(official_months)]
    print(f"Excluded {len(excluded_trades)} trades from non-official months (e.g. 2025-08)")
    
    # 3. Deduplicate
    initial_len = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['trade_id', 'entry_time', 'direction'])
    if len(df_clean) < initial_len:
        print(f"Removed {initial_len - len(df_clean)} duplicates.")
    
    # 4. Handle non-auditables
    # If tick_R is NaN or string representing no data
    df_clean['auditable_yes_no'] = df_clean['tick_outcome'].apply(lambda x: 'NO' if x == 'NO_TICK_DATA' else 'YES')
    
    # Save canonical trade-level
    df_clean.to_csv(OUTPUT_CSV, index=False)
    print(f"Canonical trade-level saved: {len(df_clean)} trades.")
    
    # 5. Recalculate Metrics
    monthly_stats = []
    
    for month in official_months:
        m_df = df_clean[df_clean['month'] == month]
        if m_df.empty:
            print(f"WARNING: No data found for official month {month}")
            continue
            
        sample = len(m_df)
        auditable_df = m_df[m_df['auditable_yes_no'] == 'YES']
        auditable_sample = len(auditable_df)
        no_auditables = sample - auditable_sample
        
        # Calculate PF and Expectancy
        # Assuming tick_R is numeric
        tick_r_values = pd.to_numeric(auditable_df['tick_R'], errors='coerce').fillna(0)
        pos_r = tick_r_values[tick_r_values > 0].sum()
        neg_r = abs(tick_r_values[tick_r_values < 0].sum())
        pf_tick = pos_r / neg_r if neg_r > 0 else float('inf')
        
        expectancy_tick = tick_r_values.mean() if not tick_r_values.empty else 0
        
        # Winrate tick
        winners_tick = len(auditable_df[auditable_df['tick_R'] > 0])
        winrate_tick = (winners_tick / auditable_sample * 100) if auditable_sample > 0 else 0
        
        # Total R
        total_r_tick = tick_r_values.sum()
        
        # Bar Metrics (for comparison)
        bar_r_values = pd.to_numeric(m_df['bar_R'], errors='coerce').fillna(0)
        pos_r_bar = bar_r_values[bar_r_values > 0].sum()
        neg_r_bar = abs(bar_r_values[bar_r_values < 0].sum())
        pf_bar = pos_r_bar / neg_r_bar if neg_r_bar > 0 else float('inf')
        expectancy_bar = bar_r_values.mean()
        winrate_bar = (len(m_df[m_df['bar_R'] > 0]) / sample * 100) if sample > 0 else 0
        total_r_bar = bar_r_values.sum()
        
        # Match rate
        matches = len(m_df[m_df['match_status'] == 'MATCH'])
        match_rate = (matches / sample * 100) if sample > 0 else 0
        
        monthly_stats.append({
            'month': month,
            'sample': sample,
            'auditable_sample': auditable_sample,
            'no_auditables': no_auditables,
            'PF_bar': pf_bar,
            'PF_tick': pf_tick,
            'expectancy_bar': expectancy_bar,
            'expectancy_tick': expectancy_tick,
            'winrate_bar': winrate_bar,
            'winrate_tick': winrate_tick,
            'total_R_bar': total_r_bar,
            'total_R_tick': total_r_tick,
            'match_rate': match_rate
        })
        
    stats_df = pd.DataFrame(monthly_stats)
    stats_df.to_csv(OUTPUT_METRICS_CSV, index=False)
    
    # 6. Aggregate Metrics
    all_tick_r = pd.to_numeric(df_clean[df_clean['auditable_yes_no'] == 'YES']['tick_R'], errors='coerce').fillna(0)
    pos_r_total = all_tick_r[all_tick_r > 0].sum()
    neg_r_total = abs(all_tick_r[all_tick_r < 0].sum())
    aggregate_pf_tick = pos_r_total / neg_r_total if neg_r_total > 0 else float('inf')
    aggregate_expectancy_tick = all_tick_r.mean()
    
    # Sequential DD
    # Sort by time
    df_clean['datetime'] = pd.to_datetime(df_clean['entry_time'], utc=True)
    df_sorted = df_clean.sort_values('datetime')
    
    cumulative_r = 0
    max_r = 0
    max_dd = 0
    for r in df_sorted['tick_R'].fillna(0):
        cumulative_r += r
        if cumulative_r > max_r:
            max_r = cumulative_r
        dd = cumulative_r - max_r
        if dd < max_dd:
            max_dd = dd
            
    aggregate_metrics = {
        'total_trades': len(df_clean),
        'auditable_trades': int(df_clean['auditable_yes_no'].value_counts().get('YES', 0)),
        'PF_tick': float(aggregate_pf_tick),
        'expectancy_tick': float(aggregate_expectancy_tick),
        'DD_tick': float(max_dd),
        'total_R_tick': float(all_tick_r.sum()),
        'winrate_tick': float(len(all_tick_r[all_tick_r > 0]) / len(all_tick_r) * 100) if len(all_tick_r) > 0 else 0,
        'missing_months': [m for m in official_months if m not in stats_df['month'].values],
        'generated_at': datetime.now().isoformat()
    }
    
    with open(OUTPUT_AGGREGATE_JSON, 'w') as f:
        json.dump(aggregate_metrics, f, indent=2)
        
    print("Reconciliation complete.")

if __name__ == "__main__":
    reconcile()
