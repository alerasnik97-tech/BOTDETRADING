import pandas as pd
import numpy as np

def analyze_ecb():
    df = pd.read_csv(r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\results\eurusd_ltf_objective_entry_replacement_ecb_autopilot\stage2\trades.csv')
    
    # Ensure correct types
    df['pnl_r'] = pd.to_numeric(df['pnl_r'])
    df['entry_time_ny'] = pd.to_datetime(df['entry_time_ny'])
    df['hour'] = df['entry_time_ny'].dt.hour
    
    results = []
    
    def get_stats(sub_df, label):
        n = len(sub_df)
        if n == 0:
            return None
        wins = sub_df[sub_df['pnl_r'] > 0]['pnl_r'].sum()
        losses = abs(sub_df[sub_df['pnl_r'] < 0]['pnl_r'].sum())
        pf = wins / losses if losses > 0 else float('inf')
        exp = sub_df['pnl_r'].mean()
        wr = (len(sub_df[sub_df['pnl_r'] > 0]) / n) * 100
        dd = sub_df['pnl_r'].cumsum().min()
        return {
            'Segment': label,
            'N': n,
            'PF': pf,
            'Exp': exp,
            'WR%': wr,
            'DD': dd
        }

    # 1. Source Kind
    for kind in df['source_kind'].unique():
        results.append(get_stats(df[df['source_kind'] == kind], f"Source: {kind}"))

    # 2. Direction
    for direction in df['direction'].unique():
        results.append(get_stats(df[df['direction'] == direction], f"Direction: {direction}"))

    # 3. Source Level Name
    for level in df['source_level_name'].unique():
        results.append(get_stats(df[df['source_level_name'] == level], f"Level: {level}"))

    # 4. Hours (Buckets)
    # Asia (0-6), London/NY overlap (7-12), PM/Post (13-23)
    df['session_bucket'] = pd.cut(df['hour'], bins=[-1, 6, 12, 23], labels=['Asia/EarlyLondon', 'MainSession', 'Late/PM'])
    for bucket in df['session_bucket'].unique():
        results.append(get_stats(df[df['session_bucket'] == bucket], f"Session: {bucket}"))

    # 5. Delay Analysis (Delay between extreme and entry)
    # entry_time_ny is string without offset, assumed to be NY time.
    # extreme_time_ny is string with offset.
    df['entry_dt'] = pd.to_datetime(df['entry_time_ny']).dt.tz_localize('America/New_York', ambiguous='infer')
    df['extreme_dt'] = pd.to_datetime(df['extreme_time_ny'], utc=True).dt.tz_convert('America/New_York')
    
    df['delay_min'] = (df['entry_dt'] - df['extreme_dt']).dt.total_seconds() / 60
    
    # Delay buckets: 0-15m, 15-30m, 30-60m, 60m+
    df['delay_bucket'] = pd.cut(df['delay_min'], bins=[-float('inf'), 15, 30, 60, float('inf')], labels=['Fast (<15m)', 'Medium (15-30m)', 'Slow (30-60m)', 'VerySlow (>60m)'])
    for bucket in ['Fast (<15m)', 'Medium (15-30m)', 'Slow (30-60m)', 'VerySlow (>60m)']:
        stats = get_stats(df[df['delay_bucket'] == bucket], f"Delay: {bucket}")
        if stats:
            results.append(stats)

    # Filter out None and format
    results = [r for r in results if r is not None]
    report_df = pd.DataFrame(results)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(report_df.to_string())

    # Check for candidates (N >= 20, PF >= 1.2, Exp >= 0.1)
    candidates = report_df[(report_df['N'] >= 20) & (report_df['PF'] >= 1.2) & (report_df['Exp'] >= 0.1)]
    print("\nCandidates (N >= 20):")
    if not candidates.empty:
        print(candidates.to_string())
    else:
        print("None found.")
    
    # Check for weak candidates (N 10-19)
    weak_candidates = report_df[(report_df['N'] >= 10) & (report_df['N'] < 20) & (report_df['PF'] >= 1.2) & (report_df['Exp'] >= 0.1)]
    print("\nWeak Candidates (10 <= N < 20):")
    if not weak_candidates.empty:
        print(weak_candidates.to_string())
    else:
        print("None found.")

    # 7. Intersection of Fast Delay with other segments
    print("\nIntersection of Fast Delay (<15m) with Source:")
    fast_df = df[df['delay_bucket'] == 'Fast (<15m)']
    fast_results = []
    for kind in fast_df['source_kind'].unique():
        fast_results.append(get_stats(fast_df[fast_df['source_kind'] == kind], f"Fast + {kind}"))
    print(pd.DataFrame(fast_results).to_string())

if __name__ == "__main__":
    analyze_ecb()
