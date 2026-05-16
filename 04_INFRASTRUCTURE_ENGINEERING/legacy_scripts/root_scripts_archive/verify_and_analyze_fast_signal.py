import pandas as pd
import numpy as np

print("=" * 60)
print("FAST SIGNAL LEDGER VERIFICATION AND ANALYSIS")
print("=" * 60)

# Load fast signal ledger
print("\nLoading fast signal ledger...")
df = pd.read_csv('EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_LEDGER.csv')
print(f"Total rows: {len(df)}")

# Verify human fields completion
human_fields = ['liquidity_source', 'trigger_type', 'confirmation_type', 
                'operational_context', 'entry_motive', 'quality_rating', 'comment']
print(f"\nHuman fields completion:")
for field in human_fields:
    filled = df[field].notna().sum()
    print(f"  {field}: {filled}/{len(df)}")

# Check if all human fields are filled
all_filled = all(df[field].notna().all() for field in human_fields[:-1])  # Exclude comment (optional)
print(f"\nAll required human fields filled: {all_filled}")

# Show sample of human fields
print(f"\nSample of human fields:")
print(df[human_fields].head(10).to_string())

# Baseline metrics
print(f"\nBaseline metrics:")
baseline_win_rate = (df['outcome'] == 'TP').sum() / len(df)
print(f"Baseline win rate: {baseline_win_rate:.1%}")
print(f"Outcome distribution:\n{df['outcome'].value_counts()}")

# Analysis by liquidity_source
print(f"\n" + "=" * 60)
print("ANALYSIS BY LIQUIDITY_SOURCE")
print("=" * 60)
if 'liquidity_source' in df.columns and df['liquidity_source'].notna().any():
    contingency = pd.crosstab(df['liquidity_source'], df['outcome'])
    print(contingency)
    
    for source in contingency.index:
        tp = contingency.loc[source, 'TP'] if 'TP' in contingency.columns else 0
        sl = contingency.loc[source, 'SL'] if 'SL' in contingency.columns else 0
        be = contingency.loc[source, 'BE'] if 'BE' in contingency.columns else 0
        total = tp + sl + be
        if total > 0:
            win_rate = tp / (tp + sl) if (tp + sl) > 0 else 0
            lift = win_rate - baseline_win_rate
            print(f"{source}: {win_rate:.1%} (lift: {lift:+.1%}, n={total})")

# Analysis by trigger_type
print(f"\n" + "=" * 60)
print("ANALYSIS BY TRIGGER_TYPE")
print("=" * 60)
if 'trigger_type' in df.columns and df['trigger_type'].notna().any():
    contingency = pd.crosstab(df['trigger_type'], df['outcome'])
    print(contingency)
    
    for trigger in contingency.index:
        tp = contingency.loc[trigger, 'TP'] if 'TP' in contingency.columns else 0
        sl = contingency.loc[trigger, 'SL'] if 'SL' in contingency.columns else 0
        be = contingency.loc[trigger, 'BE'] if 'BE' in contingency.columns else 0
        total = tp + sl + be
        if total > 0:
            win_rate = tp / (tp + sl) if (tp + sl) > 0 else 0
            lift = win_rate - baseline_win_rate
            print(f"{trigger}: {win_rate:.1%} (lift: {lift:+.1%}, n={total})")

# Analysis by confirmation_type
print(f"\n" + "=" * 60)
print("ANALYSIS BY CONFIRMATION_TYPE")
print("=" * 60)
if 'confirmation_type' in df.columns and df['confirmation_type'].notna().any():
    contingency = pd.crosstab(df['confirmation_type'], df['outcome'])
    print(contingency)
    
    for confirm in contingency.index:
        tp = contingency.loc[confirm, 'TP'] if 'TP' in contingency.columns else 0
        sl = contingency.loc[confirm, 'SL'] if 'SL' in contingency.columns else 0
        be = contingency.loc[confirm, 'BE'] if 'BE' in contingency.columns else 0
        total = tp + sl + be
        if total > 0:
            win_rate = tp / (tp + sl) if (tp + sl) > 0 else 0
            lift = win_rate - baseline_win_rate
            print(f"{confirm}: {win_rate:.1%} (lift: {lift:+.1%}, n={total})")

# Analysis by operational_context
print(f"\n" + "=" * 60)
print("ANALYSIS BY OPERATIONAL_CONTEXT")
print("=" * 60)
if 'operational_context' in df.columns and df['operational_context'].notna().any():
    contingency = pd.crosstab(df['operational_context'], df['outcome'])
    print(contingency)
    
    for context in contingency.index:
        tp = contingency.loc[context, 'TP'] if 'TP' in contingency.columns else 0
        sl = contingency.loc[context, 'SL'] if 'SL' in contingency.columns else 0
        be = contingency.loc[context, 'BE'] if 'BE' in contingency.columns else 0
        total = tp + sl + be
        if total > 0:
            win_rate = tp / (tp + sl) if (tp + sl) > 0 else 0
            lift = win_rate - baseline_win_rate
            print(f"{context}: {win_rate:.1%} (lift: {lift:+.1%}, n={total})")

# Analysis by entry_motive
print(f"\n" + "=" * 60)
print("ANALYSIS BY ENTRY_MOTIVE")
print("=" * 60)
if 'entry_motive' in df.columns and df['entry_motive'].notna().any():
    contingency = pd.crosstab(df['entry_motive'], df['outcome'])
    print(contingency)
    
    for motive in contingency.index:
        tp = contingency.loc[motive, 'TP'] if 'TP' in contingency.columns else 0
        sl = contingency.loc[motive, 'SL'] if 'SL' in contingency.columns else 0
        be = contingency.loc[motive, 'BE'] if 'BE' in contingency.columns else 0
        total = tp + sl + be
        if total > 0:
            win_rate = tp / (tp + sl) if (tp + sl) > 0 else 0
            lift = win_rate - baseline_win_rate
            print(f"{motive}: {win_rate:.1%} (lift: {lift:+.1%}, n={total})")

# Analysis by quality_rating
print(f"\n" + "=" * 60)
print("ANALYSIS BY QUALITY_RATING")
print("=" * 60)
if 'quality_rating' in df.columns and df['quality_rating'].notna().any():
    for rating in ['A', 'B', 'C']:
        df_rating = df[df['quality_rating'] == rating]
        if len(df_rating) > 0:
            win_rate = (df_rating['outcome'] == 'TP').sum() / len(df_rating)
            print(f"Quality {rating}: {win_rate:.1%} (n={len(df_rating)})")

# Analysis by side
print(f"\n" + "=" * 60)
print("ANALYSIS BY SIDE")
print("=" * 60)
longs = df[df['side'] == 'buy']
shorts = df[df['side'] == 'sell']
long_win_rate = (longs['outcome'] == 'TP').sum() / len(longs) if len(longs) > 0 else 0
short_win_rate = (shorts['outcome'] == 'TP').sum() / len(shorts) if len(shorts) > 0 else 0
print(f"Longs win rate: {long_win_rate:.1%} (n={len(longs)})")
print(f"Shorts win rate: {short_win_rate:.1%} (n={len(shorts)})")

# Analysis by time_block
print(f"\n" + "=" * 60)
print("ANALYSIS BY TIME_BLOCK")
print("=" * 60)
for block in ['3am-4am', '4am-5am', '5am-6am']:
    df_block = df[df['time_block'] == block]
    if len(df_block) > 0:
        win_rate = (df_block['outcome'] == 'TP').sum() / len(df_block)
        print(f"{block}: {win_rate:.1%} (n={len(df_block)})")

# Save results
results = {
    'baseline_win_rate': baseline_win_rate,
    'long_win_rate': long_win_rate,
    'short_win_rate': short_win_rate,
    'sample_size': len(df),
    'human_fields_completed': all_filled
}

results_df = pd.DataFrame([results])
results_df.to_csv('EURUSD_MANUAL_FAST_SIGNAL_ANALYSIS_RESULTS.csv', index=False)
print(f"\nResults saved to: EURUSD_MANUAL_FAST_SIGNAL_ANALYSIS_RESULTS.csv")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
