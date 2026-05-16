import pandas as pd
import numpy as np
import sys
from pathlib import Path

print("=" * 60)
print("EURUSD MANUAL ANNOTATION ANALYSIS")
print("=" * 60)

# First, run validation to ensure ledger is ready
print("\nRunning validation first...")
print("If validation fails, analysis will NOT run.")
print("Run: python scripts/validate_manual_annotations.py")
print("-" * 60)

# Load working ledger (not the original ledger)
print("\nLoading working ledger...")
ledger_path = 'EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_WORKING.csv'
if not Path(ledger_path).exists():
    print("ERROR: Working ledger not found.")
    print("Please run: python scripts/prepare_fast_signal_working_ledger.py")
    sys.exit(1)

df = pd.read_csv(ledger_path)
print(f"Total rows: {len(df)}")

# Check if validation status is READY
if 'annotation_status' in df.columns:
    ready_count = (df['annotation_status'] == 'READY').sum()
    if ready_count < len(df):
        print(f"\nERROR: Ledger not ready for analysis.")
        print(f"Ready rows: {ready_count}/{len(df)}")
        print(f"Please run: python scripts/validate_manual_annotations.py")
        print(f"Fix any errors and complete all annotations first.")
        sys.exit(1)
else:
    print("\nERROR: annotation_status column not found in ledger.")
    print("Please run: python scripts/prepare_fast_signal_working_ledger.py")
    sys.exit(1)

print(f"\nValidation passed. All {len(df)} rows ready for analysis.")

# Check if annotations are completed
human_fields = ['liquidity_source', 'trigger_type', 'confirmation_type', 
                'operational_context', 'entry_motive', 'quality_rating', 'comment']
human_filled = df[human_fields].notna().any(axis=1).sum()
print(f"Rows with human annotations: {human_filled}/{len(df)}")

if human_filled == 0:
    print("ERROR: No human annotations found. Please complete annotations first.")
    exit(1)

# Use only annotated rows for analysis
df_analyzed = df[df[human_fields].notna().any(axis=1)].copy()
print(f"Analyzing {len(df_analyzed)} annotated rows")

# Baseline metrics
baseline_win_rate = (df_analyzed['outcome'] == 'TP').sum() / len(df_analyzed)
print(f"\nBaseline win rate: {baseline_win_rate:.1%}")

# Analysis 1: Outcome by liquidity_source
print("\n" + "=" * 60)
print("ANALYSIS 1: OUTCOME BY LIQUIDITY_SOURCE")
print("=" * 60)
if 'liquidity_source' in df_analyzed.columns and df_analyzed['liquidity_source'].notna().any():
    contingency = pd.crosstab(df_analyzed['liquidity_source'], df_analyzed['outcome'])
    print(contingency)
    
    # Win rate per category
    for source in contingency.index:
        tp = contingency.loc[source, 'TP'] if 'TP' in contingency.columns else 0
        sl = contingency.loc[source, 'SL'] if 'SL' in contingency.columns else 0
        be = contingency.loc[source, 'BE'] if 'BE' in contingency.columns else 0
        total = tp + sl + be
        if total > 0:
            win_rate = tp / (tp + sl) if (tp + sl) > 0 else 0
            lift = win_rate - baseline_win_rate if baseline_win_rate > 0 else 0
            print(f"{source}: {win_rate:.1%} (lift: {lift:+.1%}, n={total})")

# Analysis 2: Outcome by trigger_type
print("\n" + "=" * 60)
print("ANALYSIS 2: OUTCOME BY TRIGGER_TYPE")
print("=" * 60)
if 'trigger_type' in df_analyzed.columns and df_analyzed['trigger_type'].notna().any():
    contingency = pd.crosstab(df_analyzed['trigger_type'], df_analyzed['outcome'])
    print(contingency)
    
    for trigger in contingency.index:
        tp = contingency.loc[trigger, 'TP'] if 'TP' in contingency.columns else 0
        sl = contingency.loc[trigger, 'SL'] if 'SL' in contingency.columns else 0
        be = contingency.loc[trigger, 'BE'] if 'BE' in contingency.columns else 0
        total = tp + sl + be
        if total > 0:
            win_rate = tp / (tp + sl) if (tp + sl) > 0 else 0
            lift = win_rate - baseline_win_rate if baseline_win_rate > 0 else 0
            print(f"{trigger}: {win_rate:.1%} (lift: {lift:+.1%}, n={total})")

# Analysis 3: Outcome by confirmation_type
print("\n" + "=" * 60)
print("ANALYSIS 3: OUTCOME BY CONFIRMATION_TYPE")
print("=" * 60)
if 'confirmation_type' in df_analyzed.columns and df_analyzed['confirmation_type'].notna().any():
    contingency = pd.crosstab(df_analyzed['confirmation_type'], df_analyzed['outcome'])
    print(contingency)
    
    for confirm in contingency.index:
        tp = contingency.loc[confirm, 'TP'] if 'TP' in contingency.columns else 0
        sl = contingency.loc[confirm, 'SL'] if 'SL' in contingency.columns else 0
        be = contingency.loc[confirm, 'BE'] if 'BE' in contingency.columns else 0
        total = tp + sl + be
        if total > 0:
            win_rate = tp / (tp + sl) if (tp + sl) > 0 else 0
            lift = win_rate - baseline_win_rate if baseline_win_rate > 0 else 0
            print(f"{confirm}: {win_rate:.1%} (lift: {lift:+.1%}, n={total})")

# Analysis 4: Outcome by operational_context
print("\n" + "=" * 60)
print("ANALYSIS 4: OUTCOME BY OPERATIONAL_CONTEXT")
print("=" * 60)
if 'operational_context' in df_analyzed.columns and df_analyzed['operational_context'].notna().any():
    contingency = pd.crosstab(df_analyzed['operational_context'], df_analyzed['outcome'])
    print(contingency)
    
    for context in contingency.index:
        tp = contingency.loc[context, 'TP'] if 'TP' in contingency.columns else 0
        sl = contingency.loc[context, 'SL'] if 'SL' in contingency.columns else 0
        be = contingency.loc[context, 'BE'] if 'BE' in contingency.columns else 0
        total = tp + sl + be
        if total > 0:
            win_rate = tp / (tp + sl) if (tp + sl) > 0 else 0
            lift = win_rate - baseline_win_rate if baseline_win_rate > 0 else 0
            print(f"{context}: {win_rate:.1%} (lift: {lift:+.1%}, n={total})")

# Analysis 5: Longs vs Shorts by annotations
print("\n" + "=" * 60)
print("ANALYSIS 5: LONGS VS SHORTS")
print("=" * 60)
longs = df_analyzed[df_analyzed['side'] == 'buy']
shorts = df_analyzed[df_analyzed['side'] == 'sell']
long_win_rate = (longs['outcome'] == 'TP').sum() / len(longs) if len(longs) > 0 else 0
short_win_rate = (shorts['outcome'] == 'TP').sum() / len(shorts) if len(shorts) > 0 else 0
print(f"Longs win rate: {long_win_rate:.1%} (n={len(longs)})")
print(f"Shorts win rate: {short_win_rate:.1%} (n={len(shorts)})")

# Analysis 6: 5am-6am vs rest
print("\n" + "=" * 60)
print("ANALYSIS 6: 5am-6am VS REST")
print("=" * 60)
df_5am = df_analyzed[df_analyzed['time_block'] == '5am-6am']
df_rest = df_analyzed[df_analyzed['time_block'] != '5am-6am']
win_rate_5am = (df_5am['outcome'] == 'TP').sum() / len(df_5am) if len(df_5am) > 0 else 0
win_rate_rest = (df_rest['outcome'] == 'TP').sum() / len(df_rest) if len(df_rest) > 0 else 0
print(f"5am-6am win rate: {win_rate_5am:.1%} (n={len(df_5am)})")
print(f"Rest win rate: {win_rate_rest:.1%} (n={len(df_rest)})")

# Analysis 7: Quality rating
print("\n" + "=" * 60)
print("ANALYSIS 7: QUALITY RATING")
print("=" * 60)
if 'quality_rating' in df_analyzed.columns and df_analyzed['quality_rating'].notna().any():
    for rating in ['A', 'B', 'C']:
        df_rating = df_analyzed[df_analyzed['quality_rating'] == rating]
        if len(df_rating) > 0:
            win_rate = (df_rating['outcome'] == 'TP').sum() / len(df_rating)
            print(f"Quality {rating}: {win_rate:.1%} (n={len(df_rating)})")

# Verdict determination
print("\n" + "=" * 60)
print("VERDICT DETERMINATION")
print("=" * 60)

# Check for programmable core
programmable_combinations = []
strong_discriminators = []

# Check individual discriminators
for field in ['liquidity_source', 'trigger_type', 'confirmation_type', 'operational_context']:
    if field in df_analyzed.columns and df_analyzed[field].notna().any():
        contingency = pd.crosstab(df_analyzed[field], df_analyzed['outcome'])
        for category in contingency.index:
            tp = contingency.loc[category, 'TP'] if 'TP' in contingency.columns else 0
            sl = contingency.loc[category, 'SL'] if 'SL' in contingency.columns else 0
            be = contingency.loc[category, 'BE'] if 'BE' in contingency.columns else 0
            total = tp + sl + be
            if total >= 5:  # Minimum sample
                win_rate = tp / (tp + sl) if (tp + sl) > 0 else 0
                if win_rate > 0.65:
                    strong_discriminators.append(f"{field}={category}: {win_rate:.1%} (n={total})")

# Check for programmable core (combination of 2-3 fields)
# Simplified: check if we have strong discriminators
if strong_discriminators:
    print("Strong discriminators found:")
    for d in strong_discriminators:
        print(f"  - {d}")
    verdict = "MANUAL_EDGE_PARTIALLY_TRANSLATABLE"
else:
    print("No strong discriminators found.")
    verdict = "MANUAL_EDGE_NOT_YET_TRANSLATABLE"

# Check for programmable core with sufficient sample
if len(df_analyzed) >= 10 and any(win_rate > 0.65 for win_rate in [long_win_rate, short_win_rate, win_rate_5am, win_rate_rest]):
    if long_win_rate > 0.65 or win_rate_5am > 0.65:
        verdict = "MANUAL_EDGE_PARTIALLY_TRANSLATABLE"

print(f"\nFINAL VERDICT: {verdict}")

# Save results
results = {
    'baseline_win_rate': baseline_win_rate,
    'long_win_rate': long_win_rate,
    'short_win_rate': short_win_rate,
    'win_rate_5am': win_rate_5am,
    'win_rate_rest': win_rate_rest,
    'strong_discriminators': strong_discriminators,
    'verdict': verdict,
    'sample_size': len(df_analyzed)
}

results_df = pd.DataFrame([results])
results_df.to_csv('EURUSD_MANUAL_ANNOTATION_ANALYSIS_RESULTS.csv', index=False)
print(f"\nResults saved to: EURUSD_MANUAL_ANNOTATION_ANALYSIS_RESULTS.csv")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
