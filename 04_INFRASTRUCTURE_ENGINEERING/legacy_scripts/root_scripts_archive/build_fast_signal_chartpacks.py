import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import matplotlib, create simple charts if available
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: matplotlib not available, will create visual index without charts")

print("=" * 60)
print("FAST SIGNAL CHART PACKS GENERATOR")
print("=" * 60)

# Load fast signal sample
print("\nLoading fast signal sample...")
df_sample = pd.read_csv('EURUSD_MANUAL_ANNOTATION_FAST_SIGNAL_SAMPLE.csv')
print(f"Fast signal sample size: {len(df_sample)}")

# Load market data for context
print("\nLoading market data...")
mid_path = Path('data_precision/dukascopy/EURUSD_M1_MID.csv')
df_mid = pd.read_csv(mid_path)
df_mid['timestamp'] = pd.to_datetime(df_mid['timestamp'], utc=True)
print(f"Market data rows: {len(df_mid)}")

# Convert sample timestamps to UTC
df_sample['dateStart_dt'] = pd.to_datetime(df_sample['dateStart'], utc=True)
df_sample['dateEnd_dt'] = pd.to_datetime(df_sample['dateEnd'], utc=True)

# Create output directory
output_dir = Path('manual_trade_chartpacks/fast_signal')
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {output_dir}")

# Function to get market data window around trade
def get_market_window(trade_time, df_mid, hours_before=3, hours_after=2):
    """Get market data window around trade time."""
    start_time = trade_time - pd.Timedelta(hours=hours_before)
    end_time = trade_time + pd.Timedelta(hours=hours_after)
    
    mask = (df_mid['timestamp'] >= start_time) & (df_mid['timestamp'] <= end_time)
    window_data = df_mid[mask].copy()
    
    return window_data

# Function to calculate session levels
def calculate_session_levels(trade_time, df_mid):
    """Calculate Asia and London session levels."""
    trade_date = trade_time.date()
    
    # Asia: 19:00-03:00 NY (previous day)
    asia_start = pd.Timestamp(trade_date, tz='UTC') - pd.Timedelta(hours=5)
    asia_end = pd.Timestamp(trade_date, tz='UTC') + pd.Timedelta(hours=3)
    asia_mask = (df_mid['timestamp'] >= asia_start) & (df_mid['timestamp'] <= asia_end)
    asia_data = df_mid[asia_mask]
    if len(asia_data) > 0:
        asia_high = asia_data['high'].max()
        asia_low = asia_data['low'].min()
    else:
        asia_high = np.nan
        asia_low = np.nan
    
    # London: 03:00-07:00 NY
    london_start = pd.Timestamp(trade_date, tz='UTC') + pd.Timedelta(hours=3)
    london_end = pd.Timestamp(trade_date, tz='UTC') + pd.Timedelta(hours=7)
    london_mask = (df_mid['timestamp'] >= london_start) & (df_mid['timestamp'] <= london_end)
    london_data = df_mid[london_mask]
    if len(london_data) > 0:
        london_high = london_data['high'].max()
        london_low = london_data['low'].min()
    else:
        london_high = np.nan
        london_low = np.nan
    
    # Previous day high/low
    prev_day_start = pd.Timestamp(trade_date, tz='UTC') - pd.Timedelta(days=1)
    prev_day_end = pd.Timestamp(trade_date, tz='UTC')
    prev_day_mask = (df_mid['timestamp'] >= prev_day_start) & (df_mid['timestamp'] < prev_day_end)
    prev_day_data = df_mid[prev_day_mask]
    if len(prev_day_data) > 0:
        prev_day_high = prev_day_data['high'].max()
        prev_day_low = prev_day_data['low'].min()
    else:
        prev_day_high = np.nan
        prev_day_low = np.nan
    
    # Daily open
    day_start = pd.Timestamp(trade_date, tz='UTC')
    day_end = day_start + pd.Timedelta(hours=24)
    day_mask = (df_mid['timestamp'] >= day_start) & (df_mid['timestamp'] < day_end)
    day_data = df_mid[day_mask]
    if len(day_data) > 0:
        daily_open = day_data.iloc[0]['open']
    else:
        daily_open = np.nan
    
    return {
        'asia_high': asia_high,
        'asia_low': asia_low,
        'london_high': london_high,
        'london_low': london_low,
        'prev_day_high': prev_day_high,
        'prev_day_low': prev_day_low,
        'daily_open': daily_open
    }

# Generate chart packs
chartpack_index = []
for idx, row in df_sample.iterrows():
    trade_id = row['id']
    trade_time = row['dateStart_dt']
    entry_price = row['entryPrice']
    side = row['side']
    outcome = row['outcome']
    time_block = row['time_block']
    
    # Get market window
    window_data = get_market_window(trade_time, df_mid, hours_before=3, hours_after=2)
    
    # Calculate session levels
    levels = calculate_session_levels(trade_time, df_mid)
    
    # Create chart filename
    chart_filename = f"FS_{idx+1:03d}_{trade_id}.png"
    chart_path = output_dir / chart_filename
    
    if MATPLOTLIB_AVAILABLE:
        # Create matplotlib chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot price
        ax.plot(window_data['timestamp'], window_data['close'], label='Close', color='black', linewidth=1)
        
        # Plot levels
        if not np.isnan(levels['asia_high']):
            ax.axhline(y=levels['asia_high'], color='blue', linestyle='--', alpha=0.5, label='Asia High')
        if not np.isnan(levels['asia_low']):
            ax.axhline(y=levels['asia_low'], color='blue', linestyle='--', alpha=0.5, label='Asia Low')
        if not np.isnan(levels['london_high']):
            ax.axhline(y=levels['london_high'], color='green', linestyle='--', alpha=0.5, label='London High')
        if not np.isnan(levels['london_low']):
            ax.axhline(y=levels['london_low'], color='green', linestyle='--', alpha=0.5, label='London Low')
        if not np.isnan(levels['prev_day_high']):
            ax.axhline(y=levels['prev_day_high'], color='red', linestyle='--', alpha=0.5, label='Prev Day High')
        if not np.isnan(levels['prev_day_low']):
            ax.axhline(y=levels['prev_day_low'], color='red', linestyle='--', alpha=0.5, label='Prev Day Low')
        if not np.isnan(levels['daily_open']):
            ax.axhline(y=levels['daily_open'], color='purple', linestyle='-', alpha=0.5, label='Daily Open')
        
        # Mark entry
        ax.axvline(x=trade_time, color='orange', linestyle='-', linewidth=2, label='Entry')
        ax.scatter([trade_time], [entry_price], color='orange', s=100, zorder=5)
        
        # Mark SL and TP if available
        if not np.isnan(row['initalSL']):
            ax.axhline(y=row['initalSL'], color='red', linestyle='-', linewidth=1, label='SL')
        if not np.isnan(row['maxTP']):
            ax.axhline(y=row['maxTP'], color='green', linestyle='-', linewidth=1, label='TP')
        
        # Title with trade info
        title = f"Trade {trade_id} | {time_block} | {side.upper()} | {outcome} | Entry: {entry_price:.5f}"
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Time (NY)', fontsize=9)
        ax.set_ylabel('Price', fontsize=9)
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(chart_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"Generated chart: {chart_filename}")
    else:
        # Create placeholder file with info
        with open(chart_path.with_suffix('.txt'), 'w') as f:
            f.write(f"Trade ID: {trade_id}\n")
            f.write(f"Timestamp: {row['dateStart_ny']}\n")
            f.write(f"Time Block: {time_block}\n")
            f.write(f"Side: {side}\n")
            f.write(f"Outcome: {outcome}\n")
            f.write(f"Entry Price: {entry_price}\n")
            f.write(f"SL: {row['initalSL']}\n")
            f.write(f"TP: {row['maxTP']}\n")
            f.write(f"\nSession Levels:\n")
            f.write(f"Asia High: {levels['asia_high']}\n")
            f.write(f"Asia Low: {levels['asia_low']}\n")
            f.write(f"London High: {levels['london_high']}\n")
            f.write(f"London Low: {levels['london_low']}\n")
            f.write(f"Prev Day High: {levels['prev_day_high']}\n")
            f.write(f"Prev Day Low: {levels['prev_day_low']}\n")
            f.write(f"Daily Open: {levels['daily_open']}\n")
        
        print(f"Generated info file: {chart_filename}.txt (matplotlib not available)")
    
    # Add to index
    chartpack_index.append({
        'rank': idx + 1,
        'trade_id': trade_id,
        'timestamp_ny': row['dateStart_ny'],
        'outcome': outcome,
        'side': side,
        'time_block': time_block,
        'entry_price': entry_price,
        'chart_filename': chart_filename if MATPLOTLIB_AVAILABLE else chart_filename.with_suffix('.txt').name,
        'chart_path': str(chart_path) if MATPLOTLIB_AVAILABLE else str(chart_path.with_suffix('.txt'))
    })

# Save chartpack index
df_chartpack_index = pd.DataFrame(chartpack_index)
df_chartpack_index.to_csv(output_dir / 'chartpack_index.csv', index=False)
print(f"\nSaved chartpack index to: {output_dir / 'chartpack_index.csv'}")

# Update main chartpack index
df_chartpack_index_full = df_chartpack_index.copy()
df_chartpack_index_full.to_csv('EURUSD_MANUAL_CHARTPACK_INDEX.csv', index=False)
print(f"Updated main chartpack index: EURUSD_MANUAL_CHARTPACK_INDEX.csv")

print("\n" + "=" * 60)
print("CHART PACK GENERATION COMPLETE")
print("=" * 60)
print(f"Total chart packs generated: {len(df_chartpack_index)}")
print(f"Output directory: {output_dir}")
if not MATPLOTLIB_AVAILABLE:
    print("NOTE: matplotlib not available, generated info files instead of charts")
