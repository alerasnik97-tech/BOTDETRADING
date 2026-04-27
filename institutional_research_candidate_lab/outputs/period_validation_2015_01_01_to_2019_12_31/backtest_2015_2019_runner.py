
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime, timezone
import sys

# Add project root to sys.path to import baseline_truth_model
project_root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
sys.path.append(str(project_root))

from institutional_research_candidate_lab.baseline_truth_model import run_baseline_truth_model
from institutional_research_candidate_lab.config import CandidateConfig

def run_backtest_2015_2019():
    output_dir = project_root / "institutional_research_candidate_lab" / "outputs" / "period_validation_2015_01_01_to_2019_12_31"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    staging_dir = project_root / "data_intake_2015_2019"
    prepared_dir = staging_dir / "prepared"
    
    print("Loading H1 data...")
    h1 = pd.read_csv(prepared_dir / "EURUSD_H1_BID.csv")
    h1['timestamp'] = pd.to_datetime(h1['timestamp'], utc=True).dt.tz_convert('US/Eastern')
    h1 = h1.set_index('timestamp').sort_index()
    
    print("Loading M5 data...")
    m5 = pd.read_csv(prepared_dir / "EURUSD_M5_BID.csv")
    m5['timestamp'] = pd.to_datetime(m5['timestamp'], utc=True).dt.tz_convert('US/Eastern')
    m5 = m5.set_index('timestamp').sort_index()
    
    print("Loading News data...")
    news = pd.read_csv(staging_dir / "news" / "news_eurusd_2015_2019_fortress_candidate.csv")
    # News index expects 'timestamp_ny' and 'event_name_normalized'
    news['timestamp_ny'] = pd.to_datetime(news['timestamp'], utc=True).dt.tz_convert('US/Eastern')
    news['event_name_normalized'] = news['family']
    news.attrs['coverage_start_date'] = "2015-01-01"
    news.attrs['coverage_end_date'] = "2019-12-31"
    
    config = CandidateConfig(
        start_date="2015-01-01",
        end_date="2019-12-31",
        truth_model=True
    )
    
    print(f"Running backtest from {config.start_date} to {config.end_date}...")
    results = run_baseline_truth_model(config, h1=h1, m5=m5, news=news)
    
    trades = results['trades']
    if trades.empty:
        print("No trades found.")
        return
    
    # Standardize time for metrics
    trades['entry_time'] = pd.to_datetime(trades['entry_time'], utc=True).dt.tz_convert('US/Eastern')
    trades['year'] = trades['entry_time'].dt.year
    trades['month'] = trades['entry_time'].dt.month
    trades['weekday'] = trades['entry_time'].dt.day_name()
    trades['hour'] = trades['entry_time'].dt.hour
    
    # Session breakdown
    def get_session(hour):
        if 0 <= hour < 4: return "London_Overnight"
        if 4 <= hour < 8: return "London_Morning"
        if 8 <= hour < 12: return "NY_Morning"
        if 12 <= hour < 16: return "NY_Afternoon"
        return "Late"
    trades['session'] = trades['hour'].apply(get_session)
    
    # Metrics
    summary = {}
    summary['sample_size'] = len(trades)
    summary['win_rate'] = len(trades[trades['pnl_r'] > 0]) / len(trades)
    summary['TP_count'] = len(trades[trades['exit_reason'] == 'tp_hit'])
    summary['SL_count'] = len(trades[trades['exit_reason'] == 'sl_hit'])
    summary['timeout_rate'] = len(trades[trades['exit_reason'] == 'timeout']) / len(trades)
    
    pos_r = trades[trades['pnl_r'] > 0]['pnl_r'].sum()
    neg_r = abs(trades[trades['pnl_r'] < 0]['pnl_r'].sum())
    summary['PF'] = pos_r / neg_r if neg_r != 0 else 0
    summary['expectancy_R'] = trades['pnl_r'].mean()
    summary['avg_R'] = trades['pnl_r'].mean()
    summary['median_R'] = trades['pnl_r'].median()
    summary['cumulative_R'] = trades['pnl_r'].sum()
    
    # Max Drawdown R
    trades = trades.sort_values('entry_time')
    trades['cum_r'] = trades['pnl_r'].cumsum()
    trades['peak'] = trades['cum_r'].cummax()
    trades['drawdown'] = trades['cum_r'] - trades['peak']
    summary['max_drawdown_R'] = trades['drawdown'].min()
    
    # Streaks
    trades['is_win'] = trades['pnl_r'] > 0
    streaks = (trades['is_win'] != trades['is_win'].shift()).cumsum()
    streak_counts = trades.groupby(streaks)['is_win'].agg(['first', 'count'])
    summary['max_win_streak'] = streak_counts[streak_counts['first'] == True]['count'].max()
    summary['max_loss_streak'] = streak_counts[streak_counts['first'] == False]['count'].max()
    
    summary['trades_per_month'] = len(trades) / (5 * 12)
    summary['trades_per_year'] = len(trades) / 5
    
    yearly = trades.groupby('year')['pnl_r'].agg(['count', 'sum', 'mean']).reset_index()
    yearly.columns = ['year', 'N', 'total_R', 'avg_R']
    summary['yearly_positive_ratio'] = len(yearly[yearly['total_R'] > 0]) / len(yearly)
    summary['best_year'] = int(yearly.loc[yearly['total_R'].idxmax()]['year'])
    summary['worst_year'] = int(yearly.loc[yearly['total_R'].idxmin()]['year'])
    
    summary['news_blocked_count'] = int(results['stats'].get('news_blocked', 0))
    
    # Standardize types for JSON
    def clean_dict(d):
        cleaned = {}
        for k, v in d.items():
            if isinstance(v, (np.int64, np.int32)):
                cleaned[k] = int(v)
            elif isinstance(v, (np.float64, np.float32)):
                cleaned[k] = float(v)
            elif isinstance(v, dict):
                cleaned[k] = clean_dict(v)
            elif pd.isna(v):
                cleaned[k] = None
            else:
                cleaned[k] = v
        return cleaned

    summary = clean_dict(summary)
    
    # Save outputs
    # Result by year JSON
    yearly_json = {}
    for _, row in yearly.iterrows():
        y = int(row['year'])
        y_trades = trades[trades['year'] == y]
        pos_y = y_trades[y_trades['pnl_r'] > 0]['pnl_r'].sum()
        neg_y = abs(y_trades[y_trades['pnl_r'] < 0]['pnl_r'].sum())
        yearly_json[str(y)] = {
            "N": int(row['N']),
            "total_R": round(row['total_R'], 4),
            "avg_R": round(row['avg_R'], 4),
            "pf": round(pos_y / neg_y, 4) if neg_y != 0 else 0,
            "win_rate": round(len(y_trades[y_trades['pnl_r'] > 0]) / len(y_trades), 4)
        }
    summary['result_by_year_json'] = json.dumps(yearly_json)
    
    # Save outputs
    print("Saving outputs...")
    trades.to_csv(output_dir / "trades_2015_2019.csv", index=False)
    yearly.to_csv(output_dir / "yearly_breakdown_2015_2019.csv", index=False)
    
    monthly = trades.groupby('month')['pnl_r'].agg(['count', 'sum', 'mean']).reset_index()
    monthly.to_csv(output_dir / "monthly_breakdown_2015_2019.csv", index=False)
    
    weekday = trades.groupby('weekday')['pnl_r'].agg(['count', 'sum', 'mean']).reset_index()
    weekday.to_csv(output_dir / "weekday_breakdown_2015_2019.csv", index=False)
    
    level = trades.groupby('level_type' if 'level_type' in trades.columns else 'level_group')['pnl_r'].agg(['count', 'sum', 'mean']).reset_index()
    level.to_csv(output_dir / "level_breakdown_2015_2019.csv", index=False)
    
    session = trades.groupby('session')['pnl_r'].agg(['count', 'sum', 'mean']).reset_index()
    session.to_csv(output_dir / "session_breakdown_2015_2019.csv", index=False)
    
    hourly = trades.groupby('hour')['pnl_r'].agg(['count', 'sum', 'mean']).reset_index()
    hourly.to_csv(output_dir / "hourly_breakdown_2015_2019.csv", index=False)
    
    # News context breakdown
    if 'nearest_news_event' in trades.columns:
        news_br = trades.groupby('nearest_news_event')['pnl_r'].agg(['count', 'sum', 'mean']).reset_index()
        news_br.to_csv(output_dir / "news_filter_breakdown_2015_2019.csv", index=False)
    
    drawdown = trades[['entry_time', 'cum_r', 'drawdown']]
    drawdown.to_csv(output_dir / "drawdown_curve_2015_2019.csv", index=False)
    
    with open(output_dir / "summary_2015_2019.json", 'w') as f:
        json.dump(summary, f, indent=2)
        
    # Generate MD Summary
    md_content = f"""# Resumen Backtest EURUSD 2015-2019
    
**Estrategia:** SCBI_M5_GLOBAL
**Periodo:** 2015-01-01 a 2019-12-31
**Generado:** {datetime.now(timezone.utc).isoformat()}

## Métricas Clave
- **Sample Size:** {summary['sample_size']}
- **Win Rate:** {summary['win_rate']:.2%}
- **Profit Factor:** {summary['PF']:.3f}
- **Expectancy:** {summary['expectancy_R']:.3f}R
- **Total R:** {summary['cumulative_R']:.2f}R
- **Max DD:** {summary['max_drawdown_R']:.2f}R

## Breakdown Anual
| Año | N | Total R | Win Rate | PF |
|-----|---|---------|----------|----|
"""
    for y, stats in yearly_json.items():
        md_content += f"| {y} | {stats['N']} | {stats['total_R']:.2f}R | {stats['win_rate']:.2%} | {stats['pf']:.3f} |\n"
        
    md_content += f"""
## Otros Datos
- **Trades/Mes:** {summary['trades_per_month']:.2f}
- **Timeout Rate:** {summary['timeout_rate']:.2%}
- **Best Year:** {summary['best_year']}
- **Worst Year:** {summary['worst_year']}
- **News Blocked:** {summary['news_blocked_count']}
"""
    with open(output_dir / "summary_2015_2019.md", 'w') as f:
        f.write(md_content)
        
    # Validation Notes
    notes = f"""# Notas de Validación 2015-2019
    
- **Timezone:** Convertido de UTC (CSV) a America/New_York (Backtest).
- **Datos:** Usados archivos certificados en staging (`data_intake_2015_2019/prepared/`).
- **Spread:** Simulación basada en buffers de la estrategia principal (0.3 pips long).
- **Noticias:** Aplicado filtro ±30m con Fortress Candidate 2015-2019.
- **Sunday Fix:** Aplicado colapso Viernes-Domingo para PDH/PDL.
"""
    with open(output_dir / "validation_notes_2015_2019.md", 'w') as f:
        f.write(notes)
    
    print("Backtest completed successfully.")

if __name__ == "__main__":
    try:
        run_backtest_2015_2019()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
