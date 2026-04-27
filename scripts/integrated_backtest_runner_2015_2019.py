import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from research_lab.config import EngineConfig, NewsConfig, NY_TZ, with_execution_mode, INITIAL_CAPITAL
from research_lab.data_loader import load_high_precision_package, fx_market_mask, validate_price_frame, resample_ohlcv_to_timeframe, prepare_common_frame, fx_session_date, atr
from research_lab.engine import run_backtest, entry_open_index
from research_lab.strategies import am_silver_bullet_ny_v2
from research_lab.news_filter import build_entry_block
from research_lab.ict_primitives import add_fvg_columns, add_pivot_structure_columns
from research_lab.report import summarize_result

OUTPUT_ROOT = PROJECT_ROOT / "institutional_research_candidate_lab" / "outputs" / "period_validation_2015_01_01_to_2019_12_31"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Parámetros seleccionados (de selected_params.json)
BEST_PARAMS = {
    "break_even_at_r": 1.0,
    "cooldown_bars": 0,
    "entry_end": "11:00",
    "entry_start": "10:00",
    "max_fvg_after_mss_bars": 3,
    "max_hold_bars": 60,
    "min_fvg_pips": 0.5,
    "retest_window_bars": 10,
    "session_name": "am_08_11",
    "stop_buffer_pips": 2.0,
    "target_rr": 2.0,
    "variant_label": "m1_mss_midpoint_reprice"
}

def run_historical_backtest(start_year, end_year, m1_dir, news_file):
    print(f"Starting Integrated Backtest {start_year}-{end_year}...")
    
    # 1. Load Data
    package = load_high_precision_package("EURUSD", m1_dir)
    
    # Filter by date range
    start_ts = pd.Timestamp(f"{start_year}-01-01", tz=NY_TZ)
    end_ts = pd.Timestamp(f"{end_year}-12-31", tz=NY_TZ) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    
    filtered_m1 = {}
    for side, source in package.items():
        frame = source.loc[(source.index >= start_ts) & (source.index <= end_ts)].copy()
        frame = frame[fx_market_mask(frame.index)].copy()
        validate_price_frame(frame)
        filtered_m1[f"{side}_m1"] = frame
        
    mid_m1 = filtered_m1["mid_m1"]
    
    # 2. Build Research Frame (M1 with M5 context)
    print("Building research frame...")
    base = mid_m1.copy()
    base["session_date"] = fx_session_date(base.index)
    base["atr14"] = atr(base, 14)
    base["bar_range"] = base["high"] - base["low"]
    base["range_atr"] = base["bar_range"] / base["atr14"].replace(0.0, np.nan)
    
    # Focus on window (09:30 - 12:00)
    print("DEBUG: Filtering research window...")
    research = base.between_time("09:30", "12:00").copy()
    print(f"DEBUG: Processing {len(research.groupby('session_date'))} days...")
    day_frames = []
    for date, day in research.groupby("session_date", sort=False):
        day = add_fvg_columns(day)
        day = add_pivot_structure_columns(day)
        day_frames.append(day)
    print("DEBUG: Concatenating day frames...")
    m1_frame = pd.concat(day_frames).sort_index()
    print("DEBUG: Building M5 context...")
    
    # M5 context
    m5_source = resample_ohlcv_to_timeframe(mid_m1, "M5")
    m5_frame = prepare_common_frame(m5_source, target_timeframe="M5")
    context = pd.DataFrame(index=m5_frame.index)
    context["ctx_m5_sb_anchor_high"] = m5_frame["session_range_high_03_00_08_30"]
    context["ctx_m5_sb_anchor_low"] = m5_frame["session_range_low_03_00_08_30"]
    context["ctx_m5_swept_anchor_high"] = (m5_frame["day_running_high"] > m5_frame["session_range_high_03_00_08_30"]).astype(bool)
    context["ctx_m5_swept_anchor_low"] = (m5_frame["day_running_low"] < m5_frame["session_range_low_03_00_08_30"]).astype(bool)
    
    print("DEBUG: Joining context...")
    full_frame = m1_frame.join(context.reindex(m1_frame.index, method="ffill")).dropna(subset=["ctx_m5_sb_anchor_high", "atr14"])
    print(f"DEBUG: Frame ready with {len(full_frame)} rows.")
    
    # 3. Setup Engine & News
    engine_config = EngineConfig(
        pair="EURUSD",
        risk_pct=0.5,
        max_spread_pips=2.0,
        slippage_pips=0.1,
        commission_per_lot_roundturn_usd=7.0,
        max_trades_per_day=1,
        session_cutoff="12:00",
        enforce_hard_stop=True,
        execution_mode="high_precision_mode"
    )
    
    news_config = NewsConfig(
        enabled=True,
        file_path=news_file,
        pre_minutes=30,
        post_minutes=60,
        forced_exit_pre_news=True,
        cancel_pending_pre_news=True,
        pre_news_exit_minutes=10,
        currencies=("USD", "EUR"),
        impact_levels=("HIGH",),
    )
    
    news_events = pd.read_csv(news_file)
    news_events["timestamp_ny"] = pd.to_datetime(news_events["timestamp_ny"], utc=True).dt.tz_convert(NY_TZ)
    
    news_block = build_entry_block(entry_open_index(full_frame.index), news_events, news_config)
    
    # 4. Run Backtest
    print("Running strategy execution...")
    # Alineación de paquete de precisión
    period_precision = {
        "bid_m1": filtered_m1["bid_m1"].loc[full_frame.index].copy(),
        "ask_m1": filtered_m1["ask_m1"].loc[full_frame.index].copy(),
        "mid_m1": filtered_m1["mid_m1"].loc[full_frame.index].copy(),
        "bid_exec": filtered_m1["bid_m1"].loc[full_frame.index].copy(),
        "ask_exec": filtered_m1["ask_m1"].loc[full_frame.index].copy(),
        "mid_exec": filtered_m1["mid_m1"].loc[full_frame.index].copy(),
        "bid_m15": filtered_m1["bid_m1"].loc[full_frame.index].copy(),
        "ask_m15": filtered_m1["ask_m1"].loc[full_frame.index].copy(),
        "mid_m15": filtered_m1["mid_m1"].loc[full_frame.index].copy(),
    }
    
    result = run_backtest(
        strategy_module=am_silver_bullet_ny_v2,
        frame=full_frame,
        params=BEST_PARAMS,
        engine_config=engine_config,
        news_block=news_block,
        news_filter_used=True,
        precision_package=period_precision,
        data_source_used="dukascopy_m1_reconstructed_2015_2019",
        news_events=news_events,
        news_settings=news_config
    )
    
    # 5. Summarize & Export
    print("Generating summaries...")
    raw_trades = result.trades.copy()
    if "pair" not in raw_trades.columns and not raw_trades.empty:
        raw_trades["pair"] = "EURUSD"
        
    for col in ["entry_time", "exit_time", "signal_time", "fill_time"]:
        if col in raw_trades.columns:
            raw_trades[col] = pd.to_datetime(raw_trades[col], utc=True)
            
    summary, trades_export, monthly_stats, yearly_stats, equity_export = summarize_result(
        am_silver_bullet_ny_v2.NAME,
        raw_trades,
        result.equity_curve,
        BEST_PARAMS,
        True,
        INITIAL_CAPITAL,
        None,
        costs_used={"execution_mode": "high_precision_mode", "cost_profile": "precision"},
        timeframe="M1",
        schedule_used={"entry_start": "10:00", "entry_end": "11:00", "force_close": "12:00"},
    )
    
    # Export files
    trades_export.to_csv(OUTPUT_ROOT / "trades_2015_2019.csv", index=False)
    yearly_stats.to_csv(OUTPUT_ROOT / "yearly_breakdown_2015_2019.csv", index=False)
    monthly_stats.to_csv(OUTPUT_ROOT / "monthly_breakdown_2015_2019.csv", index=False)
    equity_export.to_csv(OUTPUT_ROOT / "drawdown_curve_2015_2019.csv", index=False)
    
    with open(OUTPUT_ROOT / "summary_2015_2019.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    # Manual breakdowns (weekday, level, session)
    if not trades_export.empty:
        print("Manual breakdowns skipped for safety.")

    # Generate MD Summary
    with open(OUTPUT_ROOT / "summary_2015_2019.md", "w", encoding="utf-8") as f:
        f.write(f"# Backtest Report: 2015-2019 (Silver Bullet V2)\n\n")
        f.write(f"## Profit Factor: {summary.get('profit_factor', 0):.2f}\n")
        f.write(f"## Expectancy R: {summary.get('expectancy_r', 0):.2f}\n")
        f.write(f"## Win Rate: {summary.get('win_rate', 0)*100:.1f}%\n\n")
        f.write(f"### Yearly Breakdown:\n\n")
        f.write(yearly_stats.to_string())
        
    print(f"Backtest complete. Results in {OUTPUT_ROOT}")
    return summary

if __name__ == "__main__":
    # To be called when data is ready
    pass
