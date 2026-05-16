import sys
from pathlib import Path
import pandas as pd
import json
from datetime import datetime, timezone

PROJECT_ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
sys.path.append(str(PROJECT_ROOT))

from institutional_research_candidate_lab.config import CandidateConfig, default_paths
from institutional_research_candidate_lab.orchestrator import load_inputs, execute_candidate

def run_revalidation():
    print("=== INICIANDO RE-VALIDACION POST-FIX DOMINICAL ===")
    paths = default_paths(PROJECT_ROOT)
    output_dir = PROJECT_ROOT / "institutional_research_candidate_lab" / "outputs" / "period_validation_2026_01_01_to_2026_04_23_after_sunday_fix"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = CandidateConfig(
        variant_id="tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_0_1_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m",
        profile_name="shadow_candidate_revalidation",
        start_date="2026-01-01",
        end_date="2026-04-23",
        tp_r=1.5,
        timeout_hours=4,
        sl_buffer_pips=1.0,
        long_entry_buffer_pips=0.3,
        short_entry_buffer_pips=0.0,
        min_risk_pips=2.0,
        confirmation_window_start_hours=0,
        confirmation_window_end_hours=1,
        confirmation_mode="close_reclaim",
        confirmation_pick="first",
        level_profile="all_levels",
        news_mode="sweep_plus_minus_30m"
    )
    
    print(f"Cargando datos para el periodo {config.start_date} a {config.end_date}...")
    h1, m5, news, coverage = load_inputs(paths, start_date=config.start_date, end_date=config.end_date)
    
    print(f"Ejecutando simulación con FIX DOMINICAL aplicado...")
    result, row = execute_candidate(config, h1=h1, m5=m5, news=news)
    
    # --- EXPORTAR SEGUN MANDATO ---
    print(f"Guardando resultados en {output_dir}...")
    
    # 1. trades_after_sunday_fix.csv
    trades_df = result["trades"]
    trades_df.to_csv(output_dir / "trades_after_sunday_fix.csv", index=False)
    
    # 2. summary_after_sunday_fix.json
    summary_data = {
        "config": result["config"],
        "stats": result["stats"],
        "metrics": row,
        "coverage": coverage,
        "generated_at": result["generated_at_utc"]
    }
    with open(output_dir / "summary_after_sunday_fix.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
        
    # 3. breakdowns
    if not trades_df.empty:
        # monthly_breakdown_after_sunday_fix.csv
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True)
        trades_df["month"] = trades_df["entry_time"].dt.strftime("%Y-%m")
        trades_df.groupby("month")["pnl_r"].agg(["count", "sum", "mean"]).to_csv(output_dir / "monthly_breakdown_after_sunday_fix.csv")
        
        # weekday_breakdown_after_sunday_fix.csv
        trades_df["weekday"] = trades_df["entry_time"].dt.day_name()
        trades_df.groupby("weekday")["pnl_r"].agg(["count", "sum", "mean"]).to_csv(output_dir / "weekday_breakdown_after_sunday_fix.csv")
        
        # level_breakdown_after_sunday_fix.csv
        trades_df.groupby("level_name")["pnl_r"].agg(["count", "sum", "mean"]).to_csv(output_dir / "level_breakdown_after_sunday_fix.csv")

    # 4. monday_impact_before_vs_after.csv (comparativa de Lunes)
    # Cargar datos anteriores para comparar solo Lunes si existe el archivo
    prev_trades_path = PROJECT_ROOT / "institutional_research_candidate_lab" / "outputs" / "period_validation_2026_01_01_to_2026_04_23" / "trades_2026_01_01_to_2026_04_23.csv"
    if prev_trades_path.exists():
        prev_trades = pd.read_csv(prev_trades_path)
        prev_trades["entry_time"] = pd.to_datetime(prev_trades["entry_time"], utc=True)
        prev_mondays = prev_trades[prev_trades["entry_time"].dt.dayofweek == 0]
        curr_mondays = trades_df[trades_df["entry_time"].dt.dayofweek == 0]
        
        impact_mondays = pd.DataFrame({
            "metric": ["count", "sum_pnl_r", "mean_pnl_r"],
            "before_fix": [len(prev_mondays), prev_mondays["pnl_r"].sum(), prev_mondays["pnl_r"].mean()],
            "after_fix": [len(curr_mondays), curr_mondays["pnl_r"].sum(), curr_mondays["pnl_r"].mean()]
        })
        impact_mondays.to_csv(output_dir / "monday_impact_before_vs_after.csv", index=False)
    
    # 5. summary_after_sunday_fix.md
    with open(output_dir / "summary_after_sunday_fix.md", "w", encoding="utf-8") as f:
        f.write("# Re-validación Post-Fix Dominical (2026)\n\n")
        f.write(f"- PF: {row.get('profit_factor', 0.0):.2f}\n")
        f.write(f"- Expectancy R: {row.get('expectancy_r', 0.0):.3f}\n")
        f.write(f"- Total Trades: {len(trades_df)}\n")
        f.write(f"- Win Rate: {row.get('win_rate', 0.0):.1f}%\n")
        f.write(f"- Max Drawdown R: {row.get('max_drawdown_r', 0.0):.2f}\n")
    
    print("=== RE-VALIDACION COMPLETADA ===")

if __name__ == "__main__":
    run_revalidation()
