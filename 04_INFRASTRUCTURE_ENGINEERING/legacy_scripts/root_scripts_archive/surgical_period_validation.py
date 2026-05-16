from __future__ import annotations
import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, UTC

# Agregar el directorio raíz al path
PROJECT_ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
sys.path.append(str(PROJECT_ROOT))

from institutional_research_candidate_lab.config import CandidateConfig, LabPaths, default_paths
from institutional_research_candidate_lab.orchestrator import load_inputs, execute_candidate
from institutional_research_candidate_lab.reporting import build_variant_row

def run_surgical_validation():
    print("=== INICIANDO VALIDACION QUIRURGICA 2026-01-01 -> 2026-04-23 ===")
    
    # 1. Definir rutas
    paths = default_paths(PROJECT_ROOT)
    
    # 2. Definir Configuración del Candidato Shadow EXACTO
    # Basado en shadow_candidate_spec.md
    config = CandidateConfig(
        variant_id="tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_0_1_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m",
        profile_name="shadow_candidate",
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
    
    # 3. Cargar Datos
    print(f"Cargando datos para el periodo {config.start_date} a {config.end_date}...")
    h1, m5, news, coverage = load_inputs(paths, start_date=config.start_date, end_date=config.end_date)
    
    print(f"H1 bars: {len(h1)}")
    print(f"M5 bars: {len(m5)}")
    print(f"News events: {len(news)}")
    
    # 4. Ejecutar Validación
    print("Ejecutando simulación...")
    result, row = execute_candidate(config, h1=h1, m5=m5, news=news)
    
    # 5. Generar Carpeta de Salida
    output_dir = PROJECT_ROOT / "institutional_research_candidate_lab" / "outputs" / "period_validation_2026_01_01_to_2026_04_23"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 6. Guardar Archivos
    print(f"Guardando resultados en {output_dir}...")
    
    # Trades CSV
    trades_df = result["trades"]
    trades_file = output_dir / "trades_2026_01_01_to_2026_04_23.csv"
    trades_df.to_csv(trades_file, index=False)
    
    # Summary JSON
    summary_data = {
        "config": result["config"],
        "stats": result["stats"],
        "metrics": row,
        "coverage": coverage,
        "generated_at": result["generated_at_utc"]
    }
    summary_json = output_dir / "summary_2026_01_01_to_2026_04_23.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
        
    # Breakdowns (Manual calculation as reporting.py might not have specific functions for these files)
    if not trades_df.empty:
        # Monthly breakdown
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True)
        trades_df["month"] = trades_df["entry_time"].dt.strftime("%Y-%m")
        monthly = trades_df.groupby("month")["pnl_r"].agg(["count", "sum", "mean"]).reset_index()
        monthly.to_csv(output_dir / "monthly_breakdown_2026_q1_q2_partial.csv", index=False)
        
        # Weekday breakdown
        trades_df["weekday"] = trades_df["entry_time"].dt.day_name()
        weekday = trades_df.groupby("weekday")["pnl_r"].agg(["count", "sum", "mean"]).reset_index()
        weekday.to_csv(output_dir / "weekday_breakdown_2026_q1_q2_partial.csv", index=False)
        
        # Level breakdown
        level_br = trades_df.groupby("level_name")["pnl_r"].agg(["count", "sum", "mean"]).reset_index()
        level_br.to_csv(output_dir / "level_breakdown_2026_q1_q2_partial.csv", index=False)
        
        # News filter breakdown (approximate by looking at audit)
        audit_df = result["sweep_audit"]
        news_blocked = audit_df[audit_df["status"] == "NEWS_BLOCKED"]
        news_blocked.to_csv(output_dir / "news_filter_breakdown_2026_q1_q2_partial.csv", index=False)
    
    # 7. Generar Notas de Validación
    notes_file = output_dir / "validation_notes_2026_01_01_to_2026_04_23.md"
    with open(notes_file, "w", encoding="utf-8") as f:
        f.write(f"# Validación de Período 2026-01-01 a 2026-04-23\n\n")
        f.write(f"- **Estrategia:** `{config.variant_id}`\n")
        f.write(f"- **Timezone:** America/New_York (US/Eastern)\n")
        f.write(f"- **H1 Bars:** {len(h1)}\n")
        f.write(f"- **M5 Bars:** {len(m5)}\n")
        f.write(f"- **Trades Encontrados:** {len(trades_df)}\n")
        f.write(f"- **PNL Total (R):** {row.get('pnl_r', 0.0)}\n")
        f.write(f"- **Win Rate:** {row.get('win_rate', 0.0)}%\n")
        f.write(f"- **Expectancy R:** {row.get('expectancy_r', 0.0)}\n")
        f.write(f"\n## Notas Técnicas\n")
        f.write(f"- Los datos se cargaron de las rutas preparadas oficiales.\n")
        f.write(f"- El período quedó acotado estrictamente a lo solicitado.\n")
        if len(h1) > 0 and h1.index[-1].date() < pd.Timestamp("2026-04-23").date():
            f.write(f"- **GAP DETECTADO:** Los datos disponibles terminan en {h1.index[-1].date()}. El 23 de abril no está cubierto por la fuente actual.\n")
        f.write(f"\n--- Generado por Antigravity en sesión institucional ---\n")

    print("=== VALIDACION COMPLETADA ===")

if __name__ == "__main__":
    run_surgical_validation()
