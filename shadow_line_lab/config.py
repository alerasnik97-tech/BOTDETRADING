import os

# Configuración de Rutas (Aisladas)
BASE_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
SHADOW_DIR = os.path.join(BASE_DIR, "shadow_line_lab")
RESULTS_DIR = os.path.join(SHADOW_DIR, "results")
OUTPUTS_DIR = os.path.join(SHADOW_DIR, "outputs")

# Rutas de Datos (Read-Only)
DATA_DIR = os.path.join(BASE_DIR, "data")
NEWS_FILE = os.path.join(DATA_DIR, "news_eurusd_am_fortress_v3.csv")
PRICE_DIRS = [
    os.path.join(BASE_DIR, "data_free_2020", "prepared"),
    os.path.join(BASE_DIR, "data_candidates_2022_2025", "prepared")
]

# Configuración del Candidato: tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_0_1_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m
STRATEGY_CONFIG = {
    "variant_id": "tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_0_1_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m",
    "pair": "EURUSD",
    "levels": ["asia_h", "asia_l", "london_h", "london_l", "pdh", "pdl"],
    "confirmation_window": (0, 1), # +0h a +1h
    "confirmation_pick": "first",
    "confirmation_mode": "close_reclaim",
    "long_entry_buffer": 0.3,
    "short_entry_buffer": 0.0,
    "sl_buffer": 1.0,
    "tp_r": 1.5,
    "timeout_hours": 4,
    "min_risk_pips": 2.0,
    "news_filter_minutes": 30, # +/- 30m del sweep
    "max_trades_per_day": 1
}

# Archivos de Salida
LEDGER_FILE = os.path.join(RESULTS_DIR, "shadow_ledger.csv")
DAILY_STATUS_FILE = os.path.join(RESULTS_DIR, "shadow_daily_status.json")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "shadow_summary.json")
REPORT_FILE = os.path.join(OUTPUTS_DIR, "shadow_run_report.md")
