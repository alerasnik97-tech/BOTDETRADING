import os

# Configuración de Rutas (Aisladas)
BASE_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
GATE_DIR = os.path.join(BASE_DIR, "micro_pilot_gate")
OUTPUTS_DIR = os.path.join(GATE_DIR, "outputs")
SHADOW_DIR = os.path.join(BASE_DIR, "shadow_line_lab")
SHADOW_RESULTS = os.path.join(SHADOW_DIR, "results")

# Inputs
AUTOPILOT_STATUS = os.path.join(SHADOW_RESULTS, "shadow_autopilot_status.json")
AUTOPILOT_LOG = os.path.join(SHADOW_RESULTS, "shadow_autopilot_log.csv")
CANDIDATE_SPEC = os.path.join(BASE_DIR, "institutional_research_candidate_lab", "outputs", "shadow_candidate_spec.md")

# Outputs
SCORECARD_JSON = os.path.join(OUTPUTS_DIR, "micro_pilot_scorecard.json")
SCORECARD_MD = os.path.join(OUTPUTS_DIR, "micro_pilot_scorecard.md")
SUMMARY_TXT = os.path.join(OUTPUTS_DIR, "micro_pilot_summary.txt")

# Thresholds Conservadores (PILOT_DEFAULTS_CONSERVATIVE)
MIN_SHADOW_TRADES = 10
MAX_DRAWDOWN_R = 10.0
MAX_TRADES_PER_DAY = 1
RISK_PER_TRADE = "0.1% a 0.25%"
