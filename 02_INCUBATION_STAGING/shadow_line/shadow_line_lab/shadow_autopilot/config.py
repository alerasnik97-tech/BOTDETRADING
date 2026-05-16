import os

# Configuración de Rutas (Aisladas)
BASE_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
SHADOW_DIR = os.path.join(BASE_DIR, "shadow_line_lab")
AUTOPILOT_DIR = os.path.join(SHADOW_DIR, "shadow_autopilot")
OUTPUTS_DIR = os.path.join(AUTOPILOT_DIR, "outputs")
RESULTS_DIR = os.path.join(SHADOW_DIR, "results")

# Logs y Estados Globales
AUTOPILOT_STATUS_JSON = os.path.join(RESULTS_DIR, "shadow_autopilot_status.json")
AUTOPILOT_STATUS_MD = os.path.join(RESULTS_DIR, "shadow_autopilot_status.md")
AUTOPILOT_LOG_CSV = os.path.join(RESULTS_DIR, "shadow_autopilot_log.csv")
AUTOPILOT_SUMMARY_JSON = os.path.join(OUTPUTS_DIR, "shadow_autopilot_summary.json")
AUTOPILOT_SUMMARY_MD = os.path.join(OUTPUTS_DIR, "shadow_autopilot_summary.md")
