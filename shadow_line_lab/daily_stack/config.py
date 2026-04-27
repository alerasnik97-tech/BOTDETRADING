import os

# Configuración de Rutas (Aisladas)
BASE_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
SHADOW_DIR = os.path.join(BASE_DIR, "shadow_line_lab")
STACK_DIR = os.path.join(SHADOW_DIR, "daily_stack")
OUTPUTS_DIR = os.path.join(STACK_DIR, "outputs")
RESULTS_DIR = os.path.join(SHADOW_DIR, "results")

# Logs y Resúmenes
OPERATIONAL_LOG = os.path.join(RESULTS_DIR, "shadow_daily_operational_log.csv")
DAILY_SCORECARD_JSON = os.path.join(OUTPUTS_DIR, "shadow_daily_scorecard.json")
DAILY_SCORECARD_MD = os.path.join(OUTPUTS_DIR, "shadow_daily_scorecard.md")
INCUBATION_SUMMARY_JSON = os.path.join(OUTPUTS_DIR, "shadow_incubation_summary.json")
INCUBATION_SUMMARY_MD = os.path.join(OUTPUTS_DIR, "shadow_incubation_summary.md")
