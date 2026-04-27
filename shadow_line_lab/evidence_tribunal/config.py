import os

# Configuración de Rutas (Aisladas)
BASE_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
SHADOW_DIR = os.path.join(BASE_DIR, "shadow_line_lab")
TRIBUNAL_DIR = os.path.join(SHADOW_DIR, "evidence_tribunal")
OUTPUTS_DIR = os.path.join(TRIBUNAL_DIR, "outputs")

# Inputs
RESULTS_DIR = os.path.join(SHADOW_DIR, "results")
LEDGER_FILE = os.path.join(RESULTS_DIR, "shadow_ledger.csv")
DAILY_STATUS_FILE = os.path.join(RESULTS_DIR, "shadow_daily_status.json")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "shadow_summary.json")

# Outputs del Tribunal
SCORECARD_JSON = os.path.join(OUTPUTS_DIR, "shadow_evidence_scorecard.json")
SCORECARD_MD = os.path.join(OUTPUTS_DIR, "shadow_evidence_scorecard.md")
SUMMARY_TXT = os.path.join(OUTPUTS_DIR, "shadow_evidence_summary.txt")
ALERTS_JSON = os.path.join(OUTPUTS_DIR, "shadow_alerts.json")
EVIDENCE_LOG = os.path.join(OUTPUTS_DIR, "shadow_evidence_log.csv")
