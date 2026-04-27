import os

# Configuración de Rutas (Aisladas)
BASE_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
SHADOW_DIR = os.path.join(BASE_DIR, "shadow_line_lab")
REVIEW_DIR = os.path.join(SHADOW_DIR, "checkpoint_review")
OUTPUTS_DIR = os.path.join(REVIEW_DIR, "outputs")
RESULTS_DIR = os.path.join(SHADOW_DIR, "results")

# Inputs principales
OPERATIONAL_LOG = os.path.join(RESULTS_DIR, "shadow_daily_operational_log.csv")
SHADOW_LEDGER = os.path.join(RESULTS_DIR, "shadow_ledger.csv")
SCORECARD_JSON = os.path.join(RESULTS_DIR, "shadow_evidence_scorecard.json")

# Outputs del Checkpoint Review
CHECKPOINT_REVIEW_JSON = os.path.join(OUTPUTS_DIR, "shadow_checkpoint_review.json")
CHECKPOINT_REVIEW_MD = os.path.join(OUTPUTS_DIR, "shadow_checkpoint_review.md")
CHECKPOINT_HISTORY_CSV = os.path.join(OUTPUTS_DIR, "shadow_checkpoint_history.csv")
ESCALATION_RECOMMENDATION_MD = os.path.join(OUTPUTS_DIR, "shadow_escalation_recommendation.md")
