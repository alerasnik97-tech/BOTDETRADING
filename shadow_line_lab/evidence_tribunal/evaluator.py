import json
import pandas as pd
import os
from datetime import datetime
from shadow_line_lab.evidence_tribunal import config, thresholds, scoring

class ShadowEvidenceTribunal:
    def __init__(self):
        self.metrics = {}
        self.verdict = "SHADOW_INCUBATING"
        self.alerts = []

    def evaluate(self):
        # 1. Cargar Inputs
        ledger = self.load_ledger()
        if ledger is None or ledger.empty:
            return self.generate_error_scorecard("MISSING_OR_EMPTY_LEDGER")

        # 2. Calcular Métricas
        self.metrics = scoring.calculate_shadow_metrics(ledger)
        
        # 3. Determinar Veredicto basado en Gates
        self.verdict = self.determine_verdict()
        
        # 4. Generar Alertas
        self.alerts = self.check_alerts(ledger)
        
        # 5. Si hay alertas críticas, degradar veredicto
        if any(a['severity'] == 'CRITICAL' for a in self.alerts):
            self.verdict = "SHADOW_HOLD"
        elif any(a['severity'] == 'WARNING' for a in self.alerts) and self.verdict != "SHADOW_HOLD":
            self.verdict = "SHADOW_WARNING"

        return self.generate_scorecard()

    def load_ledger(self):
        if not os.path.exists(config.LEDGER_FILE):
            return None
        return pd.read_csv(config.LEDGER_FILE)

    def determine_verdict(self):
        n = self.metrics.get("total_shadow_trades", 0)
        
        current_gate = "N_0"
        if n >= 20: current_gate = "N_20"
        elif n >= 10: current_gate = "N_10"
        elif n >= 5: current_gate = "N_5"
        
        gate_rules = thresholds.GATES[current_gate]
        
        # Validación de Fallo por DD o PF en gates maduros
        if n >= 5:
            if self.metrics["max_drawdown_R"] > gate_rules["max_dd"]: return "SHADOW_HOLD"
            if self.metrics["pf"] < gate_rules["min_pf"]: return "SHADOW_WARNING"
            
        return gate_rules["verdict"]

    def check_alerts(self, ledger):
        alerts = []
        conf = thresholds.ALERT_CONFIG
        
        if self.metrics.get("current_streak_negative", 0) >= conf["max_consecutive_losses"]:
            alerts.append({
                "code": "HIGH_LOSS_STREAK", "severity": "WARNING",
                "explanation": f"Se han detectado {self.metrics['current_streak_negative']} pérdidas consecutivas.",
                "recommended_action": "Revisar si el mercado cambió de régimen o si hay error de ejecución."
            })
            
        if self.metrics.get("timeout_rate", 0) > conf["max_timeout_rate"]:
            alerts.append({
                "code": "HIGH_TIMEOUT_RATE", "severity": "WARNING",
                "explanation": f"Tasa de timeout ({self.metrics['timeout_rate']}) supera el umbral de {conf['max_timeout_rate']}.",
                "recommended_action": "Evaluar si el TP es demasiado ambicioso o la volatilidad es baja."
            })

        if self.metrics.get("max_drawdown_R", 0) > 10.0:
            alerts.append({
                "code": "CRITICAL_DRAWDOWN", "severity": "CRITICAL",
                "explanation": f"Drawdown de {self.metrics['max_drawdown_R']}R supera el límite de seguridad estructural.",
                "recommended_action": "HOLD inmediato. Re-evaluar robustez en laboratorio."
            })
            
        return alerts

    def generate_scorecard(self):
        scorecard = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "variant_id": "tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_0_1_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m",
            "verdict": self.verdict,
            "metrics": self.metrics,
            "alerts": self.alerts,
            "status": "OPERATIONAL"
        }
        return scorecard

    def generate_error_scorecard(self, error_code):
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "verdict": "SHADOW_HOLD",
            "status": "ERROR",
            "error": error_code
        }
