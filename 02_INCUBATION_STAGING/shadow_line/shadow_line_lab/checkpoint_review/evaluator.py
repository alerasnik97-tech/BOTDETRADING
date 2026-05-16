import json
import pandas as pd
import os
from datetime import datetime
from shadow_line_lab.checkpoint_review import config, thresholds

class CheckpointEvaluator:
    def __init__(self):
        self.metrics = {}
        self.decision = "CHECKPOINT_NOT_REACHED"
        self.target_n = 0
        self.reached_checkpoint = None

    def evaluate(self):
        # 1. Cargar evidencia
        ledger = self.load_ledger()
        if ledger is None or ledger.empty:
            return self.generate_empty_review()

        # 2. Calcular métricas reales
        self.metrics = self.calculate_metrics(ledger)
        n = self.metrics["total_trades"]

        # 3. Identificar checkpoint actual
        self.target_n = self.identify_target_n(n)
        
        if self.target_n == 0:
            self.decision = "CHECKPOINT_NOT_REACHED"
            return self.generate_review()

        # 4. Aplicar lógica de decisión por hito
        self.reached_checkpoint = thresholds.CHECKPOINT_RULES[self.target_n]
        self.decision = self.determine_decision(n)

        return self.generate_review()

    def load_ledger(self):
        if not os.path.exists(config.SHADOW_LEDGER): return None
        return pd.read_csv(config.SHADOW_LEDGER)

    def calculate_metrics(self, ledger):
        executed = ledger[ledger['classification'] == 'TRADE_EXECUTED'].copy()
        executed['pnl_r'] = pd.to_numeric(executed['pnl_r'], errors='coerce').fillna(0.0)
        
        wins = executed[executed['pnl_r'] > 0]['pnl_r'].sum()
        losses = abs(executed[executed['pnl_r'] < 0]['pnl_r'].sum())
        pf = round(wins / losses, 4) if losses > 0 else (round(wins, 4) if wins > 0 else 0.0)
        
        equity = executed['pnl_r'].cumsum()
        max_dd = round((equity.expanding().max() - equity).max(), 4) if not equity.empty else 0.0

        return {
            "total_trades": len(executed),
            "cumulative_r": round(executed['pnl_r'].sum(), 4),
            "pf": pf,
            "expectancy_r": round(executed['pnl_r'].mean(), 4) if not executed.empty else 0.0,
            "max_dd_r": max_dd,
            "win_rate": round((len(executed[executed['pnl_r'] > 0]) / len(executed)) * 100, 2) if not executed.empty else 0.0
        }

    def identify_target_n(self, n):
        if n >= 20: return 20
        if n >= 10: return 10
        if n >= 5: return 5
        return 0

    def determine_decision(self, n):
        rules = self.reached_checkpoint
        m = self.metrics
        
        # Fallo Crítico (HOLD)
        if m["max_dd_r"] > rules["max_dd"]: return "HOLD_SHADOW"
        
        # Veredicto por hito
        if n >= 20:
            if m["pf"] >= rules["min_pf"] and m["expectancy_r"] >= rules["min_expectancy"]:
                return "READY_FOR_NEXT_GATE"
            return "CONTINUE_INCUBATION"
        
        if n >= 10:
            if m["pf"] >= rules["min_pf"]: return "CONTINUE_INCUBATION"
            return "WARNING_REVIEW"
            
        if n >= 5:
            if m["expectancy_r"] >= rules["min_expectancy"]: return "CONTINUE_INCUBATION"
            return "WARNING_REVIEW"
            
        return "CONTINUE_INCUBATION"

    def generate_review(self):
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "current_n": self.metrics.get("total_trades", 0),
            "checkpoint_target": self.target_n,
            "checkpoint_name": self.reached_checkpoint["name"] if self.reached_checkpoint else "NONE",
            "decision": self.decision,
            "metrics": self.metrics,
            "baseline_comparison": self.calculate_deltas()
        }

    def calculate_deltas(self):
        if not self.metrics: return {}
        base = thresholds.CANDIDATE_BASELINE
        return {
            "pf_delta": round(self.metrics["pf"] - base["pf"], 4),
            "expectancy_delta": round(self.metrics["expectancy_r"] - base["expectancy_r"], 4)
        }

    def generate_empty_review(self):
        return {"decision": "CHECKPOINT_NOT_REACHED", "current_n": 0, "status": "NO_EVIDENCE"}
