import os
import sys
import json
import pandas as pd
from datetime import datetime

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shadow_line_lab import orchestrator
from shadow_line_lab.evidence_tribunal import evaluator as trib_evaluator
from shadow_line_lab.daily_stack import run_shadow_daily_stack
from shadow_line_lab.checkpoint_review import evaluator as check_evaluator
from shadow_line_lab.shadow_autopilot import config

class ShadowCoordinator:
    def __init__(self):
        self.state = {
            "run_date": datetime.utcnow().isoformat() + "Z",
            "runner_status": "PENDING",
            "tribunal_status": "PENDING",
            "stack_status": "PENDING",
            "checkpoint_status": "PENDING",
            "overall_status": "SHADOW_AUTOPILOT_INIT"
        }

    def execute_full_pipeline(self, date_str=None):
        if date_str is None:
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
        
        self.state["target_date"] = date_str
        print(f"--- Iniciando Shadow Autopilot: {date_str} ---")
        
        try:
            # 1. Runner
            orch = orchestrator.ShadowOrchestrator()
            orch_res = orch.run_daily_process(date_str)
            self.state["runner_status"] = orch_res.get("classification")
            
            # 2. Tribunal
            trib = trib_evaluator.ShadowEvidenceTribunal()
            trib_res = trib.evaluate()
            self.state["tribunal_status"] = trib_res.get("verdict")
            
            # 3. Daily Stack
            stack = run_shadow_daily_stack.ShadowDailyStack()
            stack_res = stack.run_complete_stack(date_str)
            self.state["stack_status"] = stack_res
            
            # 4. Checkpoint Review
            check = check_evaluator.CheckpointEvaluator()
            check_res = check.evaluate()
            self.state["checkpoint_status"] = check_res.get("decision")
            
            # 5. Consolidación de métricas
            self.state["trade_count"] = trib_res["metrics"].get("total_shadow_trades", 0)
            self.state["cumulative_R"] = trib_res["metrics"].get("cumulative_R", 0.0)
            self.state["alert_count"] = len(trib_res.get("alerts", []))
            self.state["overall_status"] = "SHADOW_AUTOPILOT_OK"
            
            return self.state

        except Exception as e:
            print(f"Error crítico en el Autopilot: {e}")
            self.state["overall_status"] = "SHADOW_AUTOPILOT_BLOCKED_BY_REAL_ERROR"
            self.state["error"] = str(e)
            return self.state
