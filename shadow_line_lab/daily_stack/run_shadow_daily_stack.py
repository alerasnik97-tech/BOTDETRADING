import os
import sys
import json
import pandas as pd
from datetime import datetime

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shadow_line_lab import orchestrator
from shadow_line_lab.evidence_tribunal import evaluator, reporting as tribunal_reporting
from shadow_line_lab.daily_stack import config, aggregator, summary_builder, reporting as stack_reporting

class ShadowDailyStack:
    def __init__(self):
        self.stack_status = "SHADOW_STACK_INIT"
        self.run_results = {}

    def run_complete_stack(self, date_str=None):
        if date_str is None:
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
        
        print(f"--- Iniciando Shadow Daily Stack: {date_str} ---")
        
        try:
            # 1. Ejecutar Shadow Orchestrator
            orch = orchestrator.ShadowOrchestrator()
            orch_res = orch.run_daily_process(date_str)
            
            # 2. Ejecutar Evidence Tribunal
            trib = evaluator.ShadowEvidenceTribunal()
            trib_scorecard = trib.evaluate()
            tribunal_reporting.generate_reports(trib_scorecard)
            
            # 3. Consolidar Resultados
            full_data = aggregator.consolidate(orch_res, trib_scorecard)
            
            # 4. Actualizar Log Operacional
            aggregator.update_operational_log(full_data)
            
            # 5. Generar Scorecard Diaria
            stack_reporting.generate_daily_scorecard(full_data)
            
            # 6. Generar Resumen Acumulado
            summary = summary_builder.build_incubation_summary(full_data)
            stack_reporting.generate_incubation_summary(summary)
            
            self.stack_status = "SHADOW_STACK_OK"
            print(f"--- Shadow Daily Stack Completado: {self.stack_status} ---")
            return self.stack_status

        except Exception as e:
            print(f"Error crítico en el stack: {e}")
            self.stack_status = "SHADOW_STACK_BLOCKED_BY_REAL_ERROR"
            return self.stack_status

if __name__ == "__main__":
    stack = ShadowDailyStack()
    stack.run_complete_stack()
