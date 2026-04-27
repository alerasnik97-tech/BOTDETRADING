import os
import sys
import json
import pandas as pd
from datetime import datetime

# Agregar el directorio raíz al path para importaciones
BASE_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
sys.path.append(BASE_DIR)

from shadow_line_lab.shadow_autopilot import run_shadow_autopilot, coordinator, state_manager, reporting as shadow_reporting
from micro_pilot_gate import evaluator as gate_evaluator, scorecard as gate_scorecard, config as gate_config

class ShadowToMicroRoutine:
    def __init__(self):
        self.results_dir = os.path.join(BASE_DIR, "shadow_line_lab", "results")
        self.outputs_dir = os.path.join(BASE_DIR, "shadow_line_lab", "outputs")
        self.status_file = os.path.join(self.results_dir, "shadow_to_micro_gate_daily_status.json")
        self.status_md = os.path.join(self.outputs_dir, "shadow_to_micro_gate_daily_status.md")
        self.log_csv = os.path.join(self.results_dir, "shadow_to_micro_gate_log.csv")
        self.alert_file = os.path.join(self.outputs_dir, "SHADOW_TO_MICRO_ALERT.md")

    def run_daily_routine(self):
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        print(f"--- Iniciando Rutina Diaria: Shadow -> Micro Pilot Gate ({date_str}) ---")
        
        # 1. Ejecutar Shadow Autopilot
        coord = coordinator.ShadowCoordinator()
        shadow_state = coord.execute_full_pipeline(date_str)
        state_manager.save_overall_status(shadow_state)
        state_manager.update_autopilot_log(shadow_state)
        shadow_reporting.generate_final_reports(shadow_state)
        
        # 2. Ejecutar Micro Pilot Gate
        gate = gate_evaluator.MicroPilotGate()
        gate_res = gate.evaluate()
        gate_scorecard.generate_scorecard_reports(gate_res)
        
        # 3. Consolidar Estado
        current_status = {
            "run_date": datetime.utcnow().isoformat() + "Z",
            "target_date": date_str,
            "shadow_autopilot_status": shadow_state["overall_status"],
            "total_shadow_trades": shadow_state.get("trade_count", 0),
            "current_checkpoint_status": shadow_state.get("checkpoint_status", "N/A"),
            "micro_pilot_gate_status": gate_res["verdict"],
            "main_blocker": self._get_main_blocker(gate_res),
            "recommendation": gate_res["recommendation"],
            "alert_triggered": False
        }
        
        # 4. Alerta de Cambio de Estado
        prev_status = self._load_previous_status()
        if prev_status and prev_status.get("micro_pilot_gate_status") != current_status["micro_pilot_gate_status"]:
            current_status["alert_triggered"] = True
            self._trigger_state_alert(prev_status, current_status)
        
        # 5. Guardar Reportes
        self._save_reports(current_status)
        
        # 6. Actualizar Log Histórico
        self._update_transition_log(current_status)
        
        print(f"--- Rutina Finalizada. Estado Gate: {current_status['micro_pilot_gate_status']} ---")
        return current_status

    def _get_main_blocker(self, gate_res):
        if gate_res["verdict"] == "MICRO_PILOT_ALLOWED": return "NONE"
        failed_gates = [k for k, v in gate_res["gates"].items() if not v]
        return ", ".join(failed_gates) if failed_gates else "N/A"

    def _load_previous_status(self):
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r') as f:
                    return json.load(f)
            except:
                return None
        return None

    def _trigger_state_alert(self, prev, current):
        msg = f"""# 🚨 ALERTA DE CAMBIO DE ESTADO (Shadow -> Micro Pilot)
**Fecha:** `{current['run_date']}`

El estado del Micro Pilot Gate ha cambiado:
- **Anterior:** `{prev['micro_pilot_gate_status']}`
- **Nuevo:** `{current['micro_pilot_gate_status']}`

**RECOMENDACIÓN:** 
{current['recommendation']}

---
*Este archivo se genera automáticamente cuando el veredicto del gate evoluciona.*
"""
        with open(self.alert_file, 'w', encoding='utf-8') as f:
            f.write(msg)
        print(f"!!! ALERTA: Cambio de estado detectado !!!")

    def _save_reports(self, status):
        # JSON
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump(status, f, indent=2)
        
        # MD
        md = f"""# Resumen Diario: Shadow to Micro Pilot
## Veredicto Gate: `{status['micro_pilot_gate_status']}`

**Fecha:** `{status['target_date']}`
**Shadow Autopilot:** `{status['shadow_autopilot_status']}`

---

### Estado de la Incubación
- **Trades Shadow Totales:** {status['total_shadow_trades']}
- **Hito Checkpoint:** `{status['current_checkpoint_status']}`
- **Bloqueador Principal:** `{status['main_blocker']}`

### Recomendación Institucional
**{status['recommendation']}**

---
*Consultar `micro_pilot_gate/activation_checklist.md` si el gate es ALLOWED.*
"""
        with open(self.status_md, 'w', encoding='utf-8') as f:
            f.write(md)

    def _update_transition_log(self, status):
        csv_cols = [
            "run_date", "shadow_autopilot_status", "checkpoint_status",
            "micro_pilot_gate_status", "total_shadow_trades",
            "main_blocker", "recommendation", "alert_triggered"
        ]
        new_row = {
            "run_date": status["run_date"],
            "shadow_autopilot_status": status["shadow_autopilot_status"],
            "checkpoint_status": status["current_checkpoint_status"],
            "micro_pilot_gate_status": status["micro_pilot_gate_status"],
            "total_shadow_trades": status["total_shadow_trades"],
            "main_blocker": status["main_blocker"],
            "recommendation": status["recommendation"],
            "alert_triggered": status["alert_triggered"]
        }
        
        if os.path.exists(self.log_csv):
            history = pd.read_csv(self.log_csv)
            # Evitar duplicados por target_date si fuera necesario, pero run_date es timestamp
            history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
            history.to_csv(self.log_csv, index=False)
        else:
            pd.DataFrame([new_row]).to_csv(self.log_csv, index=False)

if __name__ == "__main__":
    routine = ShadowToMicroRoutine()
    routine.run_daily_routine()
