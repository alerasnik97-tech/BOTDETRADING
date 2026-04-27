import os
import json
import pandas as pd
from datetime import datetime
from micro_pilot_gate import config

class MicroPilotGate:
    def __init__(self):
        self.verdict = "NOT_READY_FOR_MICRO_PILOT"
        self.gates = {
            "research_robustness": False,
            "shadow_governance": False,
            "shadow_evidence_min": False,
            "risk_containment": False
        }
        self.risk_protocol = {
            "risk_per_trade": config.RISK_PER_TRADE,
            "max_trades_per_day": config.MAX_TRADES_PER_DAY,
            "daily_stop": "1.0%",
            "weekly_stop": "2.5%",
            "pilot_defaults_conservative": True
        }

    def evaluate(self):
        # 1. Research Robustness
        self.gates["research_robustness"] = self._check_research()
        
        # 2. Shadow Governance
        self.gates["shadow_governance"] = self._check_governance()
        
        # 3. Shadow Evidence
        self.gates["shadow_evidence_min"] = self._check_evidence()
        
        # 4. Risk Containment
        self.gates["risk_containment"] = self._check_risk()

        # Decisión Final
        if all(self.gates.values()):
            self.verdict = "MICRO_PILOT_ALLOWED"
        else:
            self.verdict = "NOT_READY_FOR_MICRO_PILOT"

        return self.generate_scorecard_data()

    def _check_research(self):
        return os.path.exists(config.CANDIDATE_SPEC)

    def _check_governance(self):
        if not os.path.exists(config.AUTOPILOT_STATUS): return False
        try:
            with open(config.AUTOPILOT_STATUS, 'r') as f:
                status = json.load(f)
            return status.get("overall_status") == "SHADOW_AUTOPILOT_OK"
        except:
            return False

    def _check_evidence(self):
        if not os.path.exists(config.AUTOPILOT_LOG): return False
        try:
            history = pd.read_csv(config.AUTOPILOT_LOG)
            total_n = int(history["trade_count"].sum())
            return bool(total_n >= config.MIN_SHADOW_TRADES)
        except:
            return False

    def _check_risk(self):
        ks = os.path.join(config.GATE_DIR, "kill_switch_rules.md")
        ac = os.path.join(config.GATE_DIR, "activation_checklist.md")
        return os.path.exists(ks) and os.path.exists(ac)

    def generate_scorecard_data(self):
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "verdict": self.verdict,
            "gates": self.gates,
            "risk_protocol": self.risk_protocol,
            "recommendation": self._get_recommendation()
        }

    def _get_recommendation(self):
        if self.verdict == "MICRO_PILOT_ALLOWED":
            return "Habilitar micro piloto bajo protocolo ultra-conservador. N=20 trades reales de prueba."
        return f"Continuar en fase de incubación Shadow hasta completar N={config.MIN_SHADOW_TRADES} trades de evidencia."

if __name__ == "__main__":
    gate = MicroPilotGate()
    res = gate.evaluate()
    with open(config.SCORECARD_JSON, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"Evaluación del Micro Pilot Gate finalizada: {res['verdict']}")
