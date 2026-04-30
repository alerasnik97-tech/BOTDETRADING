import json
import os
import time

class AlertState:
    def __init__(self, state_file):
        self.state_file = state_file
        self.state = self.load_state()

    def load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {"alerts": {}}

    def save_state(self):
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2)
        except:
            pass

    def should_send(self, dedup_key, severity, cooldown_minutes=10):
        # Critical bypasses long cooldown (e.g. 1 min instead of 10)
        actual_cooldown = 1 if severity == "CRITICAL" else cooldown_minutes
        
        last_sent = self.state["alerts"].get(dedup_key, 0)
        now = time.time()
        
        if (now - last_sent) > (actual_cooldown * 60):
            self.state["alerts"][dedup_key] = now
            self.save_state()
            return True
        return False
