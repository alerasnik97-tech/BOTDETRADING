import pandas as pd
import json
import os
from datetime import datetime

class MT5DemoTelemetry:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, "mt5_demo_log.csv")
        self.telemetry_file = os.path.join(output_dir, "mt5_demo_telemetry.csv")
        self._init_files()
        
    def _init_files(self):
        if not os.path.exists(self.log_file):
            pd.DataFrame(columns=["timestamp", "event", "details"]).to_csv(self.log_file, index=False)
        if not os.path.exists(self.telemetry_file):
            pd.DataFrame(columns=["timestamp", "ticket", "action", "price", "sl", "tp", "pnl_r"]).to_csv(self.telemetry_file, index=False)
            
    def log_event(self, event, details=""):
        df = pd.DataFrame([{
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "details": details
        }])
        df.to_csv(self.log_file, mode='a', header=False, index=False)
        
    def log_trade(self, ticket, action, price, sl, tp, pnl_r=0.0):
        df = pd.DataFrame([{
            "timestamp": datetime.now().isoformat(),
            "ticket": ticket,
            "action": action,
            "price": price,
            "sl": sl,
            "tp": tp,
            "pnl_r": pnl_r
        }])
        df.to_csv(self.telemetry_file, mode='a', header=False, index=False)

    def save_status(self, status_dict):
        with open(os.path.join(self.output_dir, "mt5_demo_status.json"), "w") as f:
            json.dump(status_dict, f, indent=4)
