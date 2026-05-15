import os
import sys
import json
import uuid
import time
import pandas as pd
from pathlib import Path
from datetime import datetime

class SingleWriterRunner:
    def __init__(self):
        self.base_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB\reports\v50b_single_writer_remediation")
        self.lock_file = self.base_dir / "V50B_RUNNER.lock"
        self.run_id = str(uuid.uuid4())[:8]
        self.pid = os.getpid()

    def acquire_lock(self):
        if self.lock_file.exists():
            with open(self.lock_file, "r") as f:
                lock_data = json.load(f)
            print(f"CRITICAL: Lock exists. PID: {lock_data['pid']}, RunID: {lock_data['run_id']}")
            return False
        
        lock_info = {
            "pid": self.pid,
            "run_id": self.run_id,
            "start_time": datetime.now().isoformat()
        }
        with open(self.lock_file, "w") as f:
            json.dump(lock_info, f)
        print(f"Lock acquired. RunID: {self.run_id}")
        return True

    def release_lock(self):
        if self.lock_file.exists():
            os.remove(self.lock_file)
            print("Lock released.")

    def preflight_io(self):
        if not self.acquire_lock():
            print("ABORTED: Could not acquire lock.")
            return
        
        try:
            output_file = self.base_dir / "proof" / f"V50B_SW_IO_PREFLIGHT_{self.run_id}.csv"
            log_file = self.base_dir / "logs" / "V50B_SW_PREFLIGHT_LOG.txt"
            
            # Test append-only write
            test_data = []
            for i in range(5):
                test_data.append({
                    "run_id": self.run_id,
                    "pid": self.pid,
                    "timestamp": datetime.now().isoformat(),
                    "record_type": "IO_TEST_ONLY",
                    "usable_for_research": "NO",
                    "usable_for_ranking": "NO",
                    "data": f"IO_PROBE_{i}"
                })
                time.sleep(0.1)
            
            df = pd.DataFrame(test_data)
            # Append mode with header only if not exists
            df.to_csv(output_file, mode='a', index=False, header=not output_file.exists())
            
            with open(log_file, "a") as f:
                f.write(f"[{datetime.now().isoformat()}] SUCCESS: Preflight IO for RunID {self.run_id}\n")
            
            print(f"Preflight IO Success. Proof: {output_file.name}")
            
        finally:
            self.release_lock()

if __name__ == "__main__":
    runner = SingleWriterRunner()
    if len(sys.argv) > 1 and sys.argv[1] == "preflight_io":
        runner.preflight_io()
    else:
        print("Usage: python v50b_limited_real_runner_single_writer.py preflight_io")
