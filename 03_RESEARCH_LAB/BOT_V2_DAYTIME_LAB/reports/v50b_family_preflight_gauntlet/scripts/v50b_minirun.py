import sys
import pandas as pd
import json
import os
from pathlib import Path

# Add the directory containing the runner to sys.path
runner_dir = Path(__file__).parent.parent.parent.parent
sys.path.append(str(runner_dir))

print(f"Runner Dir: {runner_dir}")
print(f"Current Path: {sys.path}")

try:
    from v50b_family_preflight_runner import V50BRunner
except ImportError as e:
    print(f"Import Error: {e}")
    # Try adding one more level up if needed
    sys.path.append(str(runner_dir.parent))
    from v50b_family_preflight_runner import V50BRunner

def run_mini():
    print("Running V50B Mini-Run...")
    
    mini_config_path = "reports/v50b_family_preflight_gauntlet/V50B_MINIRUN_CONFIG.json"
    
    # Check if config exists, if not create from master
    if not os.path.exists(mini_config_path):
        with open("reports/v50b_family_preflight_gauntlet/V50B_RUN_CONFIG.json", 'r') as f:
            config = json.load(f)
        config["train_months"] = ["2022-05"]
        config["val_months"] = ["2023-01"]
        with open(mini_config_path, 'w') as f:
            json.dump(config, f)
        
    runner = V50BRunner(mini_config_path)
    runner.run_preflight()
    print("Mini-Run Finished.")

if __name__ == "__main__":
    run_mini()
